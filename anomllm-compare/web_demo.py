from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_MODEL = ROOT.parent / "models" / "Qwen3-8B"

TASK_TOKENS = {
    "point_anomaly",
    "range_anomaly",
    "trend_anomaly",
    "frequency_anomaly",
    "general_anomaly",
}


def parse_csv(text: str) -> list[float]:
    values: list[float] = []
    for line_no, line in enumerate(text.strip().splitlines()):
        parts = [p.strip() for p in re.split(r"[,\t;]", line) if p.strip()]
        if not parts:
            continue
        if line_no == 0 and re.search(r"[A-Za-z_\u4e00-\u9fff]", "".join(parts)):
            continue
        raw = parts[1] if len(parts) > 1 else parts[0]
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def median(values: list[float] | np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


def quantile(values: list[float] | np.ndarray, q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def robust_stats(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    med = median(arr)
    mad = median(np.abs(arr - med))
    return med, max(mad, 1e-6)


def moving_average(values: list[float] | np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return arr
    width = max(1, int(width))
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(arr, kernel, mode="same")


def slope(values: list[float] | np.ndarray, start: int, end: int) -> float:
    arr = np.asarray(values, dtype=np.float32)
    start = max(0, min(len(arr), start))
    end = max(start, min(len(arr), end))
    n = end - start
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=np.float32)
    y = arr[start:end]
    x = x - x.mean()
    den = float((x * x).sum())
    return float((x * (y - y.mean())).sum() / (den + 1e-6))


def intervals_from_mask(mask: list[bool] | np.ndarray, gap: int = 0, min_len: int = 1) -> list[dict[str, Any]]:
    flags = [bool(v) for v in mask]
    raw: list[list[int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if (not flag or i == len(flags) - 1) and start is not None:
            end = i + 1 if flag and i == len(flags) - 1 else i
            raw.append([start, end])
            start = None
    if not raw:
        return []
    merged = [raw[0]]
    for s, e in raw[1:]:
        last = merged[-1]
        if s <= last[1] + gap:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return [
        {"start": int(s), "end": int(e)}
        for s, e in merged
        if e - s >= min_len
    ]


def infer_token_from_data(values: list[float]) -> str:
    if len(values) < 12:
        return "general_anomaly"
    arr = np.asarray(values, dtype=np.float32)
    med, mad = robust_stats(arr)
    max_z = float(np.max(np.abs(arr - med) / (1.4826 * mad + 1e-6)))
    if max_z > 5:
        return "point_anomaly"
    n = len(arr)
    tail_start = int(n * 0.8)
    tail_slope = abs(slope(moving_average(arr, 9), tail_start, n))
    if tail_slope > 0.018:
        return "trend_anomaly"
    diffs = np.diff(arr)
    _, diff_mad = robust_stats(diffs)
    mid = diffs[int(n * 0.4) : int(n * 0.72)]
    _, mid_mad = robust_stats(mid)
    if mid_mad > diff_mad * 1.8:
        return "frequency_anomaly"
    return "range_anomaly"


def heuristic_tokens(request: str, values: list[float]) -> dict[str, Any]:
    text = request.lower()
    scores = {
        "point_anomaly": 0.0,
        "range_anomaly": 0.0,
        "trend_anomaly": 0.0,
        "frequency_anomaly": 0.0,
        "general_anomaly": 0.2,
    }
    if re.search(r"趋势|斜率|上升|下降|后半段|trend|slope|drift", text):
        scores["trend_anomaly"] += 1.1
    if re.search(r"尖峰|突刺|孤立|点异常|spike|point|dip", text):
        scores["point_anomaly"] += 1.1
    if re.search(r"越界|均值|漂移|持续|level|range|shift", text):
        scores["range_anomaly"] += 1.0
    if re.search(r"频率|振荡|粗糙|周期|frequency|oscillat|rough", text):
        scores["frequency_anomaly"] += 1.0
    if not request.strip() or scores["general_anomaly"] >= max(scores.values()):
        scores[infer_token_from_data(values)] += 0.7
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    conf = max(0.35, min(0.98, ranked[0][1] / (ranked[0][1] + ranked[1][1] + 0.3)))
    output = "intervals"
    if re.search(r"解释|为什么|原因|explain|why", text):
        output = "explanation"
    elif re.search(r"总结|summary", text):
        output = "summary"
    elif re.search(r"画图|plot|图", text):
        output = "plot_request"
    return {"task_token": ranked[0][0], "output_token": output, "confidence": conf}


def old_constrained(values: list[float]) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    med, mad = robust_stats(arr)
    diffs = np.concatenate([[0.0], np.abs(np.diff(arr))])
    score = np.abs(arr - med) / (1.4826 * mad + 1e-6) + diffs * 0.18
    threshold = quantile(score, 0.965)
    rows = intervals_from_mask(score > threshold, gap=3, min_len=1)[:8]
    for row in rows:
        row["source"] = "candidate_score"
    return rows


def point_policy(values: list[float]) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    med, mad = robust_stats(arr)
    z = np.abs(arr - med) / (1.4826 * mad + 1e-6)
    threshold = max(3.5, quantile(z, 0.985))
    rows = intervals_from_mask(z > threshold, gap=2, min_len=1)
    for row in rows:
        row["start"] = max(0, row["start"] - 1)
        row["end"] = min(len(arr), row["end"] + 1)
        row["source"] = "point_token_policy"
    return rows


def range_policy(values: list[float]) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    smooth = moving_average(arr, 17)
    med, mad = robust_stats(smooth)
    z = np.abs(smooth - med) / (1.4826 * mad + 1e-6)
    rows = intervals_from_mask(z > max(2.2, quantile(z, 0.93)), gap=8, min_len=8)
    for row in rows:
        row["source"] = "range_token_policy"
    return rows


def trend_policy(values: list[float]) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    n = len(arr)
    if n < 12:
        return []
    tail_start = int(n * 0.8)
    width = max(9, int(n * 0.08) | 1)
    smooth = moving_average(arr, width)
    pre = slope(smooth, max(0, tail_start - int(n * 0.25)), tail_start)
    tail = slope(smooth, tail_start, n)
    diffs = np.abs(np.diff(arr))
    noise = median(diffs) or 1e-6
    contrast = abs(tail - pre)
    if not (contrast > noise / 14 or abs(tail) > noise / 10):
        return []
    start = tail_start
    for i in range(tail_start, max(tail_start, n - 6)):
        local = abs(slope(smooth, i, min(n, i + 18)))
        if local > noise / 13:
            start = i
            break
    return [{"start": int(start), "end": int(n), "source": "trend_token_policy"}]


def frequency_policy(values: list[float]) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < 12:
        return []
    diffs = np.abs(np.diff(arr))
    rough = moving_average(diffs, 11)
    med = median(rough)
    mask = np.concatenate([[False], (rough > med * 1.55) & (rough > quantile(rough, 0.82))])
    rows = intervals_from_mask(mask, gap=8, min_len=8)
    for row in rows:
        row["source"] = "frequency_token_policy"
    return rows


def token_policy(values: list[float], tokens: dict[str, Any]) -> list[dict[str, Any]]:
    token = tokens.get("task_token", "general_anomaly")
    if token == "trend_anomaly":
        return trend_policy(values)
    if token == "point_anomaly":
        return point_policy(values)
    if token == "range_anomaly":
        return range_policy(values)
    if token == "frequency_anomaly":
        return frequency_policy(values)
    rows = old_constrained(values)
    for row in rows:
        row["source"] = "general_fallback"
    return rows


def build_task_token_prompt(user_request: str, csv_text: str) -> str:
    preview = "\n".join(csv_text.splitlines()[:80])
    return (
        "Read the user's request and CSV preview. Return only JSON.\n"
        "Allowed task_token: point_anomaly, range_anomaly, trend_anomaly, "
        "frequency_anomaly, general_anomaly.\n"
        "Allowed output_token: intervals, explanation, summary, plot_request.\n"
        'Schema: {"task_token":"trend_anomaly","output_token":"intervals","confidence":0.8}\n\n'
        f"User request:\n{user_request.strip()}\n\nCSV preview:\n{preview}"
    )


def parse_token_response(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    token = str(obj.get("task_token", "general_anomaly")).strip()
    if token not in TASK_TOKENS:
        token = "general_anomaly"
    output = str(obj.get("output_token", "intervals")).strip()
    try:
        confidence = float(obj.get("confidence", 0.5) or 0.5)
    except Exception:
        confidence = 0.5
    return {"task_token": token, "output_token": output, "confidence": confidence}


def extract_tokens_from_free_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    alias = [
        ("trend_anomaly", ["trend_anomaly", "趋势异常", "trend anomaly", "slope"]),
        ("point_anomaly", ["point_anomaly", "点异常", "尖峰", "突刺", "spike"]),
        ("range_anomaly", ["range_anomaly", "越界", "均值漂移", "level shift", "range anomaly"]),
        ("frequency_anomaly", ["frequency_anomaly", "频率异常", "振荡", "frequency anomaly"]),
        ("general_anomaly", ["general_anomaly", "general anomaly"]),
    ]
    best_token = None
    best_score = 0
    for token, keys in alias:
        score = sum(1 for key in keys if key.lower() in lowered)
        if score > best_score:
            best_token = token
            best_score = score
    if best_token is None:
        return None
    output = "intervals"
    if any(key in lowered for key in ["explanation", "解释", "why", "原因"]):
        output = "explanation"
    elif any(key in lowered for key in ["summary", "总结"]):
        output = "summary"
    elif any(key in lowered for key in ["plot", "画图"]):
        output = "plot_request"
    return {"task_token": best_token, "output_token": output, "confidence": min(0.85, 0.45 + 0.2 * best_score)}


class LLMRouter:
    def __init__(self, model_path: Path, quantization: str = "4bit", enabled: bool = True):
        self.model_path = model_path
        self.quantization = quantization
        self.enabled = enabled
        self.lock = threading.Lock()
        self.loaded = False
        self.error: str | None = None
        self.tok = None
        self.model = None
        self.device = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "error": self.error,
        }

    def _load(self) -> None:
        if self.loaded or not self.enabled:
            return
        with self.lock:
            if self.loaded or not self.enabled:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tok = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
                kwargs: dict[str, Any] = {"trust_remote_code": True}
                if self.device.type == "cuda" and self.quantization == "4bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    kwargs["device_map"] = {"": 0}
                elif self.device.type == "cuda" and self.quantization == "8bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    kwargs["device_map"] = {"": 0}
                else:
                    kwargs["dtype"] = torch.bfloat16 if self.device.type == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs).eval()
                if "device_map" not in kwargs:
                    self.model = self.model.to(self.device)
                self.loaded = True
                self.error = None
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.enabled = False

    def infer(self, request: str, csv_text: str) -> tuple[dict[str, Any] | None, str]:
        if not self.enabled:
            return None, "disabled"
        self._load()
        if not self.loaded or self.model is None or self.tok is None or self.device is None:
            return None, self.error or "not_loaded"
        prompt = build_task_token_prompt(request, csv_text)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            text = prompt
        try:
            import torch

            with torch.no_grad():
                enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.tok.model_max_length).to(self.device)
                out = self.model.generate(
                    **enc,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
                gen = out[0, enc.input_ids.shape[1] :]
                raw = self.tok.decode(gen, skip_special_tokens=True)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        tokens = parse_token_response(raw)
        if tokens is None:
            return None, f"unparseable_llm_response: {raw[:160]}"
        tokens["raw_response"] = raw
        return tokens, "llm"


class LMStudioRouter:
    def __init__(self, base_url: str, model: str, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loaded": self.enabled and self.error is None,
            "base_url": self.base_url,
            "model": self.model,
            "error": self.error,
        }

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def infer(self, request: str, csv_text: str) -> tuple[dict[str, Any] | None, str]:
        if not self.enabled:
            return None, "disabled"
        prompt = build_task_token_prompt(request, csv_text)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
        }
        try:
            data = self._post_json("/chat/completions", payload)
            message = data["choices"][0]["message"]
            raw = message.get("content", "") or message.get("reasoning_content", "")
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return None, self.error
        tokens = parse_token_response(raw)
        if tokens is None:
            tokens = extract_tokens_from_free_text(raw)
        if tokens is None:
            self.error = f"unparseable_lmstudio_response: {raw[:160]}"
            return None, self.error
        self.error = None
        tokens["raw_response"] = raw
        return tokens, "lmstudio"


ROUTER: LLMRouter | None = None


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self.send_json({"ok": True, "router": ROUTER.status() if ROUTER else None})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body or "{}")
        except Exception as exc:
            self.send_json({"ok": False, "error": f"bad_request: {exc}"}, status=400)
            return

        request = str(data.get("request", ""))
        csv_text = str(data.get("csv", ""))
        values = parse_csv(csv_text)
        llm_tokens = None
        route_source = "heuristic"
        route_note = ""
        if ROUTER is not None:
            llm_tokens, source = ROUTER.infer(request, csv_text)
            if llm_tokens is not None:
                route_source = source
            else:
                route_note = source
        tokens = llm_tokens or heuristic_tokens(request, values)
        tokens["router_source"] = route_source
        if route_note:
            tokens["router_note"] = route_note
        old_rows = old_constrained(values)
        new_rows = token_policy(values, tokens)
        trace = [
            {
                "step": "USER_REQUEST_ROUTER",
                "router_source": route_source,
                "task_token": tokens.get("task_token"),
                "output_token": tokens.get("output_token"),
                "confidence": round(float(tokens.get("confidence", 0.0)), 3),
                "note": route_note,
            },
            {"step": "OLD_CONSTRAINED_SELECTION", "intervals": [{k: r[k] for k in ("start", "end")} for r in old_rows]},
            {"step": "TOKEN_POLICY", "intervals": [{k: r[k] for k in ("start", "end")} for r in new_rows]},
        ]
        self.send_json(
            {
                "ok": True,
                "values": values,
                "tokens": tokens,
                "oldIntervals": old_rows,
                "newIntervals": new_rows,
                "trace": trace,
                "router": ROUTER.status() if ROUTER else None,
            }
        )

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    global ROUTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--backend", choices=["transformers", "lmstudio", "none"], default="transformers")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    ap.add_argument("--lmstudio-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--lmstudio-model", default="qwen-router")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    if not WEB_ROOT.exists():
        raise SystemExit(f"missing web assets: {WEB_ROOT}")
    if args.no_llm or args.backend == "none":
        ROUTER = LLMRouter(args.model, args.quantization, enabled=False)
    elif args.backend == "lmstudio":
        ROUTER = LMStudioRouter(args.lmstudio_url, args.lmstudio_model, enabled=True)
    else:
        ROUTER = LLMRouter(args.model, args.quantization, enabled=True)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"serving app at http://{args.host}:{args.port}", flush=True)
    print(f"backend={args.backend} router={ROUTER.status()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
