"""Local HuggingFace LLM evaluation on AnomLLM synthetic anomaly data.

This is for the question: can an LLM directly read serialized time series and
emit anomaly intervals on the AnomLLM dataset? It does not call OpenAI APIs.

Metric: same affiliation-F1 implementation used by AnomLLM, via eval_metrics.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import interpolate
from transformers import AutoModelForCausalLM, AutoTokenizer

import eval_metrics as M


def load_eval(data_root: Path, atype: str):
    with open(data_root / atype / "eval" / "data.pkl", "rb") as f:
        d = pickle.load(f)
    series = [np.asarray(s, np.float32).reshape(-1) for s in d["series"]]
    gts = [M.interval_to_vector(a[0], len(s)) for s, a in zip(series, d["anom"])]
    return series, gts


def scale_series(x: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return x
    n = int(len(x) * scale)
    xo = np.linspace(0, 1, len(x))
    xn = np.linspace(0, 1, n)
    return interpolate.interp1d(xo, x, kind="linear")(xn).astype(np.float32)


def serialize_csv(x: np.ndarray, scale: float) -> str:
    xs = scale_series(x, scale)
    vals = np.round(xs, 2)
    lines = ["idx,value"]
    lines.extend(f"{i},{v:.2f}" for i, v in enumerate(vals))
    return "\n".join(lines)


def build_prompt(x: np.ndarray, atype: str, scale: float) -> str:
    n_scaled = int(len(x) * scale)
    return (
        "You are detecting anomalies in a univariate time series.\n"
        f"The data below is a scaled representation of an original length-{len(x)} series. "
        f"The displayed x-axis coordinates run from 0 to {n_scaled - 1}. "
        "Report anomaly ranges using the displayed coordinates only.\n"
        f"Dataset type hint: {atype} anomaly.\n\n"
        f"{serialize_csv(x, scale)}\n\n"
        "Return only a valid JSON list of anomaly intervals. "
        "Use this exact format and no extra text:\n"
        '[{"start": 12, "end": 25}, {"start": 80, "end": 93}]\n'
        "If there are no anomalies, return []."
    )


def extract_json_list(text: str) -> list[dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        chunk = text[start : end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                out = []
                for item in obj:
                    if isinstance(item, dict) and "start" in item and "end" in item:
                        out.append({"start": item["start"], "end": item["end"]})
                return out
        except Exception:
            pass

    # Robust fallback for malformed JSON: start/end pairs anywhere in the text.
    pairs = re.findall(r"start[^0-9-]*(-?\d+).*?end[^0-9-]*(-?\d+)", text, flags=re.I | re.S)
    return [{"start": int(s), "end": int(e)} for s, e in pairs]


def intervals_to_vector(intervals: list[dict[str, Any]], length: int, scale: float) -> np.ndarray:
    out = np.zeros(length, dtype=int)
    for item in intervals:
        try:
            s = int(round(float(item["start"]) / scale))
            e = int(round(float(item["end"]) / scale))
        except Exception:
            continue
        s = max(0, min(length, s))
        e = max(s, min(length, e))
        if e > s:
            out[s:e] = 1
    return out


@torch.no_grad()
def generate_one(model, tok, prompt: str, device, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    enc = tok(text, return_tensors="pt", truncation=True, max_length=tok.model_max_length).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    gen = out[0, enc.input_ids.shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("anomllm-compare/data"))
    ap.add_argument("--types", default="point,range,trend,freq")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=Path("anomllm-compare/llm_local_results.json"))
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "results": {},
    }
    for atype in args.types.split(","):
        series, gts = load_eval(args.data_root, atype)
        n = min(args.num, len(series))
        preds = []
        raw = []
        print(f"\n===== {atype} local LLM ({args.model}), n={n} =====", flush=True)
        for i in range(n):
            prompt = build_prompt(series[i], atype, args.scale)
            text = generate_one(model, tok, prompt, device, args.max_new_tokens)
            intervals = extract_json_list(text)
            pred = intervals_to_vector(intervals, len(series[i]), args.scale)
            preds.append(pred)
            if args.save_raw:
                raw.append({"i": i, "response": text, "parsed": intervals})
            if (i + 1) % 5 == 0 or i + 1 == n:
                partial = M.mean_metrics(gts[: i + 1], preds)["affi f1"]
                print(f"  {i + 1:3d}/{n}  running affi-F1={partial:.3f}", flush=True)

        metrics = M.mean_metrics(gts[:n], preds)
        report["results"][atype] = {"n": n, "metrics": metrics}
        if args.save_raw:
            report["results"][atype]["raw"] = raw
        print(f"  final affi-F1={metrics['affi f1']:.3f}  point-F1={metrics['f1']:.3f}", flush=True)

    args.output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
