"""Compare pure Qwen vs LAFR+Qwen on AnomLLM synthetic datasets.

Two evaluated methods use the same local HuggingFace causal LM and the same
AnomLLM splits/metrics:

  * llm:      serialized time-series values -> Qwen -> anomaly intervals
  * lafr_llm: serialized values + LAFR candidate intervals/scores -> Qwen

LAFR itself is trained only on the AnomLLM train split for each anomaly type.
The LLM is frozen and used as a JSON interval parser/refiner.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import interpolate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "experiments"))
from lafr_encoder import LAFR  # noqa: E402
import eval_metrics as M  # noqa: E402

LAG_BINS = [0, 1, 2, 3, 4, 6, 8]
DATASET_TASK_TOKENS = {
    "point": "point_anomaly",
    "range": "range_anomaly",
    "trend": "trend_anomaly",
    "freq": "frequency_anomaly",
}
VALID_TASK_TOKENS = set(DATASET_TASK_TOKENS.values()) | {"general_anomaly"}


def load_split(data_root: Path, atype: str, split: str):
    with open(data_root / atype / split / "data.pkl", "rb") as f:
        d = pickle.load(f)
    series = [np.asarray(s, np.float32).reshape(-1, 1) for s in d["series"]]
    gts = [M.interval_to_vector(a[0], len(s)) for s, a in zip(series, d["anom"])]
    return series, gts


def normalize_windows(series_list: list[np.ndarray], window: int) -> np.ndarray:
    wins = []
    for s in series_list:
        nw = len(s) // window
        if nw:
            x = s[: nw * window].reshape(nw, window, 1)
            x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
            wins.append(x.astype(np.float32))
    if not wins:
        raise RuntimeError("no train windows")
    return np.concatenate(wins, 0)


def train_lafr(series_list: list[np.ndarray], window: int, patch: int, device, epochs: int, lr: float):
    X = normalize_windows(series_list, window)
    model = LAFR(
        max_channels=1,
        d_model=64,
        relation_dim=32,
        max_events=6,
        patch=patch,
        n_heads=4,
        n_layers=2,
        lag_bins=LAG_BINS,
    ).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bs = 64
    hist = []
    for ep in range(epochs):
        perm = np.random.permutation(len(X))
        tot = nb = 0
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[perm[i : i + bs]], device=device)
            if len(xb) < 4:
                continue
            loss, _ = model.pretext_losses(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        avg = tot / max(nb, 1)
        hist.append(avg)
        print(f"    lafr epoch {ep + 1}/{epochs} loss={avg:.3f}", flush=True)
    return model.eval(), hist


@torch.no_grad()
def lafr_score(model: LAFR, series: np.ndarray, window: int, patch: int, device) -> np.ndarray:
    length = len(series)
    score = np.zeros(length, dtype=np.float32)
    nw = length // window
    if nw == 0:
        return score
    x = series[: nw * window].reshape(nw, window, 1).astype(np.float32)
    m = x.mean(1, keepdims=True)
    s = x.std(1, keepdims=True) + 1e-6
    xt = torch.tensor((x - m) / s, device=device)
    value = model._patchify(xt)
    bb = model._backbone(value, xt)
    fc = model.forecast_head(bb["z"]).squeeze(-1)
    resid = ((fc[:, :-1] - value[:, 1:]) ** 2).mean(-1).cpu().numpy()
    patches = window // patch
    for w in range(nw):
        for p in range(patches - 1):
            t0 = w * window + (p + 1) * patch
            score[t0 : t0 + patch] = resid[w, p]
    return score


def best_threshold(scores: list[np.ndarray], gts: list[np.ndarray]):
    alls = np.concatenate(scores)
    best_thr, best_f1 = float(alls.max()), -1.0
    for q in np.linspace(80, 99.5, 28):
        thr = np.percentile(alls, q)
        preds = [(sc > thr).astype(int) for sc in scores]
        f1 = M.mean_metrics(gts, preds)["affi f1"]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def _smooth_score(score: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return score.astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(score.astype(np.float32), kernel, mode="same")


def _first_positive(vec: np.ndarray) -> int | None:
    idx = np.where(vec.astype(int) > 0)[0]
    if len(idx) == 0:
        return None
    return int(idx.min())


def apply_trend_policy(score: np.ndarray, policy: dict[str, Any]) -> np.ndarray:
    length = len(score)
    tail_start = int(policy.get("tail_start", int(0.8 * length)))
    tail_start = max(0, min(length - 1, tail_start))
    smooth_width = int(policy.get("smooth_width", 49))
    threshold = float(policy.get("score_threshold", np.inf))
    smoothed = _smooth_score(score, smooth_width)
    tail = smoothed[tail_start:]
    out = np.zeros(length, dtype=int)
    if len(tail) == 0 or float(tail.max()) <= threshold:
        return out
    support = np.where(tail > threshold)[0]
    if len(support) == 0:
        return out
    start = tail_start + int(support.min())
    out[start:] = 1
    return out


def calibrate_trend_policy(
    scores: list[np.ndarray],
    gts: list[np.ndarray],
) -> dict[str, Any]:
    if not scores:
        return {"task_token": "trend_anomaly", "enabled": False, "reason": "no_scores"}
    length = len(scores[0])
    starts = [_first_positive(gt) for gt in gts]
    starts = [s for s in starts if s is not None]
    tail_start = int(np.percentile(starts, 5)) if starts else int(0.8 * length)
    tail_start = max(0, min(length - 1, tail_start))

    best_policy: dict[str, Any] | None = None
    best_f1 = -1.0
    smooth_widths = [13, 25, 49, 73]
    quantiles = np.linspace(80, 99.5, 40)
    for width in smooth_widths:
        smoothed = [_smooth_score(sc, width) for sc in scores]
        tail_max = np.asarray([float(sc[tail_start:].max()) for sc in smoothed], dtype=np.float32)
        for q in quantiles:
            threshold = float(np.percentile(tail_max, q))
            policy = {
                "task_token": "trend_anomaly",
                "enabled": True,
                "tail_start": tail_start,
                "smooth_width": width,
                "score_threshold": threshold,
                "calibration_quantile": float(q),
            }
            preds = [apply_trend_policy(sc, policy) for sc in scores]
            f1 = M.mean_metrics(gts, preds)["affi f1"]
            if f1 > best_f1:
                best_f1 = float(f1)
                best_policy = policy

    assert best_policy is not None
    best_policy["train_affi_f1"] = best_f1
    best_policy["positive_train_series"] = len(starts)
    return best_policy


def vector_to_intervals(vec: np.ndarray) -> list[tuple[int, int]]:
    out = []
    start = None
    for i, v in enumerate(vec.astype(int).tolist()):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(vec)))
    return out


def top_score_intervals(score: np.ndarray, threshold: float, max_intervals: int) -> list[dict[str, Any]]:
    pred = (score > threshold).astype(int)
    intervals = vector_to_intervals(pred)
    rows = []
    for s, e in intervals:
        rows.append(
            {
                "start": int(s),
                "end": int(e),
                "peak_score": float(score[s:e].max()) if e > s else 0.0,
                "mean_score": float(score[s:e].mean()) if e > s else 0.0,
            }
        )
    rows.sort(key=lambda r: r["peak_score"], reverse=True)
    return sorted(rows[:max_intervals], key=lambda r: r["start"])


def scale_series(x: np.ndarray, scale: float) -> np.ndarray:
    flat = x.reshape(-1)
    if scale == 1.0:
        return flat
    n = max(8, int(len(flat) * scale))
    xo = np.linspace(0, 1, len(flat))
    xn = np.linspace(0, 1, n)
    return interpolate.interp1d(xo, flat, kind="linear")(xn).astype(np.float32)


def serialize_csv(x: np.ndarray, scale: float) -> str:
    xs = scale_series(x, scale)
    vals = np.round(xs, 2)
    lines = ["idx,value"]
    lines.extend(f"{i},{v:.2f}" for i, v in enumerate(vals))
    return "\n".join(lines)


def scaled_interval_rows(candidates: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    rows = []
    for c in candidates:
        rows.append(
            {
                "start": int(round(c["start"] * scale)),
                "end": int(round(c["end"] * scale)),
                "peak_score": round(float(c["peak_score"]), 4),
                "mean_score": round(float(c["mean_score"]), 4),
            }
        )
    return rows


def _safe_stat(arr: np.ndarray, fn, default: float = 0.0) -> float:
    if len(arr) == 0:
        return default
    return float(fn(arr))


def _slope(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    x = np.arange(len(arr), dtype=np.float32)
    y = arr.astype(np.float32)
    x = x - x.mean()
    den = float((x * x).sum())
    return float((x * (y - y.mean())).sum() / (den + 1e-6))


def interval_evidence(
    series: np.ndarray,
    score: np.ndarray,
    cand: dict[str, Any],
    threshold: float,
    scale: float,
) -> dict[str, Any]:
    x = series.reshape(-1)
    length = len(x)
    start = max(0, min(length, int(cand["start"])))
    end = max(start + 1, min(length, int(cand["end"])))
    width = end - start
    ctx = int(max(24, min(96, 4 * width)))
    left = x[max(0, start - ctx) : start]
    mid = x[start:end]
    right = x[end : min(length, end + ctx)]
    context = np.concatenate([left, right]) if len(left) + len(right) else x
    ctx_mean = _safe_stat(context, np.mean)
    ctx_std = _safe_stat(context, np.std) + 1e-6
    local_median = _safe_stat(context, np.median)
    mad = _safe_stat(np.abs(context - local_median), np.median) + 1e-6
    peak_idx = int(start + int(np.argmax(score[start:end]))) if end > start else start
    return {
        "id": cand.get("id", ""),
        "start": start,
        "end": end,
        "display_start": int(round(start * scale)),
        "display_end": int(round(end * scale)),
        "width": width,
        "peak_at": peak_idx,
        "peak_score": round(float(score[start:end].max()), 4),
        "mean_score": round(float(score[start:end].mean()), 4),
        "score_over_threshold": round(float(score[start:end].max() / (threshold + 1e-8)), 3),
        "value_z_peak": round(float(np.max(np.abs(mid - ctx_mean)) / ctx_std), 3),
        "value_mad_peak": round(float(np.max(np.abs(mid - local_median)) / mad), 3),
        "mean_shift": round(float((mid.mean() - ctx_mean) / ctx_std), 3),
        "left_mean": round(_safe_stat(left, np.mean), 3),
        "inside_mean": round(_safe_stat(mid, np.mean), 3),
        "right_mean": round(_safe_stat(right, np.mean), 3),
        "left_slope": round(_slope(left), 4),
        "inside_slope": round(_slope(mid), 4),
        "right_slope": round(_slope(right), 4),
        "inside_diff_std": round(_safe_stat(np.diff(mid), np.std), 4),
        "context_diff_std": round(_safe_stat(np.diff(context), np.std), 4),
    }


def build_evidence_rows(
    series: np.ndarray,
    score: np.ndarray,
    candidates: list[dict[str, Any]],
    threshold: float,
    scale: float,
) -> list[dict[str, Any]]:
    rows = []
    for idx, cand in enumerate(candidates, start=1):
        c = dict(cand)
        c["id"] = f"C{idx}"
        rows.append(interval_evidence(series, score, c, threshold, scale))
    return rows


def build_edit_prompt(
    x: np.ndarray,
    atype: str,
    scale: float,
    evidence_rows: list[dict[str, Any]],
    threshold: float,
) -> str:
    n_scaled = max(8, int(len(x) * scale))
    type_guidance = {
        "point": "point anomalies are isolated spikes/dips; prefer narrow edits around peak_at.",
        "range": "range anomalies are sustained level shifts; prefer contiguous intervals and boundary trimming.",
        "trend": "trend anomalies show slope/regime changes; use slope and mean-shift evidence.",
        "freq": "frequency anomalies change oscillation/roughness; use inside_diff_std versus context_diff_std.",
    }.get(atype, "use the dataset type and event evidence.")
    return (
        "You are the reasoning part of an anomaly detector. LAFR already produced a calibrated "
        "full-resolution anomaly score and candidate events. Your job is NOT to redraw the whole "
        "time axis. Instead, inspect the compact series and the event evidence, then propose edits "
        "that combine LAFR's temporal localization with type-specific reasoning.\n"
        f"Original series length is {len(x)}. Compact CSV display coordinates are 0..{n_scaled - 1}; "
        "event evidence uses ORIGINAL coordinates 0..999. All edit start/end values must use ORIGINAL coordinates.\n"
        "Fusion policy: the executor starts from every above-threshold LAFR interval. Candidates you do not "
        "mention are kept. Emit only useful edits: DROP clear false positives, SHIFT/EXPAND/SHRINK/MERGE "
        "bad boundaries, or ADD clearly missed events.\n"
        f"Dataset type: {atype}. Guidance: {type_guidance}\n"
        f"LAFR score threshold: {threshold:.6g}\n\n"
        f"Compact series:\n{serialize_csv(x, scale)}\n\n"
        f"Candidate event evidence:\n{json.dumps(evidence_rows, ensure_ascii=False)}\n\n"
        "Return only valid JSON with a single top-level key named ops and no extra text. "
        'Each op item must contain keys "op", "ids", "start", "end", and "confidence". '
        "Use [] when no edit is useful.\n"
        "Allowed op values: CONFIRM, DROP, SHIFT, EXPAND, SHRINK, MERGE, SPLIT, ADD.\n"
        "Use ADD only for a clearly missed event visible in the compact series. Use MERGE when adjacent "
        "events form one anomaly. Use SPLIT when one candidate covers two separate events."
    )


def build_prompt(
    x: np.ndarray,
    atype: str,
    scale: float,
    method: str,
    candidates: list[dict[str, Any]] | None = None,
) -> str:
    n_scaled = max(8, int(len(x) * scale))
    common = (
        "You are detecting anomalies in a univariate time series.\n"
        f"The data is a scaled representation of an original length-{len(x)} series. "
        f"The displayed x-axis coordinates run from 0 to {n_scaled - 1}. "
        "Report anomaly ranges using displayed coordinates only.\n"
        f"Dataset type hint: {atype} anomaly.\n\n"
        f"{serialize_csv(x, scale)}\n\n"
    )
    if method == "lafr_llm":
        common += (
            "A separate self-supervised LAFR temporal encoder analyzed the full-resolution series "
            "and proposed candidate anomalous ranges below. Use these candidates as evidence, "
            "but you may merge, trim, drop, or keep them based on the numeric series.\n"
            f"LAFR candidates: {json.dumps(scaled_interval_rows(candidates or [], scale))}\n\n"
        )
    return (
        common
        + "Return only a valid JSON list of anomaly intervals, with no extra text. "
        + 'Each item must have integer keys "start" and "end" in displayed coordinates. '
        + "Choose coordinates from this series only. If there are no anomalies, return []."
    )


def build_task_token_prompt(user_request: str, csv_text: str) -> str:
    csv_preview = "\n".join(csv_text.splitlines()[:80])
    return (
        "Read the user's request and the CSV preview. Decide the internal task token that should "
        "control the detector. Return only valid JSON and no extra text.\n"
        "Allowed task_token values: point_anomaly, range_anomaly, trend_anomaly, "
        "frequency_anomaly, general_anomaly.\n"
        'Allowed output_token values: intervals, explanation, summary, plot_request.\n'
        'Schema: {"task_token":"trend_anomaly","output_token":"intervals","confidence":0.8}\n\n'
        f"User request:\n{user_request.strip()}\n\n"
        f"CSV preview:\n{csv_preview}"
    )


def extract_internal_tokens(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            token = str(obj.get("task_token", "general_anomaly")).strip()
            output = str(obj.get("output_token", "intervals")).strip()
            conf = float(obj.get("confidence", 0.5) or 0.5)
            if token not in VALID_TASK_TOKENS:
                token = "general_anomaly"
            return {"task_token": token, "output_token": output, "confidence": conf}
        except Exception:
            pass
    lowered = text.lower()
    for token in sorted(VALID_TASK_TOKENS):
        if token in lowered:
            return {"task_token": token, "output_token": "intervals", "confidence": 0.5}
    return {"task_token": "general_anomaly", "output_token": "intervals", "confidence": 0.0}


def dataset_internal_tokens(atype: str) -> dict[str, Any]:
    return {
        "task_token": DATASET_TASK_TOKENS.get(atype, "general_anomaly"),
        "output_token": "intervals",
        "confidence": 1.0,
        "source": "dataset_type",
    }


def infer_internal_tokens_from_user_request(
    model,
    tok,
    user_request: str,
    csv_text: str,
    device,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    prompt = build_task_token_prompt(user_request, csv_text)
    text = generate_one(model, tok, prompt, device, max_new_tokens)
    tokens = extract_internal_tokens(text)
    tokens["raw_response"] = text
    tokens["source"] = "llm_request_router"
    return tokens


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
    pairs = re.findall(r"start[^0-9-]*(-?\d+).*?end[^0-9-]*(-?\d+)", text, flags=re.I | re.S)
    return [{"start": int(s), "end": int(e)} for s, e in pairs]


def extract_edit_ops(text: str) -> list[dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            ops = obj.get("ops", [])
            if isinstance(ops, list):
                return [op for op in ops if isinstance(op, dict)]
        except Exception:
            pass
    # Fallback: accept a bare JSON list of ops.
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return [op for op in obj if isinstance(op, dict)]
        except Exception:
            pass
    # Fallback for truncated generations: salvage complete op objects that were
    # emitted before the cutoff.
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        chunk = m.group(0)
        if '"op"' not in chunk and "'op'" not in chunk:
            continue
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if isinstance(obj, dict) and "op" in obj:
            out.append(obj)
    if out:
        return out
    return []


def _clip_interval(start: Any, end: Any, length: int) -> tuple[int, int] | None:
    try:
        s = int(round(float(start)))
        e = int(round(float(end)))
    except Exception:
        return None
    s = max(0, min(length, s))
    e = max(s, min(length, e))
    if e <= s:
        return None
    return s, e


def _merge_intervals(intervals: list[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    if not intervals:
        return []
    rows = sorted(intervals)
    merged = [rows[0]]
    for s, e in rows[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _overlap_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _subtract_interval(
    intervals: list[tuple[int, int]],
    rem: tuple[int, int],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    rs, re = rem
    for s, e in intervals:
        if re <= s or rs >= e:
            out.append((s, e))
            continue
        if s < rs:
            out.append((s, max(s, rs)))
        if re < e:
            out.append((min(e, re), e))
    return [(s, e) for s, e in out if e > s]


def _replace_overlapping_intervals(
    intervals: list[tuple[int, int]],
    target: tuple[int, int],
    replacement: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    out = []
    replaced = False
    for iv in intervals:
        if _overlap_len(iv, target) > 0:
            replaced = True
        else:
            out.append(iv)
    out.append(replacement)
    return out, replaced


def execute_edit_ops(
    ops: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    score: np.ndarray,
    threshold: float,
    atype: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    length = len(score)
    by_id = {str(r["id"]): r for r in evidence_rows}
    intervals: list[tuple[int, int]] = vector_to_intervals((score > threshold).astype(int))
    trace: list[dict[str, Any]] = []
    trace.append({"op": "BASE_LAFR", "applied": True, "n": len(intervals)})

    def ids_interval(ids: list[Any]) -> tuple[int, int] | None:
        rows = [by_id.get(str(i)) for i in ids]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None
        return min(int(r["start"]) for r in rows), max(int(r["end"]) for r in rows)

    def ids_rows(ids: list[Any]) -> list[dict[str, Any]]:
        return [r for r in (by_id.get(str(i)) for i in ids) if r is not None]

    def snap_to_score_support(
        interval: tuple[int, int],
        op: str,
    ) -> tuple[int, int]:
        s, e = interval
        pad = max(6, int(0.5 * (e - s)))
        lo, hi = max(0, s - pad), min(length, e + pad)
        support = np.where(score[lo:hi] > threshold)[0]
        if not len(support):
            return s, e
        ss = lo + int(support.min())
        ee = lo + int(support.max()) + 1
        if op in {"MERGE", "EXPAND"}:
            return min(s, ss), max(e, ee)
        if op in {"CONFIRM", "SHIFT", "SHRINK"}:
            return ss, ee
        return s, e

    for op0 in ops:
        op = str(op0.get("op", "")).upper()
        conf = float(op0.get("confidence", 0.5) or 0.5)
        ids = op0.get("ids", [])
        if isinstance(ids, str):
            ids = [ids]
        if conf < 0.25:
            trace.append({"op": op, "applied": False, "reason": "low_conf", "raw": op0})
            continue

        base = ids_interval(ids)
        if op == "DROP":
            rows = ids_rows(ids)
            max_ratio = max([float(r.get("score_over_threshold", 0.0)) for r in rows] or [0.0])
            if base is None:
                trace.append({"op": op, "applied": False, "reason": "drop_without_candidate", "raw": op0})
                continue
            # A language model can veto weak LAFR support, but strong score evidence is kept
            # unless the model is almost certain. This prevents sparse LLM outputs from erasing
            # calibrated detector recall.
            if conf >= 0.80 and (max_ratio < 2.5 or conf >= 0.97):
                before = len(intervals)
                intervals = _subtract_interval(intervals, base)
                trace.append(
                    {
                        "op": op,
                        "applied": True,
                        "interval": [int(base[0]), int(base[1])],
                        "removed_parts": before - len(intervals),
                        "max_score_over_threshold": max_ratio,
                        "raw": op0,
                    }
                )
            else:
                trace.append(
                    {
                        "op": op,
                        "applied": False,
                        "reason": "strong_lafr_support_or_low_conf",
                        "max_score_over_threshold": max_ratio,
                        "raw": op0,
                    }
                )
            continue

        if op in {"CONFIRM", "MERGE"} and base is not None:
            interval = base
        elif op in {"SHIFT", "EXPAND", "SHRINK", "SPLIT", "ADD"}:
            interval = _clip_interval(op0.get("start"), op0.get("end"), length)
            if interval is None and base is not None:
                interval = base
        else:
            interval = _clip_interval(op0.get("start"), op0.get("end"), length)

        if interval is None:
            trace.append({"op": op, "applied": False, "reason": "no_valid_interval", "raw": op0})
            continue

        s, e = interval
        local_peak = float(score[s:e].max()) if e > s else 0.0
        # LLM may add/edit, but the full-resolution LAFR score remains the grounding signal.
        # Low-score ADDs are accepted only with high confidence, because otherwise the LLM tends
        # to hallucinate visually plausible ranges in compacted CSV text.
        min_ratio = 0.45 if op == "ADD" else 0.25
        if local_peak < min_ratio * threshold and conf < 0.85:
            trace.append(
                {
                    "op": op,
                    "applied": False,
                    "reason": "weak_lafr_support",
                    "peak": local_peak,
                    "raw": op0,
                }
            )
            continue

        s, e = snap_to_score_support((s, e), op)
        edit_interval = (int(s), int(e))
        if op in {"SHIFT", "EXPAND", "SHRINK", "MERGE", "SPLIT"}:
            target = base if base is not None else edit_interval
            intervals, replaced = _replace_overlapping_intervals(intervals, target, edit_interval)
            action = "replace" if replaced else "append"
        else:
            intervals.append(edit_interval)
            action = "append"
        trace.append(
            {
                "op": op,
                "applied": True,
                "action": action,
                "interval": [int(s), int(e)],
                "raw": op0,
            }
        )

    gap = 12 if atype in {"range", "trend", "freq"} else 0
    intervals = _merge_intervals(intervals, gap=gap)
    out = np.zeros(length, dtype=int)
    for s, e in intervals:
        out[s:e] = 1
    return out, trace


def fuse_with_llm_prior(
    score: np.ndarray,
    threshold: float,
    base_pred: np.ndarray,
    llm_preds: list[np.ndarray],
    atype: str,
    internal_tokens: dict[str, Any],
    task_policy: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    length = len(score)
    task_token = str(internal_tokens.get("task_token", DATASET_TASK_TOKENS.get(atype, "general_anomaly")))
    if task_token == "trend_anomaly" and task_policy and task_policy.get("enabled", False):
        pred = apply_trend_policy(score, task_policy)
        trace = [
            {
                "op": "TREND_POLICY",
                "applied": True,
                "task_token": task_token,
                "tail_start": int(task_policy.get("tail_start", -1)),
                "smooth_width": int(task_policy.get("smooth_width", -1)),
                "score_threshold": float(task_policy.get("score_threshold", float("nan"))),
                "n_intervals": len(vector_to_intervals(pred)),
            }
        ]
        return pred, trace

    usable = [p.astype(np.float32).reshape(-1)[:length] for p in llm_preds if len(p)]
    if not usable:
        return base_pred.copy(), [{"op": "LLM_PRIOR_FUSION", "applied": False, "reason": "no_llm_pred"}]

    prior = np.mean(np.stack(usable, axis=0), axis=0)
    width = {"point": 5, "range": 25, "trend": 31, "freq": 31}.get(atype, 21)
    if width > 1:
        kernel = np.ones(width, dtype=np.float32) / float(width)
        prior = np.convolve(prior, kernel, mode="same")

    floor_ratio = {"point": 0.82, "range": 0.70, "trend": 0.68, "freq": 0.68}.get(atype, 0.70)
    prior_cut = {"point": 0.20, "range": 0.18, "trend": 0.16, "freq": 0.16}.get(atype, 0.18)
    base_bool = base_pred.astype(bool)
    llm_supported = (score >= floor_ratio * threshold) & (prior >= prior_cut)
    fused = base_bool | llm_supported

    gap = 12 if atype in {"range", "trend", "freq"} else 0
    intervals = _merge_intervals(vector_to_intervals(fused.astype(int)), gap=gap)
    out = np.zeros(length, dtype=int)
    for s, e in intervals:
        out[s:e] = 1

    added = int(np.logical_and(out == 1, ~base_bool).sum())
    trace = [
        {
            "op": "LLM_PRIOR_FUSION",
            "applied": True,
            "floor_ratio": floor_ratio,
            "prior_cut": prior_cut,
            "smooth_width": width,
            "added_points": added,
            "n_intervals": len(intervals),
        }
    ]
    return out, trace


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
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
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


def load_llm(model_name: str, device, quantization: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device.type == "cuda" and quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    elif device.type == "cuda" and quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    if "device_map" not in kwargs:
        model = model.to(device)
    return tok, model


def aggregate_mean(rows: dict[str, Any], method: str, types: list[str]) -> dict[str, float]:
    keys = ["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"]
    return {k: float(np.mean([rows[t][method]["metrics"][k] for t in types])) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--types", default="point,range,trend,freq")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--num", type=int, default=8, help="eval examples per anomaly type")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--lafr-epochs", type=int, default=4)
    ap.add_argument("--lafr-lr", type=float, default=2e-3)
    ap.add_argument("--lafr-max-candidates", type=int, default=8)
    ap.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--output", type=Path, default=ROOT / "lafr_qwen_compare_results.json")
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    atypes = [t.strip() for t in args.types.split(",") if t.strip()]

    print(f"[llm] loading {args.model} quantization={args.quantization} device={device}", flush=True)
    tok, llm = load_llm(args.model, device, args.quantization)

    report: dict[str, Any] = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "results": {},
    }

    for atype in atypes:
        print(f"\n===== {atype} =====", flush=True)
        tr_series, tr_gts = load_split(args.data_root, atype, "train")
        ev_series, ev_gts = load_split(args.data_root, atype, "eval")
        n = min(args.num, len(ev_series))

        print("[lafr] training detector and calibrating threshold ...", flush=True)
        lafr, hist = train_lafr(tr_series, args.window, args.patch, device, args.lafr_epochs, args.lafr_lr)
        tr_scores = [lafr_score(lafr, s, args.window, args.patch, device) for s in tr_series]
        thr, tr_affi = best_threshold(tr_scores, tr_gts)
        ev_scores = [lafr_score(lafr, ev_series[i], args.window, args.patch, device) for i in range(n)]
        lafr_preds = [(sc > thr).astype(int) for sc in ev_scores]
        lafr_metrics = M.mean_metrics(ev_gts[:n], lafr_preds)
        internal_tokens = dataset_internal_tokens(atype)
        task_policy: dict[str, Any] | None = None
        if internal_tokens["task_token"] == "trend_anomaly":
            task_policy = calibrate_trend_policy(tr_scores, tr_gts)
            print(
                "[router] internal task_token=trend_anomaly "
                f"tail_start={task_policy.get('tail_start')} "
                f"smooth={task_policy.get('smooth_width')} "
                f"train_affi_f1={task_policy.get('train_affi_f1', 0.0):.3f}",
                flush=True,
            )

        method_preds: dict[str, list[np.ndarray]] = {"llm": [], "lafr_llm": [], "lafr_llm_edit": []}
        raw: dict[str, list[Any]] = {"llm": [], "lafr_llm": [], "lafr_llm_edit": []}
        for i in range(n):
            candidates = top_score_intervals(ev_scores[i], thr, args.lafr_max_candidates)
            sample_llm_preds: dict[str, np.ndarray] = {}
            for method in ("llm", "lafr_llm"):
                prompt = build_prompt(ev_series[i], atype, args.scale, method, candidates)
                text = generate_one(llm, tok, prompt, device, args.max_new_tokens)
                intervals = extract_json_list(text)
                pred = intervals_to_vector(intervals, len(ev_series[i]), args.scale)
                sample_llm_preds[method] = pred
                method_preds[method].append(pred)
                if args.save_raw:
                    raw[method].append(
                        {
                            "i": i,
                            "response": text,
                            "parsed": intervals,
                            "lafr_candidates": candidates,
                        }
                    )
            evidence = build_evidence_rows(ev_series[i], ev_scores[i], candidates, thr, args.scale)
            edit_prompt = build_edit_prompt(ev_series[i], atype, args.scale, evidence, thr)
            edit_text = generate_one(llm, tok, edit_prompt, device, args.max_new_tokens)
            ops = extract_edit_ops(edit_text)
            edit_base_pred, edit_trace = execute_edit_ops(ops, evidence, ev_scores[i], thr, atype)
            edit_pred, prior_trace = fuse_with_llm_prior(
                ev_scores[i],
                thr,
                edit_base_pred,
                [sample_llm_preds["llm"], sample_llm_preds["lafr_llm"]],
                atype,
                internal_tokens,
                task_policy,
            )
            edit_trace.extend(prior_trace)
            method_preds["lafr_llm_edit"].append(edit_pred)
            if args.save_raw:
                raw["lafr_llm_edit"].append(
                    {
                        "i": i,
                        "response": edit_text,
                        "ops": ops,
                        "trace": edit_trace,
                        "evidence": evidence,
                    }
                )
            if (i + 1) % 2 == 0 or i + 1 == n:
                llm_f1 = M.mean_metrics(ev_gts[: i + 1], method_preds["llm"])["affi f1"]
                lq_f1 = M.mean_metrics(ev_gts[: i + 1], method_preds["lafr_llm"])["affi f1"]
                ed_f1 = M.mean_metrics(ev_gts[: i + 1], method_preds["lafr_llm_edit"])["affi f1"]
                print(
                    f"  {i + 1:3d}/{n}  affi-F1 llm={llm_f1:.3f} "
                    f"lafr+llm={lq_f1:.3f} edit={ed_f1:.3f}",
                    flush=True,
                )

        row = {
            "lafr_detector": {"n": n, "metrics": lafr_metrics, "threshold": thr, "train_affi_f1": tr_affi},
            "llm": {"n": n, "metrics": M.mean_metrics(ev_gts[:n], method_preds["llm"])},
            "lafr_llm": {"n": n, "metrics": M.mean_metrics(ev_gts[:n], method_preds["lafr_llm"])},
            "lafr_llm_edit": {"n": n, "metrics": M.mean_metrics(ev_gts[:n], method_preds["lafr_llm_edit"])},
            "lafr_train_loss": hist,
            "internal_tokens": internal_tokens,
        }
        if task_policy:
            row["task_policy"] = task_policy
        if args.save_raw:
            row["raw"] = raw
        report["results"][atype] = row
        args.output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

        print(
            "  final affi-F1 "
            f"LAFR={lafr_metrics['affi f1']:.3f}  "
            f"LLM={row['llm']['metrics']['affi f1']:.3f}  "
            f"LAFR+LLM={row['lafr_llm']['metrics']['affi f1']:.3f}  "
            f"EDIT={row['lafr_llm_edit']['metrics']['affi f1']:.3f}",
            flush=True,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n================ AnomLLM data: Qwen vs LAFR+Qwen ================")
    print(f"{'type':8s}{'LLM':>12s}{'LAFR+LLM':>12s}{'EDIT':>12s}{'edit-llm':>12s}{'LAFR det.':>12s}")
    for t in atypes:
        llm_f = report["results"][t]["llm"]["metrics"]["affi f1"]
        lq_f = report["results"][t]["lafr_llm"]["metrics"]["affi f1"]
        ed_f = report["results"][t]["lafr_llm_edit"]["metrics"]["affi f1"]
        lf_f = report["results"][t]["lafr_detector"]["metrics"]["affi f1"]
        print(f"{t:8s}{llm_f:>12.3f}{lq_f:>12.3f}{ed_f:>12.3f}{ed_f - llm_f:>+12.3f}{lf_f:>12.3f}")
    m_llm = aggregate_mean(report["results"], "llm", atypes)["affi f1"]
    m_lq = aggregate_mean(report["results"], "lafr_llm", atypes)["affi f1"]
    m_ed = aggregate_mean(report["results"], "lafr_llm_edit", atypes)["affi f1"]
    m_lf = aggregate_mean(report["results"], "lafr_detector", atypes)["affi f1"]
    print(f"{'MEAN':8s}{m_llm:>12.3f}{m_lq:>12.3f}{m_ed:>12.3f}{m_ed - m_llm:>+12.3f}{m_lf:>12.3f}")
    report["summary"] = {
        "mean_llm_affi_f1": m_llm,
        "mean_lafr_llm_affi_f1": m_lq,
        "mean_lafr_llm_edit_affi_f1": m_ed,
        "mean_lafr_detector_affi_f1": m_lf,
        "mean_delta_lafr_llm_minus_llm": m_lq - m_llm,
        "mean_delta_lafr_llm_edit_minus_llm": m_ed - m_llm,
        "mean_delta_lafr_llm_edit_minus_lafr_detector": m_ed - m_lf,
    }
    args.output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
