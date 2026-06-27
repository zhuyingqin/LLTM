"""LAFR training + mechanism evaluation on CARE-based ground-truth data.

CARE has NO structural labels (no per-pair lag, no change-point truth) and the
existing retrieval/IoU eval is degenerate (random tokens win). To actually measure
whether LAFR's MECHANISMS work, we build evaluation data ON TOP OF real CARE signals
by injecting *known* structure, so we have ground truth:

  * change-point localization : take a real CARE window, inject a regime shift at a
    KNOWN time t* (scale+offset a subset of channels for t >= t*). GT = t*.
  * lead/lag recovery         : take a real CARE channel as driver, set a responder
    channel = driver delayed by a KNOWN lag tau (+ noise). GT = (0 leads 1 by tau).
  * self-sup quality          : forecast / masked-recon error on held-out real windows.
  * instance retrieval        : two augmented views of each held-out window; recall@1.

Each learned metric is reported next to a CLASSIC baseline (max-jump change detector;
cross-correlation lag estimator) so "does the learned mechanism match/beat the obvious
estimator" is answerable. Train/val split is FILE-disjoint to avoid leakage.

Self-contained: only depends on `lafr_encoder.LAFR`; does not import the old pipeline.

Usage:
  python experiments/lafr_eval.py --max-files 16 --epochs 20 --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lafr_encoder import LAFR  # noqa: E402


SEMANTIC_FIRST = [
    "wind_speed_3_avg", "wind_speed_4_avg",
    "power_29_avg", "power_30_avg",
    "reactive_power_27_avg", "reactive_power_28_avg",
]


# ---------------------------------------------------------------------------
# CARE loading (self-contained, fixed channel order across files)
# ---------------------------------------------------------------------------
def discover_csvs(data_root: Path) -> List[Path]:
    return sorted(p for p in data_root.rglob("*.csv")
                  if "/datasets/" in p.as_posix() and p.name[0].isdigit())


def avg_columns(path: Path) -> List[str]:
    head = pd.read_csv(path, sep=";", nrows=0)
    return [c for c in head.columns if c.endswith("_avg")]


def choose_channels(files: Sequence[Path], n_channels: int) -> List[str]:
    """Fixed, file-consistent channel order: semantic columns first, then sensor_*_avg,
    restricted to columns present in ALL sampled files (so channel index = same sensor)."""
    common = None
    for f in files:
        cols = set(avg_columns(f))
        common = cols if common is None else (common & cols)
    common = common or set()
    ordered = [c for c in SEMANTIC_FIRST if c in common]
    sensors = sorted([c for c in common if c.startswith("sensor_")],
                     key=lambda c: int(c.split("_")[1]))
    ordered += [c for c in sensors if c not in ordered]
    if len(ordered) < n_channels:
        raise RuntimeError(f"only {len(ordered)} common avg channels (<{n_channels})")
    return ordered[:n_channels]


def load_windows(files: Sequence[Path], channels: Sequence[str], window: int,
                 max_rows: int, stride: int) -> np.ndarray:
    """Return (N, T, C) raw windows (un-normalized) from the given files."""
    out: List[np.ndarray] = []
    for f in files:
        df = pd.read_csv(f, sep=";", usecols=list(channels), nrows=max_rows)
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        arr = pd.DataFrame(arr).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        for start in range(0, len(arr) - window + 1, stride):
            out.append(arr[start:start + window])
    if not out:
        raise RuntimeError("no windows loaded")
    return np.stack(out)  # (N, T, C)


def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(0, keepdims=True)
    std = flat.std(0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None]) / std[None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Ground-truth injection on real CARE windows
# ---------------------------------------------------------------------------
def make_changepoint(win: np.ndarray, rng: np.random.Generator, patch: int,
                     gain: float = 1.8, shift: float = 1.5) -> Tuple[np.ndarray, int]:
    """Inject a regime shift at a known patch boundary t* into a real window.
    Returns (modified window, GT change patch index)."""
    T, C = win.shape
    P = T // patch
    gp = int(rng.integers(max(1, int(P * 0.25)), max(2, int(P * 0.75))))  # GT change patch
    t_star = gp * patch
    w = win.copy()
    k = max(2, C // 3)
    subset = rng.choice(C, size=k, replace=False)
    w[t_star:, subset] = w[t_star:, subset] * gain + shift
    return w, gp


def make_lag(win: np.ndarray, rng: np.random.Generator, patch: int,
             lag_patches_choices: Sequence[int]) -> Tuple[np.ndarray, int]:
    """Channel 0 = a real driver; channel 1 = driver delayed by tau patches (+noise).
    Returns (window, tau in patches). Convention: channel 0 LEADS channel 1."""
    T, C = win.shape
    tau_p = int(rng.choice(lag_patches_choices))
    lag = tau_p * patch
    w = win.copy()
    # use the LIVELIEST channel as the driver (anonymized sensor_* slot 0 is often
    # near-constant, which would make the lag undetectable); swap it into slot 0 and
    # unit-normalize so the injected structure is channel-choice agnostic.
    drv_idx = int(np.argmax(w.std(axis=0)))
    if drv_idx != 0:
        w[:, [0, drv_idx]] = w[:, [drv_idx, 0]]
    driver = w[:, 0].copy()
    driver = (driver - driver.mean()) / (driver.std() + 1e-6)
    w[:, 0] = driver
    resp = np.zeros_like(driver)
    resp[lag:] = driver[:T - lag]
    resp += 0.05 * rng.standard_normal(T).astype(np.float32)
    w[:, 1] = resp
    return w, tau_p


# ---------------------------------------------------------------------------
# Classic baselines (structure-blind / textbook estimators)
# ---------------------------------------------------------------------------
def baseline_changepoint(win: np.ndarray, patch: int) -> int:
    """Predicted change patch = largest patch-to-patch jump in patch-mean (L2)."""
    P = win.shape[0] // patch
    pv = win[: P * patch].reshape(P, patch, win.shape[1]).mean(1)  # (P, C)
    jump = np.linalg.norm(np.diff(pv, axis=0), axis=1)             # (P-1,)
    return int(np.argmax(jump)) + 1


def baseline_lag(win: np.ndarray, patch: int, max_tau_p: int) -> int:
    """Signed cross-correlation lag estimate between channel 0 (driver) and 1."""
    P = win.shape[0] // patch
    pv = win[: P * patch].reshape(P, patch, win.shape[1]).mean(1)
    a = pv[:, 0] - pv[:, 0].mean()
    b = pv[:, 1] - pv[:, 1].mean()
    best_tau, best_corr = 0, -1e9
    for tau in range(-max_tau_p, max_tau_p + 1):
        if tau >= 0:
            x, y = a[: P - tau], b[tau:] if tau else b
        else:
            x, y = a[-tau:], b[: P + tau]
        if len(x) < 3:
            continue
        c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 and y.std() > 1e-6 else 0.0
        if c > best_corr:
            best_corr, best_tau = c, tau
    return best_tau  # positive => channel 0 leads channel 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lafr(model: LAFR, train_x: np.ndarray, args, device) -> List[Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    n = len(train_x)
    rng = np.random.default_rng(args.seed)
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        agg: Dict[str, float] = {}
        nb = 0
        for s in range(0, n, args.batch_size):
            idx = order[s:s + args.batch_size]
            if len(idx) < 4:
                continue
            xb = torch.tensor(train_x[idx], device=device)
            loss, logs = model.pretext_losses(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        history.append({k: v / max(nb, 1) for k, v in agg.items()})
        print(f"  epoch {epoch+1:2d}/{args.epochs}  " +
              "  ".join(f"{k}={history[-1].get(k, float('nan')):.3f}"
                        for k in ("loss", "cond", "sparse", "fc", "mr", "cc", "bd", "inst")))
    model.eval()
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_changepoint(model, val_x, mean, std, args, device) -> Dict[str, float]:
    rng = np.random.default_rng(args.seed + 1)
    hits_l = hits_b = 0
    mae_l = mae_b = 0.0
    n = min(args.eval_samples, len(val_x))
    for i in range(n):
        raw = val_x[i]
        w, gp = make_changepoint(raw, rng, args.patch)
        xb = torch.tensor(standardize(w[None], mean, std), device=device)
        out = model(xb)
        pred_l = int(out.boundary_logits[0].argmax().item())
        pred_b = baseline_changepoint(w, args.patch)
        hits_l += int(abs(pred_l - gp) <= args.cp_tol)
        hits_b += int(abs(pred_b - gp) <= args.cp_tol)
        mae_l += abs(pred_l - gp); mae_b += abs(pred_b - gp)
    return {
        "lafr_hit@tol": hits_l / n, "baseline_hit@tol": hits_b / n,
        "lafr_mae_patch": mae_l / n, "baseline_mae_patch": mae_b / n,
        "tol_patches": float(args.cp_tol), "n": float(n),
    }


def make_confounder(win: np.ndarray, rng: np.random.Generator, patch: int,
                    child_noise: float = 0.7) -> Tuple[np.ndarray, int]:
    """Plant a confounder on real CARE signal. Slot 0 = a real driver channel; slots
    1,2 = driver + independent noise (confounded children: marginally correlated but
    CONDITIONALLY INDEPENDENT given 0); slot 3 = independent noise; slot 4 = driver
    delayed by tau patches (directed lag edge 0->4). Returns (window, tau_patches).
    GT graph on block {0,1,2,3}: edges {0-1, 0-2}; non-edges everything else."""
    T, C = win.shape
    w = win.copy()
    # liveliest channel -> slot 0 as the driver, unit-normalized, so the confounder is
    # well-posed regardless of which (possibly near-constant) sensor lands in slot 0.
    drv_idx = int(np.argmax(w.std(axis=0)))
    if drv_idx != 0:
        w[:, [0, drv_idx]] = w[:, [drv_idx, 0]]
    drv = w[:, 0].copy()
    drv = (drv - drv.mean()) / (drv.std() + 1e-6)
    w[:, 0] = drv
    w[:, 1] = drv + child_noise * rng.standard_normal(T).astype(np.float32)
    w[:, 2] = drv + child_noise * rng.standard_normal(T).astype(np.float32)
    w[:, 3] = rng.standard_normal(T).astype(np.float32)
    tau_p = int(rng.choice([2, 3]))
    lag = tau_p * patch
    w[lag:, 4] = drv[:T - lag] + 0.3 * rng.standard_normal(T - lag).astype(np.float32)
    return w, tau_p


def _abs_corr_block(w: np.ndarray, block: Sequence[int]) -> np.ndarray:
    X = w[:, block]
    X = X - X.mean(0, keepdims=True)
    cov = (X.T @ X) / len(X)
    d = np.sqrt(np.diag(cov) + 1e-8)
    return np.abs(cov / np.outer(d, d))


def _abs_pcorr_block(w: np.ndarray, block: Sequence[int], ridge: float = 0.1) -> np.ndarray:
    X = w[:, block]
    X = X - X.mean(0, keepdims=True)
    cov = (X.T @ X) / len(X) + ridge * np.eye(len(block))
    prec = np.linalg.inv(cov)
    d = np.sqrt(np.diag(prec) + 1e-8)
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return np.abs(pc)


def _edge_auc(M: np.ndarray, pos: Sequence[Tuple[int, int]]) -> float:
    """Ranking AUC over the off-diagonal pairs of a small block: do true edges score
    above non-edges? Threshold-free, comparable across methods."""
    n = M.shape[0]
    pos_set = {tuple(sorted(p)) for p in pos}
    pos_s, neg_s = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (pos_s if (i, j) in pos_set else neg_s).append(M[i, j])
    if not pos_s or not neg_s:
        return float("nan")
    wins = sum(ps > ns for ps in pos_s for ns in neg_s)
    return wins / (len(pos_s) * len(neg_s))


@torch.no_grad()
def eval_dependency(model, val_x, mean, std, args, device) -> Dict[str, float]:
    """CORE eval: recover a CONFOUNDED conditional dependency graph on CARE-based data.
    LAFR's learned graph vs marginal correlation vs partial correlation (precision)."""
    rng = np.random.default_rng(args.seed + 2)
    block = [0, 1, 2, 3]
    pos_edges = [(0, 1), (0, 2)]                       # true direct edges
    n = min(args.eval_samples, len(val_x))
    auc = {"lafr": 0.0, "marginal": 0.0, "partial": 0.0}
    # accumulate spurious and direct edge magnitudes SEPARATELY, then divide the means.
    # A per-window ratio M[1,2]/direct blows up whenever a single window's direct edge is
    # ~0 (multicollinear driver/children under full 16-var conditioning make LAFR's
    # precision ill-conditioned for the odd window); aggregate-then-divide is robust.
    spur = {"lafr": 0.0, "marginal": 0.0, "partial": 0.0}
    drct = {"lafr": 0.0, "marginal": 0.0, "partial": 0.0}
    dir_hit = 0
    for i in range(n):
        w, tau = make_confounder(val_x[i], rng, args.patch)
        xb = torch.tensor(standardize(w[None], mean, std), device=device)
        out = model(xb)
        G = out.dependency_graph[0].cpu().numpy()
        Gb = G[np.ix_(block, block)]
        Mb = _abs_corr_block(w, block)
        Pb = _abs_pcorr_block(w, block)
        for name, M in (("lafr", Gb), ("marginal", Mb), ("partial", Pb)):
            auc[name] += _edge_auc(M, pos_edges)
            spur[name] += M[1, 2]
            drct[name] += 0.5 * (M[0, 1] + M[0, 2])
        dir_hit += int(float(out.lag_pred[0, 0, 4].item()) > 0)   # 0 leads 4
    return {
        **{f"auc_{k}": v / n for k, v in auc.items()},
        **{f"spurious_ratio_{k}": spur[k] / (drct[k] + 1e-8) for k in spur},
        "lag_direction_hit": dir_hit / n, "n": float(n),
    }


@torch.no_grad()
def eval_selfsup(model, val_x, args, device) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    nb = 0
    for s in range(0, len(val_x), args.batch_size):
        xb = torch.tensor(val_x[s:s + args.batch_size], device=device)
        if len(xb) < 4:
            continue
        _, logs = model.pretext_losses(xb)
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {f"val_{k}": v / max(nb, 1) for k, v in agg.items()}


@torch.no_grad()
def eval_retrieval(model, val_x, args, device) -> Dict[str, float]:
    n = min(args.retrieval_pool, len(val_x))
    xb = torch.tensor(val_x[:n], device=device)
    # two augmented views: query vs gallery; recall@1 of matching the same window
    q = torch.nn.functional.normalize(model(model._augment(xb)).episode_embedding, dim=-1)
    g = torch.nn.functional.normalize(model(model._augment(xb)).episode_embedding, dim=-1)
    sim = q @ g.t()                                  # (n, n)
    pred = sim.argmax(dim=1)
    recall1 = float((pred == torch.arange(n, device=device)).float().mean())
    # random baseline = 1/n
    return {"view_retrieval_recall@1": recall1, "random_baseline": 1.0 / n, "pool": float(n)}


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path("data/care_to_compare_extracted/CARE_To_Compare"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/lafr_eval"))
    ap.add_argument("--max-files", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--stride", type=int, default=72)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--eval-samples", type=int, default=300)
    ap.add_argument("--retrieval-pool", type=int, default=200)
    ap.add_argument("--cp-tol", type=int, default=1)
    ap.add_argument("--lag-choices", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--split-mode", choices=["random", "cross_farm"], default="random",
                    help="random = file-disjoint random split; cross_farm = train on all "
                         "farms except --holdout-farm, evaluate on the unseen farm.")
    ap.add_argument("--holdout-farm", type=str, default="Wind Farm B",
                    help="farm name held out for evaluation when --split-mode cross_farm.")
    ap.add_argument("--normalize", choices=["global", "per_window"], default="global",
                    help="global = one train-fit per-channel mean/std (breaks across farms with "
                         "different anonymized scales); per_window = instance-normalize each window "
                         "(scale-invariant, required for cross_farm transfer).")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_csvs(args.data_root)[: args.max_files]
    if not files:
        raise SystemExit(f"no CARE csvs under {args.data_root}")
    channels = choose_channels(files, args.channels)
    print(f"[data] {len(files)} files, C={len(channels)} channels: "
          f"{[c.replace('_avg','') for c in channels[:6]]}...")

    # train/val split: file-disjoint random, or hold out an entire unseen wind farm
    def farm_of(p: Path) -> str:
        return p.parent.parent.name                      # .../<Wind Farm X>/datasets/<n>.csv
    if args.split_mode == "cross_farm":
        val_files = [f for f in files if farm_of(f) == args.holdout_farm]
        train_files = [f for f in files if farm_of(f) != args.holdout_farm]
        if not val_files or not train_files:
            raise SystemExit(f"cross_farm split empty: holdout='{args.holdout_farm}', "
                             f"farms present={sorted({farm_of(f) for f in files})}")
        print(f"[split] cross_farm: train farms={sorted({farm_of(f) for f in train_files})} "
              f"-> held-out (unseen) test farm='{args.holdout_farm}'")
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(files))
        n_val = max(1, int(len(files) * args.val_frac))
        val_files = [files[i] for i in perm[:n_val]]
        train_files = [files[i] for i in perm[n_val:]]
        print(f"[split] random file-disjoint: {len(train_files)} train / {len(val_files)} val files")

    train_raw = load_windows(train_files, channels, args.window, args.max_rows, args.stride)
    val_raw = load_windows(val_files, channels, args.window, args.max_rows, args.stride)
    if args.normalize == "per_window":
        # instance-normalize each window per channel -> scale-invariant, so a model
        # trained on one farm transfers to another with different anonymized scales.
        # The global mean/std become identity, so the eval's standardize(.) calls (which
        # re-standardize injected windows for the model) are a no-op on already-normed data.
        def per_window(W: np.ndarray) -> np.ndarray:
            m = W.mean(axis=1, keepdims=True)
            s = W.std(axis=1, keepdims=True) + 1e-6
            return ((W - m) / s).astype(np.float32)
        train_raw = per_window(train_raw)
        val_raw = per_window(val_raw)
        mean = np.zeros((1, len(channels)), np.float32)
        std = np.ones((1, len(channels)), np.float32)
        train_x, val_x = train_raw, val_raw
    else:
        mean, std = fit_standardizer(train_raw)
        train_x = standardize(train_raw, mean, std)
        val_x = standardize(val_raw, mean, std)
    print(f"[data] norm={args.normalize}  train windows={len(train_x)} (files {len(train_files)}), "
          f"val windows={len(val_x)} (files {len(val_files)})")

    lag_bins = [0, 1, 2, 3, 4, 6, 8]
    model = LAFR(max_channels=args.channels, d_model=64, relation_dim=32,
                 max_events=6, patch=args.patch, n_heads=4, n_layers=2,
                 lag_bins=lag_bins).to(device)
    print(f"[model] LAFR params={sum(p.numel() for p in model.parameters()):,}")

    print("[train] self-supervised on real CARE windows ...")
    history = train_lafr(model, train_x, args, device)

    print("[eval] mechanism evaluation on CARE-injected ground truth ...")
    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "channels": channels,
        "n_train": len(train_x), "n_val": len(val_x),
        "train_history": history,
        "changepoint": eval_changepoint(model, val_raw, mean, std, args, device),
        "dependency": eval_dependency(model, val_raw, mean, std, args, device),
        "selfsup_val": eval_selfsup(model, val_x, args, device),
        "retrieval": eval_retrieval(model, val_x, args, device),
    }
    (args.output_dir / "lafr_eval_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "lafr_encoder.pt")

    # ---- pretty print ----
    cp, dep = report["changepoint"], report["dependency"]
    print("\n================ LAFR mechanism evaluation (CARE-based GT) ================")
    print(f"[CORE: conditional dependency graph]  (confounder: driver -> 2 children)")
    print(f"  edge-ranking AUC (higher=better):   LAFR={dep['auc_lafr']:.3f}   "
          f"marginal-corr={dep['auc_marginal']:.3f}   partial-corr={dep['auc_partial']:.3f}")
    print(f"  spurious/direct ratio (lower=better):LAFR={dep['spurious_ratio_lafr']:.3f}   "
          f"marginal-corr={dep['spurious_ratio_marginal']:.3f}   partial-corr={dep['spurious_ratio_partial']:.3f}")
    print(f"  (marginal correlation links the confounded children; conditional methods reject it)")
    print(f"[edge direction (lag attribute)]  0 leads 4 hit-rate={dep['lag_direction_hit']:.3f}")
    print(f"[change-point localization]  (tol=±{args.cp_tol} patch)")
    print(f"  hit@tol: LAFR={cp['lafr_hit@tol']:.3f} vs baseline={cp['baseline_hit@tol']:.3f}   "
          f"MAE: LAFR={cp['lafr_mae_patch']:.3f} vs {cp['baseline_mae_patch']:.3f}")
    print(f"[self-sup on held-out]  forecast={report['selfsup_val'].get('val_fc',float('nan')):.3f}  "
          f"masked={report['selfsup_val'].get('val_mr',float('nan')):.3f}")
    print(f"[instance retrieval]  recall@1={report['retrieval']['view_retrieval_recall@1']:.3f}  "
          f"(random={report['retrieval']['random_baseline']:.3f})")
    print(f"\nwrote {args.output_dir / 'lafr_eval_report.json'}")


if __name__ == "__main__":
    main()
