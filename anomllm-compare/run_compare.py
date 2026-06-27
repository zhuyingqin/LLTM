"""AnomLLM benchmark, our way: compare LAFR (ours) vs Isolation-Forest / percentile-threshold
(AnomLLM's non-LLM baselines) on AnomLLM's OWN synthetic anomaly datasets, scored with their
EXACT affinity-F1 metric. The LLM numbers (GPT-4o image/text etc.) are reported in the paper;
here we establish whether a small TRAINED temporal encoder beats their non-LLM baselines on
their own turf (a fair, fully-local, API-free comparison).
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "experiments"))      # the main project's LAFR
from lafr_encoder import LAFR                              # noqa: E402
import eval_metrics as M                                   # noqa: E402

LAG_BINS = [0, 1, 2, 3, 4, 6, 8]


def load_split(data_root, atype, split):
    with open(Path(data_root) / atype / split / "data.pkl", "rb") as f:
        d = pickle.load(f)
    series = [np.asarray(s, np.float32).reshape(-1, 1) for s in d["series"]]   # (L,1)
    gts = [M.interval_to_vector(a[0], len(s)) for s, a in zip(series, d["anom"])]
    return series, gts


# ---------- baselines (ported from AnomLLM src/baselines/isoforest.py) ----------
def iso_forest_pred(series):
    from sklearn.ensemble import IsolationForest
    pred = IsolationForest(random_state=42).fit(series).predict(series)
    return np.where(pred == -1, 1, 0).astype(int).flatten()


def threshold_pred(series):
    lo, hi = np.percentile(series, 2), np.percentile(series, 98)
    return np.logical_or(series <= lo, series >= hi).astype(int).flatten()


# ---------- ours: LAFR forecast-residual anomaly score ----------
def lafr_score(model, series, window, patch, device):
    L = len(series); Pp = window // patch
    score = np.zeros(L, dtype=np.float32)
    nw = L // window
    if nw == 0:
        return score
    x = series[: nw * window].reshape(nw, window, 1).astype(np.float32)
    m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True) + 1e-6
    xt = torch.tensor((x - m) / s, device=device)
    with torch.no_grad():
        value = model._patchify(xt)                          # (nw,P,1)
        bb = model._backbone(value, xt)
        fc = model.forecast_head(bb["z"]).squeeze(-1)        # (nw,P,1)
        resid = ((fc[:, :-1] - value[:, 1:]) ** 2).mean(-1).cpu().numpy()   # (nw,P-1)
    for w in range(nw):
        for p in range(Pp - 1):
            t0 = w * window + (p + 1) * patch
            score[t0:t0 + patch] = resid[w, p]
    return score


def train_lafr(series_list, window, patch, device, epochs, lr):
    wins = []
    for s in series_list:
        nw = len(s) // window
        if nw:
            x = s[: nw * window].reshape(nw, window, 1)
            x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
            wins.append(x.astype(np.float32))
    X = np.concatenate(wins, 0)
    model = LAFR(max_channels=1, d_model=64, relation_dim=32, max_events=6, patch=patch,
                 n_heads=4, n_layers=2, lag_bins=LAG_BINS).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = 64
    for ep in range(epochs):
        perm = np.random.permutation(len(X)); tot = nb = 0
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[perm[i:i + bs]], device=device)
            if len(xb) < 4:
                continue
            loss, _ = model.pretext_losses(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"    lafr ep{ep+1}/{epochs} loss={tot/max(nb,1):.3f}")
    return model.eval()


def best_threshold(scores, gts):
    """pick the score percentile (on TRAIN) that maximizes mean affinity F1."""
    alls = np.concatenate(scores)
    best_thr, best_f1 = float(alls.max()), -1.0
    for q in np.linspace(80, 99.5, 28):
        thr = np.percentile(alls, q)
        preds = [(sc > thr).astype(int) for sc in scores]
        f1 = M.mean_metrics(gts, preds)["affi f1"]
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--types", default="point,range,trend,freq")
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=str(ROOT / "compare_results.json"))
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    np.random.seed(13); torch.manual_seed(13)

    rows = {}
    for atype in args.types.split(","):
        print(f"\n===== {atype} =====")
        tr_series, tr_gts = load_split(args.data_root, atype, "train")
        ev_series, ev_gts = load_split(args.data_root, atype, "eval")

        # baselines
        iso = [iso_forest_pred(s) for s in ev_series]
        thr = [threshold_pred(s) for s in ev_series]
        m_iso = M.mean_metrics(ev_gts, iso)
        m_thr = M.mean_metrics(ev_gts, thr)

        # ours: train LAFR on train, pick threshold on train, eval
        model = train_lafr(tr_series, args.window, args.patch, device, args.epochs, args.lr)
        tr_scores = [lafr_score(model, s, args.window, args.patch, device) for s in tr_series]
        thr_val, tr_f1 = best_threshold(tr_scores, tr_gts)
        ev_scores = [lafr_score(model, s, args.window, args.patch, device) for s in ev_series]
        lafr_pred = [(sc > thr_val).astype(int) for sc in ev_scores]
        m_lafr = M.mean_metrics(ev_gts, lafr_pred)

        rows[atype] = {"iso_forest": m_iso, "threshold": m_thr, "lafr_ours": m_lafr,
                       "lafr_train_affi_f1": tr_f1}
        print(f"  affi-F1:  iso_forest={m_iso['affi f1']:.3f}  threshold={m_thr['affi f1']:.3f}  "
              f"LAFR(ours)={m_lafr['affi f1']:.3f}")

    # ---- summary table (affinity F1, the paper's main metric) ----
    print("\n================ AnomLLM benchmark — AFFINITY F1 (higher=better) ================")
    print(f"{'type':8s}{'iso_forest':>12s}{'threshold':>12s}{'LAFR(ours)':>12s}{'ours-iso':>12s}")
    types = list(rows)
    for t in types:
        a, b, c = rows[t]['iso_forest']['affi f1'], rows[t]['threshold']['affi f1'], rows[t]['lafr_ours']['affi f1']
        print(f"{t:8s}{a:>12.3f}{b:>12.3f}{c:>12.3f}{c-a:>+12.3f}")
    ma = np.mean([rows[t]['iso_forest']['affi f1'] for t in types])
    mb = np.mean([rows[t]['threshold']['affi f1'] for t in types])
    mc = np.mean([rows[t]['lafr_ours']['affi f1'] for t in types])
    print(f"{'MEAN':8s}{ma:>12.3f}{mb:>12.3f}{mc:>12.3f}{mc-ma:>+12.3f}")
    print("\n(point F1 also recorded in the json. Paper's LLM affinity-F1 numbers are in "
          "arXiv:2410.05440 Tables — add them for the full picture; their headline is that LLMs "
          "rarely beat Isolation-Forest on these synthetic types.)")

    import json
    Path(args.output).write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
