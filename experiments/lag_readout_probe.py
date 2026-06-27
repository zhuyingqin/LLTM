"""Fast lag-readout ablation (no training).

The directional lag is (almost) a deterministic function of the cross-correlation
alignment, so we can compare readout strategies directly on CARE-injected lag pairs
without the 2-min full self-sup run. Pick the best readout, then bake it into LAFR.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lafr_eval as E
from lafr_encoder import LAFR

LAG_BINS = [0, 1, 2, 3, 4, 6, 8]
SIGNED = torch.tensor([-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8], dtype=torch.float32)


def signed_scores(d: torch.Tensor) -> torch.Tensor:
    # d: directional evidence at positive lags (L-1,). Build signed-axis score vector.
    zero = torch.zeros(1)
    return torch.cat([(-d).flip(0), zero, d])          # matches SIGNED order


def signed_xcorr(a01: torch.Tensor, a10: torch.Tensor) -> torch.Tensor:
    # Build the FULL signed cross-correlation over SIGNED axis directly (no difference):
    #   g>0 -> corr(i leads j by g) = align[i,j,g]
    #   g<0 -> corr(i lags j by |g|) = align[j,i,|g|]
    return torch.cat([a10[1:].flip(0), a01[:1], a01[1:]])      # matches SIGNED order


def readouts(d: torch.Tensor, a01: torch.Tensor, a10: torch.Tensor) -> dict:
    s = signed_scores(d)                                       # old: directional difference
    sx = signed_xcorr(a01, a10)                                # new: signed cross-corr
    out = {}
    for t in (0.5, 0.1):
        out[f"diff@{t}"] = float((torch.softmax(s / t, 0) * SIGNED).sum())
    for t in (0.5, 0.2, 0.1, 0.05):
        out[f"xcorr@{t}"] = float((torch.softmax(sx / t, 0) * SIGNED).sum())
    out["xcorr_argmax"] = float(SIGNED[int(sx.argmax())])
    return out


def main() -> None:
    root = Path("data/care_to_compare_extracted/CARE_To_Compare")
    files = E.discover_csvs(root)[:16]
    channels = E.choose_channels(files, 16)
    raw = E.load_windows(files, channels, 72, 4000, 72)
    mean, std = E.fit_standardizer(raw)
    rng = np.random.default_rng(7)
    model = LAFR(max_channels=16, patch=6, lag_bins=LAG_BINS).eval()

    n = 400
    names = ["diff@0.5", "diff@0.1", "xcorr@0.5", "xcorr@0.2", "xcorr@0.1", "xcorr@0.05",
             "xcorr_argmax", "baseline_xcorr"]
    dir_hit = {k: 0 for k in names}
    mae = {k: 0.0 for k in names}
    for _ in range(n):
        w, tau = E.make_lag(raw[rng.integers(len(raw))], rng, 6, [1, 2, 3, 4])
        xb = torch.tensor(E.standardize(w[None], mean, std))
        with torch.no_grad():
            val = model._patchify(xb)
            align = model.lag_bank._cross_corr(val)            # (1,C,C,L)
        a01 = align[0, 0, 1]                                   # corr(0 leads 1 by g)
        a10 = align[0, 1, 0]                                   # corr(1 leads 0 by g)
        d = (a01 - a10)[1:]                                    # directional, lags>0
        r = readouts(d, a01, a10)
        r["baseline_xcorr"] = float(E.baseline_lag(w, 6, 4))
        for k in names:
            dir_hit[k] += int(r[k] > 0)                         # GT: ch0 leads ch1 -> +
            mae[k] += abs(r[k] - tau)
    print(f"injected lag pairs n={n}  (GT: channel 0 leads channel 1)\n")
    print(f"{'readout':16s}{'direction_hit':>16s}{'lag_MAE(patch)':>18s}")
    print("-" * 50)
    for k in names:
        print(f"{k:16s}{dir_hit[k]/n:>16.3f}{mae[k]/n:>18.3f}")


if __name__ == "__main__":
    main()
