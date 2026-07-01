"""Run a standalone Mackey-Glass forecasting experiment with LAFR.

The forecasting head is part of LAFR itself. This script generates a synthetic
Mackey-Glass sequence, builds a small multichannel view of it, trains LAFR on
prefix-to-future patch prediction, and saves plots/metrics under results/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lafr_encoder import LAFR  # noqa: E402


def mackey_glass(
    n: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    power: int = 10,
    dt: float = 1.0,
    seed: int = 13,
    warmup: int = 300,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = n + warmup + tau + 1
    x = np.empty(total, dtype=np.float32)
    x[: tau + 1] = 1.2 + 0.01 * rng.standard_normal(tau + 1).astype(np.float32)
    for t in range(tau, total - 1):
        xtau = x[t - tau]
        dx = beta * xtau / (1.0 + xtau**power) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx
    return x[warmup + tau + 1 :].astype(np.float32)


def causal_moving_average(x: np.ndarray, width: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    csum = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    for i in range(len(x)):
        start = max(0, i - width + 1)
        out[i] = (csum[i + 1] - csum[start]) / float(i - start + 1)
    return out


def make_channels(x: np.ndarray) -> np.ndarray:
    lag1 = np.roll(x, 1)
    lag6 = np.roll(x, 6)
    ma7 = causal_moving_average(x, 7)
    dx = np.concatenate([[0.0], np.diff(x)]).astype(np.float32)
    lag1[0] = x[0]
    lag6[:6] = x[0]
    arr = np.stack([x, lag1, lag6, ma7, dx], axis=1).astype(np.float32)
    return arr


def standardize(train: np.ndarray, all_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return ((all_data - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def make_windows(data: np.ndarray, total_steps: int, stride: int) -> np.ndarray:
    rows = []
    for start in range(0, len(data) - total_steps + 1, stride):
        rows.append(data[start : start + total_steps])
    return np.stack(rows).astype(np.float32)


def metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = math.sqrt(mse)
    denom = float(np.mean((target - target.mean()) ** 2) + 1e-8)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": float(1.0 - mse / denom)}


@torch.no_grad()
def predict_batches(model: LAFR, windows: np.ndarray, context_patches: int, horizon_patches: int,
                    device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, len(windows), batch_size):
        xb = torch.tensor(windows[start : start + batch_size, : context_patches * model.patch], device=device)
        preds.append(model.forecast(xb, horizon_patches).cpu().numpy())
    return np.concatenate(preds, axis=0)


def plot_prediction(
    out_dir: Path,
    pred_main: np.ndarray,
    true_main: np.ndarray,
    naive_main: np.ndarray,
    patch: int,
    lead_patches: int,
) -> None:
    n_show = min(420, len(true_main))
    t = np.arange(n_show) * patch
    err = pred_main[:n_show] - true_main[:n_show]
    naive_err = naive_main[:n_show] - true_main[:n_show]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, true_main[:n_show], label="MG true", color="#222222", linewidth=1.8)
    axes[0].plot(t, pred_main[:n_show], label="LAFR forecast", color="#0B6E69", linewidth=1.5)
    axes[0].plot(t, naive_main[:n_show], label="Naive last-patch", color="#B25E00", linewidth=1.0, alpha=0.75)
    axes[0].set_ylabel("standardized patch mean")
    axes[0].set_title(f"Mackey-Glass Forecast With LAFR (fixed lead={lead_patches} patches)")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, err, label="LAFR error", color="#0B6E69", linewidth=1.2)
    axes[1].plot(t, naive_err, label="Naive error", color="#B25E00", linewidth=1.0, alpha=0.65)
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("error")
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "mg_lafr_forecast.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=40, alpha=0.75, color="#0B6E69", label="LAFR")
    ax.hist(naive_err, bins=40, alpha=0.45, color="#B25E00", label="Naive")
    ax.set_title("Forecast Error Distribution")
    ax.set_xlabel("prediction error")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "mg_lafr_error_hist.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("results/mg_lafr_forecast"))
    ap.add_argument("--length", type=int, default=7000)
    ap.add_argument("--context-patches", type=int, default=24)
    ap.add_argument("--horizon-patches", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--stride-patches", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = mackey_glass(args.length, seed=args.seed)
    channels = make_channels(raw)
    split = int(len(channels) * 0.7)
    data, mean, std = standardize(channels[:split], channels)

    total_patches = args.context_patches + args.horizon_patches
    total_steps = total_patches * args.patch
    stride = args.stride_patches * args.patch
    train_windows = make_windows(data[:split], total_steps, stride)
    test_windows = make_windows(data[split - args.context_patches * args.patch :], total_steps, stride)

    model = LAFR(
        max_channels=data.shape[1],
        d_model=48,
        relation_dim=24,
        event_dim=48,
        episode_dim=48,
        max_events=4,
        patch=args.patch,
        n_heads=4,
        n_layers=2,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        order = np.random.permutation(len(train_windows))
        total_loss = total_mae = nb = 0
        model.train()
        for start in range(0, len(train_windows), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = torch.tensor(train_windows[idx], device=device)
            loss, logs = model.forecasting_loss(xb, args.context_patches, args.horizon_patches)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += logs["forecast_mse"]
            total_mae += logs["forecast_mae"]
            nb += 1
        history.append({"epoch": epoch + 1, "train_mse": total_loss / nb, "train_mae": total_mae / nb})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"train_mse={history[-1]['train_mse']:.5f} "
                f"train_mae={history[-1]['train_mae']:.5f}",
                flush=True,
            )

    pred = predict_batches(model, test_windows, args.context_patches, args.horizon_patches, device, args.batch_size)
    target = np.stack([
        w[args.context_patches * args.patch : total_steps].reshape(args.horizon_patches, args.patch, data.shape[1]).mean(axis=1)
        for w in test_windows
    ])
    context_value = np.stack([
        w[: args.context_patches * args.patch].reshape(args.context_patches, args.patch, data.shape[1]).mean(axis=1)
        for w in test_windows
    ])
    naive = np.repeat(context_value[:, -1:, :], args.horizon_patches, axis=1)

    pred_main = pred[:, :, 0].reshape(-1)
    true_main = target[:, :, 0].reshape(-1)
    naive_main = naive[:, :, 0].reshape(-1)
    lead_idx = args.horizon_patches - 1
    pred_lead_main = pred[:, lead_idx, 0]
    true_lead_main = target[:, lead_idx, 0]
    naive_lead_main = naive[:, lead_idx, 0]
    lead_metrics = {}
    for h in range(args.horizon_patches):
        lead_metrics[f"lead_{h + 1}"] = {
            "lafr": metrics(pred[:, h, 0], target[:, h, 0]),
            "naive": metrics(naive[:, h, 0], target[:, h, 0]),
        }
    report = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "train_windows": int(len(train_windows)),
        "test_windows": int(len(test_windows)),
        "history": history,
        "lafr_main": metrics(pred_main, true_main),
        "naive_main": metrics(naive_main, true_main),
        "fixed_lead_main": {
            "lead_patches": args.horizon_patches,
            "lafr": metrics(pred_lead_main, true_lead_main),
            "naive": metrics(naive_lead_main, true_lead_main),
        },
        "lead_metrics_main": lead_metrics,
        "lafr_all_channels": metrics(pred.reshape(-1), target.reshape(-1)),
        "naive_all_channels": metrics(naive.reshape(-1), target.reshape(-1)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "lafr_mg_forecast.pt")
    plot_prediction(
        args.output_dir,
        pred_lead_main,
        true_lead_main,
        naive_lead_main,
        args.patch,
        args.horizon_patches,
    )

    print("\nMackey-Glass forecasting metrics, main channel:", flush=True)
    print(f"  LAFR  mse={report['lafr_main']['mse']:.6f} mae={report['lafr_main']['mae']:.6f} r2={report['lafr_main']['r2']:.4f}", flush=True)
    print(f"  naive mse={report['naive_main']['mse']:.6f} mae={report['naive_main']['mae']:.6f} r2={report['naive_main']['r2']:.4f}", flush=True)
    print(
        f"  fixed lead-{args.horizon_patches}: "
        f"LAFR mse={report['fixed_lead_main']['lafr']['mse']:.6f} "
        f"mae={report['fixed_lead_main']['lafr']['mae']:.6f} "
        f"r2={report['fixed_lead_main']['lafr']['r2']:.4f}",
        flush=True,
    )
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
