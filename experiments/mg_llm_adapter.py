"""MG (Mackey-Glass) LLM+time-module experiment — the SIMPLE validation track.

Core scheme (unified entry + internal routing, decided 2026-07-01):
  MG time series -> LAFR (frozen, self-sup pretrained) -> adapter-aligned tokens
  -> frozen Qwen3.5-0.8B -> answer.

Everything is reused from llm_adapter.py (TemporalAdapter, FrozenLLM, make_qa with
the correct/shuffled/no_token grounding trap) and lafr_eval.py (GT injectors,
train_lafr). Only the data source changes: instead of CARE SCADA windows, base
windows are built from independent Mackey-Glass realizations, so the injected
lag / confounder / change-point ground truth is exact by construction.

Run:  python experiments/mg_llm_adapter.py                 (full, ~laptop GPU)
      python experiments/mg_llm_adapter.py --smoke          (tiny end-to-end check)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_adapter as A                     # noqa: E402  adapter + grounding protocol
import lafr_eval as E                       # noqa: E402  train_lafr (self-sup)
from lafr_encoder import LAFR               # noqa: E402
from mg_lafr_forecast import mackey_glass   # noqa: E402


def make_mg_bank(n_channels: int, length: int, seed: int) -> np.ndarray:
    """One long MG series per channel (different seeds + delays -> independent)."""
    taus = [17, 22, 30]
    bank = [mackey_glass(length, tau=taus[c % len(taus)], seed=seed + 101 * c)
            for c in range(n_channels)]
    return np.stack(bank, axis=1).astype(np.float32)          # (L, C)


def sample_windows(bank: np.ndarray, n: int, window: int, rng: np.random.Generator) -> np.ndarray:
    """Base windows: each channel sliced at an independent random offset."""
    L, C = bank.shape
    out = np.empty((n, window, C), dtype=np.float32)
    for i in range(n):
        for c in range(C):
            s = int(rng.integers(0, L - window))
            out[i, :, c] = bank[s:s + window, c]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--output-dir", type=Path, default=Path("results/mg_llm_adapter"))
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--m-slots", type=int, default=4)
    ap.add_argument("--bank-length", type=int, default=6000)
    ap.add_argument("--lafr-windows", type=int, default=600)
    ap.add_argument("--lafr-epochs", type=int, default=6)
    ap.add_argument("--lafr-batch", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, same full pipeline")
    args = ap.parse_args()
    if args.smoke:
        args.lafr_windows, args.lafr_epochs = 96, 1
        args.n_train, args.n_eval, args.epochs = 24, 18, 1
        args.output_dir = args.output_dir / "smoke"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    P = args.window // args.patch
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- MG data ----------------------------------------------------------
    bank = make_mg_bank(args.channels, args.bank_length, args.seed)
    lafr_train = sample_windows(bank, args.lafr_windows, args.window, rng)
    mean, std = E.fit_standardizer(lafr_train)
    lafr_train = E.standardize(lafr_train, mean, std)
    print(f"[data] MG bank {bank.shape}, LAFR pretrain windows {lafr_train.shape}")

    # ---- LAFR: self-supervised pretrain on plain MG windows, then freeze ---
    lafr = LAFR(max_channels=args.channels, d_model=64, relation_dim=32, max_events=6,
                patch=args.patch, n_heads=4, n_layers=2, lag_bins=A.LAG_BINS).to(device)
    lafr_args = argparse.Namespace(lr=1e-3, epochs=args.lafr_epochs,
                                   batch_size=args.lafr_batch, seed=args.seed)
    print("[lafr] self-supervised pretraining on MG ...")
    E.train_lafr(lafr, lafr_train, lafr_args, device)
    for p in lafr.parameters():
        p.requires_grad_(False)
    lafr.eval()
    torch.save(lafr.state_dict(), args.output_dir / "lafr_mg_qa.pt")

    # ---- frozen LLM --------------------------------------------------------
    llm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=llm_dtype)
    llm = A.FrozenLLM(model, tok, device)
    label_ids = llm.label_token_ids(["A", "B", "C", "D", "Y", "N"])
    print(f"[llm] {args.llm} frozen: d_llm={llm.d_llm}, label_ids={label_ids}")

    # ---- QA with exact injected ground truth -------------------------------
    n_base = (args.n_train + args.n_eval) // 3 + 4
    base = E.standardize(sample_windows(bank, n_base, args.window, rng), mean, std)
    qa_all = A.make_qa(base, rng, args.patch, [1, 2, 3, 4])
    rng.shuffle(qa_all)
    qa_train = qa_all[: args.n_train]
    qa_eval = qa_all[args.n_train: args.n_train + args.n_eval]
    print(f"[data] QA train={len(qa_train)} eval={len(qa_eval)}")

    adapter = A.TemporalAdapter(lafr.event_dim, 64, llm.d_llm, P, m_slots=args.m_slots).to(device)
    print(f"[adapter] params={sum(p.numel() for p in adapter.parameters()):,}  m_slots={args.m_slots}")

    def grounding(tag: str):
        res = {m: A.answer_acc(llm, adapter, lafr, qa_eval, device, m, label_ids, args.batch_size)
               for m in ("correct", "shuffled", "no_token")}
        print(f"[{tag}] correct={res['correct']:.3f}  shuffled={res['shuffled']:.3f}  "
              f"no_token={res['no_token']:.3f}  (gap={res['correct'] - res['shuffled']:+.3f})")
        return res

    pre = grounding("pre-train")
    print("[train] adapter only (LAFR + LLM frozen) ...")
    hist = A.train_adapter(llm, adapter, lafr, qa_train, device, args.epochs, args.lr, args.batch_size)
    post = grounding("post-train")

    by_task = {}
    for t in sorted({q.task for q in qa_eval}):
        sub = [q for q in qa_eval if q.task == t]
        by_task[t] = {m: A.answer_acc(llm, adapter, lafr, sub, device, m, label_ids, args.batch_size)
                      for m in ("correct", "shuffled", "no_token")}
        g = by_task[t]
        print(f"   task={t:5s} n={len(sub):3d}  correct={g['correct']:.3f}  "
              f"shuffled={g['shuffled']:.3f}  no_token={g['no_token']:.3f}  "
              f"gap={g['correct'] - g['shuffled']:+.3f}")

    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "device": str(device), "runtime_sec": round(time.time() - t0, 1),
              "train_ce": hist, "pre_train": pre, "post_train": post, "by_task": by_task,
              "n_train": len(qa_train), "n_eval": len(qa_eval)}
    (args.output_dir / "mg_llm_adapter_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    torch.save(adapter.state_dict(), args.output_dir / "adapter.pt")
    print(f"\nwrote {args.output_dir / 'mg_llm_adapter_report.json'}")
    grounded = sum(1 for t, g in by_task.items() if g["correct"] - g["shuffled"] > 0.1)
    print(f"GROUNDING: {grounded}/{len(by_task)} tasks grounded (correct-shuffled > 0.1)")


if __name__ == "__main__":
    main()
