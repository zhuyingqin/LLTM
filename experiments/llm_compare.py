"""Head-to-head: TEXT-serialized time series -> frozen LLM  vs  our LAFR-token adapter.

Same frozen Qwen, same QA, same eval split. Answers the question MODEL_ARCHITECTURE.md §2.2
poses: do LAFR's latent event tokens beat just dumping the raw series into the LLM as text?

Three methods compared on identical (question, window) eval examples:
  * text_zeroshot : window -> patch-mean numbers as text, frozen LLM, NO training
                    (the literal "feed the time series to the LLM" baseline).
  * text_soft     : same text numbers + a trainable SOFT-PROMPT prefix, trained on the QA
                    (matched-compute baseline: trainable params + frozen LLM, but TS as text).
  * adapter (ours): LAFR event/channel/pair tokens via the trained adapter.

Each is reported with the grounding controls (correct / shuffled / no_token).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lafr_eval as E                                    # noqa: E402
import llm_adapter as A                                  # noqa: E402  reuse the verified pieces
from lafr_encoder import LAFR                            # noqa: E402


def replicate_qa(args) -> Tuple[list, list]:
    """Regenerate the EXACT train/eval QA split used by llm_adapter.main (same seed/params)."""
    files = E.discover_csvs(args.data_root)[: args.max_files]
    channels = E.choose_channels(files, args.channels)
    raw = E.load_windows(files, channels, args.window, args.max_rows, args.stride)
    rng = np.random.default_rng(args.seed)
    raw = raw[rng.permutation(len(raw))]
    n_base = (args.n_train + args.n_eval) // 3 + 4
    qa_all = A.make_qa(raw[:n_base], rng, args.patch, [1, 2, 3, 4])
    rng.shuffle(qa_all)
    return qa_all[: args.n_train], qa_all[args.n_train: args.n_train + args.n_eval]


def serialize(w: np.ndarray, m_slots: int, patch: int) -> str:
    """Per-window-normalized patch means of the first m_slots channels, channel-labeled A.. so
    the LLM has the SAME channel-binding affordance our adapter gives. This is the standard
    'TS as text' interface (Time-LLM / Gruver et al.) at patch resolution."""
    x = w.astype(np.float32)
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
    P = x.shape[0] // patch
    pv = x[: P * patch].reshape(P, patch, x.shape[1]).mean(1)        # (P, C)
    lines = []
    for c in range(m_slots):
        nums = ",".join(f"{v:+.2f}" for v in pv[:, c])
        lines.append(f"{A.LETTERS[c]}=[{nums}]")
    return " ".join(lines)


def build_text_seq(llm, soft, wtext, question, answer, mode):
    if mode == "no_token":
        return A.assemble(llm, ["Sensor readings unavailable. ", f"Question: {question}\nAnswer:"], answer)
    segs = []
    if soft is not None:
        segs.append(soft)                                            # (P,d) trainable prefix
    segs.append(f"Sensor readings over time (patch means):\n{wtext}\nQuestion: {question}\nAnswer:")
    return A.assemble(llm, segs, answer)


@torch.no_grad()
def text_acc(llm, soft, qa, texts, label_ids, mode, batch=8) -> float:
    order = np.random.default_rng(0).permutation(len(qa))
    correct = 0
    for s in range(0, len(qa), batch):
        chunk = list(range(s, min(s + batch, len(qa))))
        seqs = []
        for i in chunk:
            j = int(order[i]) if mode == "shuffled" else i
            emb, _ = build_text_seq(llm, soft, texts[j], qa[i].question, None, mode)
            seqs.append((emb, torch.zeros(emb.shape[0], dtype=torch.long, device=llm.device)))
        emb, _, att = A.pad_batch(llm, seqs)
        logits = llm.model(inputs_embeds=emb, attention_mask=att).logits[:, -1]
        for k, i in enumerate(chunk):
            c0, c1 = qa[i].choices
            pick = c0 if logits[k, label_ids[c0]] >= logits[k, label_ids[c1]] else c1
            correct += int(pick == qa[i].answer)
    return correct / len(qa)


def by_task(fn) -> Dict[str, Dict[str, float]]:
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lafr-ckpt", type=Path, default=Path("results/lafr_eval_full_fixed/lafr_encoder.pt"))
    ap.add_argument("--adapter-ckpt", type=Path, default=Path("results/llm_adapter_full_qwen3/adapter.pt"))
    ap.add_argument("--data-root", type=Path,
                    default=Path("data/care_to_compare_extracted/CARE_To_Compare/Wind Farm C"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/llm_compare"))
    ap.add_argument("--max-files", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--stride", type=int, default=72)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--m-slots", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--soft-tokens", type=int, default=16)
    ap.add_argument("--soft-train-n", type=int, default=160,
                    help="subset of qa_train used to train the soft prompt (eval set unchanged); "
                         "text sequences are ~3x longer so this caps the slow backward pass.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    P = args.window // args.patch

    lafr = LAFR(max_channels=args.channels, d_model=64, relation_dim=32, max_events=6,
                patch=args.patch, n_heads=4, n_layers=2, lag_bins=A.LAG_BINS).to(device).eval()
    lafr.load_state_dict(torch.load(args.lafr_ckpt, map_location=device))
    for p in lafr.parameters():
        p.requires_grad_(False)

    llm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=llm_dtype)
    llm = A.FrozenLLM(model, tok, device)
    label_ids = llm.label_token_ids(["A", "B", "C", "D", "Y", "N"])

    qa_train, qa_eval = replicate_qa(args)
    tasks = sorted({q.task for q in qa_eval})
    print(f"[data] QA train={len(qa_train)} eval={len(qa_eval)} tasks={tasks}")

    # ---------- method 1+2: TEXT serialization ----------
    txt_tr = [serialize(q.window, args.m_slots, args.patch) for q in qa_train]
    txt_ev = [serialize(q.window, args.m_slots, args.patch) for q in qa_eval]

    def text_eval(soft, tag):
        res = {}
        for t in tasks:
            idx = [i for i, q in enumerate(qa_eval) if q.task == t]
            sub = [qa_eval[i] for i in idx]; subtx = [txt_ev[i] for i in idx]
            res[t] = {m: text_acc(llm, soft, sub, subtx, label_ids, m, args.batch_size)
                      for m in ("correct", "shuffled", "no_token")}
        print(f"  [{tag}]  " + "  ".join(f"{t}:{res[t]['correct']:.3f}" for t in tasks))
        return res

    print("[text_zeroshot] frozen LLM reads serialized numbers, no training ...")
    zs = text_eval(None, "text_zeroshot")

    print(f"[text_soft] training a {args.soft_tokens}-token soft prompt on the SAME QA ...")
    soft = nn.Parameter(0.02 * torch.randn(args.soft_tokens, llm.d_llm, device=device))
    opt = torch.optim.AdamW([soft], lr=args.lr)
    n_soft = min(args.soft_train_n, len(qa_train))
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(n_soft)
        tot = nb = 0
        for s in range(0, n_soft, args.batch_size):
            idx = perm[s:s + args.batch_size]
            seqs = [build_text_seq(llm, soft, txt_tr[i], qa_train[i].question, qa_train[i].answer, "correct")
                    for i in idx]
            loss = A.ce_loss(llm, seqs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"    epoch {ep+1}/{args.epochs}  soft-prompt CE={tot/max(nb,1):.4f}")
    ts = text_eval(soft, "text_soft")

    # ---------- method 3: OUR adapter (load trained) ----------
    adapter = A.TemporalAdapter(lafr.event_dim, 64, llm.d_llm, P, m_slots=args.m_slots).to(device)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device))
    adapter.eval()
    ours = {}
    for t in tasks:
        sub = [q for q in qa_eval if q.task == t]
        ours[t] = {m: A.answer_acc(llm, adapter, lafr, sub, device, m, label_ids, args.batch_size)
                   for m in ("correct", "shuffled", "no_token")}
    print(f"  [adapter(ours)]  " + "  ".join(f"{t}:{ours[t]['correct']:.3f}" for t in tasks))

    # ---------- comparison table ----------
    print("\n================ TEXT-TS  vs  LAFR-ADAPTER (correct-condition accuracy) ================")
    print(f"{'task':6s}{'text_zeroshot':>16s}{'text_soft':>14s}{'adapter(ours)':>16s}{'ours-best_text':>16s}")
    methods = {"text_zeroshot": zs, "text_soft": ts, "adapter": ours}
    summary = {}
    for t in tasks:
        z, s_, o = zs[t]['correct'], ts[t]['correct'], ours[t]['correct']
        best_text = max(z, s_)
        print(f"{t:6s}{z:>16.3f}{s_:>14.3f}{o:>16.3f}{o-best_text:>+16.3f}")
        summary[t] = {"text_zeroshot": z, "text_soft": s_, "adapter": o, "ours_minus_best_text": o - best_text}
    mean_o = np.mean([ours[t]['correct'] for t in tasks])
    mean_bt = np.mean([max(zs[t]['correct'], ts[t]['correct']) for t in tasks])
    print(f"{'MEAN':6s}{np.mean([zs[t]['correct'] for t in tasks]):>16.3f}"
          f"{np.mean([ts[t]['correct'] for t in tasks]):>14.3f}{mean_o:>16.3f}{mean_o-mean_bt:>+16.3f}")

    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "text_zeroshot": zs, "text_soft": ts, "adapter": ours, "summary": summary,
              "mean_adapter": float(mean_o), "mean_best_text": float(mean_bt)}
    (args.output_dir / "compare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output_dir / 'compare_report.json'}")
    print(f"VERDICT: ours mean {mean_o:.3f} vs best-text mean {mean_bt:.3f}  "
          f"-> {'LAFR tokens WIN' if mean_o-mean_bt>0.05 else 'NO clear win'} ({mean_o-mean_bt:+.3f})")


if __name__ == "__main__":
    main()
