"""Qualitative side-by-side: what does the frozen LLM actually SAY when fed
(a) the time series as TEXT numbers   vs   (b) our LAFR tokens (trained adapter)?

Same frozen Qwen3-0.6B, same questions. Greedy free-form generation (not just the
argmax letter) so you can read the model's actual output under each interface.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_adapter as A
import llm_compare as C
from lafr_encoder import LAFR


@torch.no_grad()
def generate(llm, emb, max_new=12) -> str:
    att = torch.ones(1, emb.shape[0], dtype=torch.long, device=emb.device)
    out = llm.model.generate(inputs_embeds=emb.unsqueeze(0), attention_mask=att,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=llm.tok.eos_token_id or 0)
    return llm.tok.decode(out[0], skip_special_tokens=True).strip().replace("\n", " ")


@torch.no_grad()
def letter_pick(llm, emb, choices, label_ids) -> str:
    att = torch.ones(1, emb.shape[0], dtype=torch.long, device=emb.device)
    logits = llm.model(inputs_embeds=emb.unsqueeze(0), attention_mask=att).logits[0, -1]
    c0, c1 = choices
    return c0 if logits[label_ids[c0]] >= logits[label_ids[c1]] else c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lafr-ckpt", type=Path, default=Path("results/lafr_eval_full_fixed/lafr_encoder.pt"))
    ap.add_argument("--adapter-ckpt", type=Path, default=Path("results/llm_adapter_full_qwen3/adapter.pt"))
    ap.add_argument("--data-root", type=Path,
                    default=Path("data/care_to_compare_extracted/CARE_To_Compare/Wind Farm C"))
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    # match the trained/eval config so replicate_qa reproduces the SAME eval examples
    cfg = SimpleNamespace(data_root=args.data_root, max_files=20, max_rows=4000, channels=16,
                          window=72, stride=72, patch=6, m_slots=4, n_train=240, n_eval=150, seed=13)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    lafr = LAFR(max_channels=16, d_model=64, relation_dim=32, max_events=6, patch=6,
                n_heads=4, n_layers=2, lag_bins=A.LAG_BINS).to(device).eval()
    lafr.load_state_dict(torch.load(args.lafr_ckpt, map_location=device)); [p.requires_grad_(False) for p in lafr.parameters()]
    dt = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = A.FrozenLLM(AutoModelForCausalLM.from_pretrained(args.llm, dtype=dt), tok, device)
    label_ids = llm.label_token_ids(["A", "B", "C", "D", "Y", "N"])
    adapter = A.TemporalAdapter(lafr.event_dim, 64, llm.d_llm, 12, m_slots=4).to(device)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device)); adapter.eval()

    _, qa_eval = C.replicate_qa(cfg)
    picks = []
    for t in ("lead", "conf", "cp"):
        picks += [q for q in qa_eval if q.task == t][: args.per_task]

    for q in picks:
        wtext = C.serialize(q.window, 4, 6)
        # text baseline
        emb_t, _ = C.build_text_seq(llm, None, wtext, q.question, None, "correct")
        gen_t = generate(llm, emb_t); pick_t = letter_pick(llm, emb_t, q.choices, label_ids)
        # ours
        out = A.encode_lafr(lafr, q.window[None], device)
        v, ch, pr = adapter(out)
        emb_o, _ = A.build_seq(llm, adapter, v[0], ch[0], pr[0], q.question, None, "correct")
        gen_o = generate(llm, emb_o); pick_o = letter_pick(llm, emb_o, q.choices, label_ids)

        print("=" * 92)
        print(f"TASK={q.task}   GROUND-TRUTH ANSWER = '{q.answer}'   choices={q.choices}")
        print(f"Q: {q.question}")
        print(f"  numbers fed to TEXT: {wtext[:120]}...")
        print(f"  [TEXT-TS ]  pick='{pick_t}' {'OK' if pick_t==q.answer else 'WRONG'}   gen: \"{gen_t[:70]}\"")
        print(f"  [OURS    ]  pick='{pick_o}' {'OK' if pick_o==q.answer else 'WRONG'}   gen: \"{gen_o[:70]}\"")
    print("=" * 92)


if __name__ == "__main__":
    main()
