"""FULL, untruncated side-by-side of the two interfaces into the SAME frozen Qwen3-0.6B.

For a few eval examples it prints, with NOTHING truncated:
  1. the COMPLETE text string the TEXT-serialized baseline tokenizes and feeds;
  2. the OURS prompt skeleton + a readable table of WHAT each LAFR token actually carries
     (event times; per-pair lag tau, conditional-dependency G, strength; channel summary),
     since ours feeds continuous inputs_embeds, not literal text;
  3. the full greedy generation of the frozen LLM under each interface.
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
def generate(llm, emb, max_new=40) -> str:
    att = torch.ones(1, emb.shape[0], dtype=torch.long, device=emb.device)
    out = llm.model.generate(inputs_embeds=emb.unsqueeze(0), attention_mask=att,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=llm.tok.eos_token_id or 0)
    return llm.tok.decode(out[0], skip_special_tokens=True).strip()


def ours_text_view(adapter, out) -> str:
    """Human-readable rendering of the continuous tokens the adapter injects."""
    P = 12
    t = (out.soft_boundary_positions[0] / (P - 1)).clamp(0, 1).tolist()
    lines = ["Sensor event memory:"]
    lines.append("  event tokens (K=6) at normalized times: " +
                 ", ".join(f"t{k}={v:.2f}" for k, v in enumerate(t)))
    s = out.step_hidden.mean(dim=1)[0]                         # (C, d) per-channel summary
    lines.append("  Sensors (each = a learned summary vector of that channel):")
    for k in range(4):
        lines.append(f"    {A.LETTERS[k]} = <channel-{k} summary, ||.||={s[k].norm():.2f}>")
    lines.append("  Relations (each token carries [lag tau, conditional-dependency G, strength]):")
    for m, (i, j) in enumerate(zip(adapter.pair_i.tolist(), adapter.pair_j.tolist())):
        tau = out.lag_pred[0, i, j].item()
        g = out.dependency_graph[0, i, j].item()
        st = out.relation_strength[0, i, j].item()
        lines.append(f"    {A.LETTERS[i]}-{A.LETTERS[j]} = <tau={tau:+.2f}, G={g:.3f}, strength={st:.2f}>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lafr-ckpt", type=Path, default=Path("results/lafr_eval_full_fixed/lafr_encoder.pt"))
    ap.add_argument("--adapter-ckpt", type=Path, default=Path("results/llm_adapter_full_qwen3/adapter.pt"))
    ap.add_argument("--data-root", type=Path,
                    default=Path("data/care_to_compare_extracted/CARE_To_Compare/Wind Farm C"))
    ap.add_argument("--tasks", default="lead,cp,conf")
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
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
    adapter = A.TemporalAdapter(lafr.event_dim, 64, llm.d_llm, 12, m_slots=4).to(device)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device)); adapter.eval()

    _, qa_eval = C.replicate_qa(cfg)
    picks = []
    for t in args.tasks.split(","):
        picks += [q for q in qa_eval if q.task == t][: args.per_task]

    for q in picks:
        wtext = C.serialize(q.window, 4, 6)
        text_prompt = f"Sensor readings over time (patch means):\n{wtext}\nQuestion: {q.question}\nAnswer:"
        emb_t, _ = C.build_text_seq(llm, None, wtext, q.question, None, "correct")
        gen_t = generate(llm, emb_t, args.max_new)
        out = A.encode_lafr(lafr, q.window[None], device)
        v, ch, pr = adapter(out)
        emb_o, _ = A.build_seq(llm, adapter, v[0], ch[0], pr[0], q.question, None, "correct")
        gen_o = generate(llm, emb_o, args.max_new)

        print("#" * 100)
        print(f"# TASK = {q.task}      GROUND-TRUTH ANSWER = '{q.answer}'   (choices {q.choices})")
        print("#" * 100)
        print("\n----- (A) TEXT-SERIALIZED BASELINE: the COMPLETE string fed to the LLM -----\n")
        print(text_prompt)
        print(f"\n  >>> LLM full output: \"{gen_t}\"\n")
        print("----- (B) OURS: the prompt skeleton + WHAT each injected token encodes -----\n")
        print(ours_text_view(adapter, out))
        print(f"\nQuestion: {q.question}\nAnswer:")
        print(f"\n  >>> LLM full output: \"{gen_o}\"\n")


if __name__ == "__main__":
    main()
