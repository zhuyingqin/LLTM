"""FAIR 'let it think' comparison: give each interface a long chain-of-thought budget,
then force a final-letter answer (no fragile text parsing). Same frozen Qwen3-0.6B, same
eval subset, SAME protocol for both:

   [ content ] Question: ... Let's think step by step:  -> generate up to --think tokens
   ... <model reasoning> ... \nTherefore the final answer (one letter) is:  -> read choice logit

This addresses the concern that the earlier eval (immediate single letter, gen capped at 40)
unfairly cut off a reasoning model mid-thought. If the TEXT baseline stays ~chance even WITH
full reasoning, the LAFR-token win is real; if it jumps up, we must revise.
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

THINK_TRIGGER = "\nLet's think step by step:"
FORCE = "\nTherefore, the final answer (exactly one letter) is:"


@torch.no_grad()
def generate_ids(llm, ctx_emb, max_new):
    att = torch.ones(1, ctx_emb.shape[0], dtype=torch.long, device=ctx_emb.device)
    out = llm.model.generate(inputs_embeds=ctx_emb.unsqueeze(0), attention_mask=att,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=llm.tok.eos_token_id or 0)
    return out[0]                                            # new tokens only (inputs_embeds)


@torch.no_grad()
def cot_answer(llm, ctx_emb, choices, label_ids, think) -> tuple:
    """Generate reasoning from ctx, then force a final letter and read its logit."""
    think_ids = generate_ids(llm, ctx_emb, think)
    think_emb = llm.embed(think_ids.unsqueeze(0))[0]
    force_emb = llm.text_embeds(FORCE)
    full = torch.cat([ctx_emb, think_emb, force_emb], 0)
    att = torch.ones(1, full.shape[0], dtype=torch.long, device=full.device)
    logit = llm.model(inputs_embeds=full.unsqueeze(0), attention_mask=att).logits[0, -1]
    c0, c1 = choices
    pick = c0 if logit[label_ids[c0]] >= logit[label_ids[c1]] else c1
    reasoning = llm.tok.decode(think_ids, skip_special_tokens=True).strip().replace("\n", " ")
    return pick, reasoning


def text_ctx(llm, wtext, question, tail=THINK_TRIGGER):
    return A.assemble(llm, [f"Sensor readings over time (patch means):\n{wtext}\n"
                            f"Question: {question}{tail}"], None)[0]


def ours_ctx(llm, adapter, v, ch, pr, question, tail=THINK_TRIGGER):
    segs = ["Sensor event memory: ", v, " Sensors:"]
    for k, L in enumerate(adapter.slot_letters):
        segs += [f" {L}=", ch[k:k + 1]]
    segs.append(" Relations:")
    for m, lab in enumerate(adapter.pair_labels):
        segs += [f" {lab}=", pr[m:m + 1]]
    segs.append(f"\nQuestion: {question}{tail}")
    return A.assemble(llm, segs, None)[0]


@torch.no_grad()
def immediate_pick(llm, ctx_emb, choices, label_ids) -> str:
    """No generation: read the choice-letter logit right after 'Answer:' (the original metric)."""
    att = torch.ones(1, ctx_emb.shape[0], dtype=torch.long, device=ctx_emb.device)
    logit = llm.model(inputs_embeds=ctx_emb.unsqueeze(0), attention_mask=att).logits[0, -1]
    c0, c1 = choices
    return c0 if logit[label_ids[c0]] >= logit[label_ids[c1]] else c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lafr-ckpt", type=Path, default=Path("results/lafr_eval_full_fixed/lafr_encoder.pt"))
    ap.add_argument("--adapter-ckpt", type=Path, default=Path("results/llm_adapter_full_qwen3/adapter.pt"))
    ap.add_argument("--data-root", type=Path,
                    default=Path("data/care_to_compare_extracted/CARE_To_Compare/Wind Farm C"))
    ap.add_argument("--per-task", type=int, default=24)
    ap.add_argument("--think", type=int, default=160)
    ap.add_argument("--show", type=int, default=1, help="print this many full CoT examples per task")
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
    label_ids = llm.label_token_ids(["A", "B", "C", "D", "Y", "N"])
    adapter = A.TemporalAdapter(lafr.event_dim, 64, llm.d_llm, 12, m_slots=4).to(device)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device)); adapter.eval()

    _, qa_eval = C.replicate_qa(cfg)
    tasks = ["lead", "conf", "cp"]
    sub = {t: [q for q in qa_eval if q.task == t][: args.per_task] for t in tasks}
    print(f"[cot-compare] think={args.think} tokens, n={args.per_task}/task, protocol=CoT->force-letter\n")

    # text gets the expensive CoT (the fairness question); ours uses its NATIVE immediate
    # protocol (it was trained to answer immediately, and CoT generation is slow + degenerate
    # at 0.6B). We also report text-immediate so the CoT effect on text is isolated.
    res = {t: {"text_imm": 0, "text_cot": 0, "ours_imm": 0} for t in tasks}
    for t in tasks:
        shown = 0
        for q in sub[t]:
            wtext = C.serialize(q.window, 4, 6)
            t_imm = immediate_pick(llm, text_ctx(llm, wtext, q.question, "\nAnswer:"), q.choices, label_ids)
            t_cot, tre = cot_answer(llm, text_ctx(llm, wtext, q.question), q.choices, label_ids, args.think)
            out = A.encode_lafr(lafr, q.window[None], device)
            v, ch, pr = adapter(out)
            o_imm = immediate_pick(llm, ours_ctx(llm, adapter, v[0], ch[0], pr[0], q.question, "\nAnswer:"),
                                   q.choices, label_ids)
            res[t]["text_imm"] += int(t_imm == q.answer)
            res[t]["text_cot"] += int(t_cot == q.answer)
            res[t]["ours_imm"] += int(o_imm == q.answer)
            if shown < args.show:
                shown += 1
                print(f"--- {t} example  (GT={q.answer}, choices {q.choices}) ---")
                print(f"  [TEXT immediate]={t_imm}   [TEXT-CoT]={t_cot} {'OK' if t_cot==q.answer else 'X'}"
                      f"   [OURS immediate]={o_imm} {'OK' if o_imm==q.answer else 'X'}")
                print(f"     text reasoning: {tre[:280]}\n")

    n = args.per_task
    print("\n========= FAIR TEST: does CHAIN-OF-THOUGHT rescue the TEXT baseline? =========")
    print(f"{'task':6s}{'text_imm':>10s}{'text_CoT':>10s}{'ours_imm':>10s}{'ours-textCoT':>14s}")
    for t in tasks:
        ti, tc, oi = res[t]["text_imm"]/n, res[t]["text_cot"]/n, res[t]["ours_imm"]/n
        print(f"{t:6s}{ti:>10.3f}{tc:>10.3f}{oi:>10.3f}{oi-tc:>+14.3f}")
    mti = np.mean([res[t]["text_imm"]/n for t in tasks])
    mtc = np.mean([res[t]["text_cot"]/n for t in tasks])
    moi = np.mean([res[t]["ours_imm"]/n for t in tasks])
    print(f"{'MEAN':6s}{mti:>10.3f}{mtc:>10.3f}{moi:>10.3f}{moi-mtc:>+14.3f}")
    print(f"\nVERDICT: text_immediate={mti:.3f} -> text_CoT={mtc:.3f} (CoT effect {mtc-mti:+.3f}); "
          f"ours={moi:.3f}.  {'CoT does NOT rescue text; LAFR tokens still win' if moi-mtc>0.05 else 'text closes the gap with CoT'} "
          f"(ours-textCoT {moi-mtc:+.3f})")


if __name__ == "__main__":
    main()
