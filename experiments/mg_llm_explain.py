"""MG explanation track — the LLM OUTPUTS natural-language explanations.

Upgrade over mg_llm_adapter.py (which probes single-letter logits): the training
target is now a full explanation sentence generated from the injected ground
truth, and evaluation parses the LLM's FREE-FORM generation for factual claims
(who leads, by how many patches, direct vs confounded, regime change + where)
and scores them against the known structure. The grounding trap is kept: the
same generation is produced under correct / shuffled / no_token conditions, and
the explanations only count as grounded if correct-condition facts are right
far more often than the controls.

Pipeline (unchanged direction): MG series -> LAFR (frozen, self-sup pretrained)
-> adapter tokens -> frozen Qwen3.5-0.8B -> generated explanation.

Run:  python experiments/mg_llm_explain.py            (full)
      python experiments/mg_llm_explain.py --smoke    (tiny end-to-end check)
      python experiments/mg_llm_explain.py --demo     (load ckpts, print explanations)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_adapter as A                     # noqa: E402
import lafr_eval as E                       # noqa: E402
from lafr_encoder import LAFR               # noqa: E402
from mg_llm_adapter import make_mg_bank, sample_windows  # noqa: E402

LEAD_Q = ("Two of the sensors move together with a time delay. Which sensor leads, "
          "A or B, and by roughly how many patches? Explain briefly.")
CONF_Q = ("Sensor B and sensor C are both correlated. Is their link a DIRECT "
          "dependency, or CONFOUNDED through a shared driver? Explain briefly.")
CP_Q = ("Did the process switch to a new operating state during this window? "
        "If yes, around which patch? Explain briefly.")


@dataclass
class ExplainExample:
    window: np.ndarray
    question: str
    answer: str                     # target explanation (train_adapter reads this)
    task: str
    facts: Dict = field(default_factory=dict)


def lead_target(x: str, y: str, tau_p: int) -> str:
    return (f"Sensor {x} leads. Sensor {y} repeats sensor {x}'s pattern about "
            f"{tau_p} patches later, so {x} is the driver and {y} is the delayed follower.")


def conf_target(direct: bool) -> str:
    if direct:
        return ("The link between B and C is direct: C follows B itself, "
                "not a shared driver.")
    return ("The link between B and C is confounded: both follow the same driver A, "
            "so there is no direct dependency between them.")


def cp_target(gp: Optional[int]) -> str:
    if gp is None:
        return "No. The window is stable; no regime change occurs."
    return (f"Yes. A regime change occurs around patch {gp}: part of the channels "
            f"jump to a new operating level and stay there.")


def make_explain_qa(base_windows: np.ndarray, rng: np.random.Generator, patch: int,
                    lag_choices, eos: str) -> List[ExplainExample]:
    qa: List[ExplainExample] = []
    for w0 in base_windows:
        # lead: ch0 leads ch1 by tau patches; randomize which slot is called "A"
        w, tau_p = E.make_lag(w0, rng, patch, lag_choices)
        if rng.random() < 0.5:
            qa.append(ExplainExample(w.copy(), LEAD_Q, lead_target("A", "B", tau_p) + eos,
                                     "lead", {"leader": "A", "tau": tau_p}))
        else:
            w2 = w.copy(); w2[:, [0, 1]] = w2[:, [1, 0]]
            qa.append(ExplainExample(w2, LEAD_Q, lead_target("B", "A", tau_p) + eos,
                                     "lead", {"leader": "B", "tau": tau_p}))
        # confounder vs direct on the B-C pair
        wc, _ = E.make_confounder(w0, rng, patch)
        if rng.random() < 0.5:
            qa.append(ExplainExample(wc.copy(), CONF_Q, conf_target(False) + eos,
                                     "conf", {"direct": False}))
        else:
            wd = wc.copy(); wd[:, [1, 2]] = wd[:, [0, 1]]
            qa.append(ExplainExample(wd, CONF_Q, conf_target(True) + eos,
                                     "conf", {"direct": True}))
        # change-point: global property with a known location when present
        if rng.random() < 0.5:
            wcp, gp = E.make_changepoint(w0, rng, patch)
            qa.append(ExplainExample(wcp, CP_Q, cp_target(gp) + eos, "cp", {"cp": gp}))
        else:
            qa.append(ExplainExample(w0.copy(), CP_Q, cp_target(None) + eos, "cp", {"cp": None}))
    return qa


# ---------------------------------------------------------------------------
# Fact scoring: parse the generated text and check it against injected GT
# ---------------------------------------------------------------------------
def score_generation(task: str, facts: Dict, text: str) -> Dict:
    t = text.lower()
    if task == "lead":
        m = re.search(r"sensor\s+([ab])\s+leads", t)
        ok = (m.group(1).upper() == facts["leader"]) if m else None
        mt = re.search(r"(\d+)\s+patch", t)
        tau_err = abs(int(mt.group(1)) - facts["tau"]) if (mt and ok) else None
        return {"parsed": m is not None, "correct": ok, "tau_err": tau_err}
    if task == "conf":
        has_dir, has_conf = "direct" in t, "confound" in t
        # target phrasing mentions exactly one verdict word first; use whichever comes first
        if has_dir and has_conf:
            verdict = "direct" if t.find("direct") < t.find("confound") else "confounded"
        elif has_dir or has_conf:
            verdict = "direct" if has_dir else "confounded"
        else:
            return {"parsed": False, "correct": None}
        return {"parsed": True, "correct": verdict == ("direct" if facts["direct"] else "confounded")}
    # cp
    yes = bool(re.search(r"\byes\b|regime change occurs", t))
    no = bool(re.search(r"\bno\b|stable", t))
    if not (yes or no):
        return {"parsed": False, "correct": None}
    said_yes = yes and not (no and t.find("no") < t.find("yes") if yes else False)
    ok = said_yes == (facts["cp"] is not None)
    mp = re.search(r"patch\s+(\d+)", t)
    loc_err = abs(int(mp.group(1)) - facts["cp"]) if (mp and ok and facts["cp"] is not None) else None
    return {"parsed": True, "correct": ok, "loc_err": loc_err}


@torch.no_grad()
def generate_one(llm, emb: torch.Tensor, max_new: int) -> str:
    att = torch.ones(1, emb.shape[0], dtype=torch.long, device=emb.device)
    out = llm.model.generate(inputs_embeds=emb.unsqueeze(0), attention_mask=att,
                             max_new_tokens=max_new, do_sample=False,
                             eos_token_id=llm.tok.eos_token_id,
                             pad_token_id=llm.tok.eos_token_id or 0)
    return llm.tok.decode(out[0], skip_special_tokens=True).strip().replace("\n", " ")


@torch.no_grad()
def gen_eval(llm, adapter, lafr, qa: List[ExplainExample], device, mode: str,
             max_new: int, n_samples: int) -> Dict:
    wins = np.stack([q.window for q in qa])
    out = A.encode_lafr(lafr, wins, device)
    v, ch, pair = adapter(out)
    order = np.random.default_rng(0).permutation(len(qa))
    per_task: Dict[str, List[Dict]] = {}
    samples: List[Dict] = []
    for i, q in enumerate(qa):
        j = int(order[i]) if mode == "shuffled" else i
        emb, _ = A.build_seq(llm, adapter, v[j], ch[j], pair[j], q.question, None, mode)
        text = generate_one(llm, emb, max_new)
        s = score_generation(q.task, q.facts, text)
        per_task.setdefault(q.task, []).append(s)
        if sum(x["task"] == q.task for x in samples) < n_samples:
            samples.append({"task": q.task, "facts": {k: v2 for k, v2 in q.facts.items()},
                            "generated": text, "score": s})
    summary = {}
    for t, rows in per_task.items():
        parsed = [r for r in rows if r["parsed"]]
        correct = [r for r in parsed if r["correct"]]
        errs = [r[k] for r in parsed for k in ("tau_err", "loc_err")
                if r.get(k) is not None]
        summary[t] = {"n": len(rows), "parse_rate": len(parsed) / len(rows),
                      "fact_acc": (len(correct) / len(parsed)) if parsed else 0.0,
                      "mean_num_err": (float(np.mean(errs)) if errs else None)}
    return {"summary": summary, "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--output-dir", type=Path, default=Path("results/mg_llm_explain"))
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--m-slots", type=int, default=4)
    ap.add_argument("--bank-length", type=int, default=6000)
    ap.add_argument("--lafr-windows", type=int, default=600)
    ap.add_argument("--lafr-epochs", type=int, default=6)
    ap.add_argument("--lafr-batch", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-eval", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--n-samples", type=int, default=4, help="saved sample generations per task/condition")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--demo", action="store_true", help="load saved ckpts and print explanations")
    ap.add_argument("--demo-n", type=int, default=2, help="demo windows per task")
    args = ap.parse_args()
    if args.smoke:
        args.lafr_windows, args.lafr_epochs = 96, 1
        args.n_train, args.n_eval, args.epochs = 18, 12, 1
        args.output_dir = args.output_dir / "smoke"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    P = args.window // args.patch
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    llm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=llm_dtype)
    llm = A.FrozenLLM(model, tok, device)
    eos = tok.eos_token or ""
    print(f"[llm] {args.llm} frozen: d_llm={llm.d_llm}, eos={eos!r}")

    bank = make_mg_bank(args.channels, args.bank_length, args.seed)
    lafr = LAFR(max_channels=args.channels, d_model=64, relation_dim=32, max_events=6,
                patch=args.patch, n_heads=4, n_layers=2, lag_bins=A.LAG_BINS).to(device)
    adapter = A.TemporalAdapter(lafr.event_dim, 64, llm.d_llm, P, m_slots=args.m_slots).to(device)

    if args.demo:
        lafr.load_state_dict(torch.load(args.output_dir / "lafr_mg_explain.pt", map_location=device))
        adapter.load_state_dict(torch.load(args.output_dir / "adapter.pt", map_location=device))
        lafr.eval(); adapter.eval()
        mean, std = E.fit_standardizer(sample_windows(bank, 64, args.window, rng))
        base = E.standardize(sample_windows(bank, args.demo_n * 3 + 4, args.window, rng), mean, std)
        qa = make_explain_qa(base, rng, args.patch, [1, 2, 3, 4], eos="")
        for t in ("lead", "conf", "cp"):
            for q in [x for x in qa if x.task == t][: args.demo_n]:
                out = A.encode_lafr(lafr, q.window[None], device)
                v, ch, pr = adapter(out)
                emb, _ = A.build_seq(llm, adapter, v[0], ch[0], pr[0], q.question, None, "correct")
                print("=" * 88)
                print(f"TASK={q.task}  GT={q.facts}")
                print(f"Q: {q.question}")
                print(f"LLM: {generate_one(llm, emb, args.max_new)}")
        return

    lafr_train = sample_windows(bank, args.lafr_windows, args.window, rng)
    mean, std = E.fit_standardizer(lafr_train)
    lafr_train = E.standardize(lafr_train, mean, std)
    print(f"[data] MG bank {bank.shape}, LAFR pretrain windows {lafr_train.shape}")
    lafr_args = argparse.Namespace(lr=1e-3, epochs=args.lafr_epochs,
                                   batch_size=args.lafr_batch, seed=args.seed)
    print("[lafr] self-supervised pretraining on MG ...")
    E.train_lafr(lafr, lafr_train, lafr_args, device)
    for p in lafr.parameters():
        p.requires_grad_(False)
    lafr.eval()
    torch.save(lafr.state_dict(), args.output_dir / "lafr_mg_explain.pt")

    n_base = (args.n_train + args.n_eval) // 3 + 4
    base = E.standardize(sample_windows(bank, n_base, args.window, rng), mean, std)
    qa_all = make_explain_qa(base, rng, args.patch, [1, 2, 3, 4], eos)
    rng.shuffle(qa_all)
    qa_train = qa_all[: args.n_train]
    qa_eval = qa_all[args.n_train: args.n_train + args.n_eval]
    print(f"[data] explanation QA train={len(qa_train)} eval={len(qa_eval)}")
    print(f"[target example] {qa_train[0].answer[:110]}")

    print("[train] adapter only, CE over the FULL explanation (LAFR + LLM frozen) ...")
    hist = A.train_adapter(llm, adapter, lafr, qa_train, device, args.epochs, args.lr, args.batch_size)

    results = {}
    for mode in ("correct", "shuffled", "no_token"):
        r = gen_eval(llm, adapter, lafr, qa_eval, device, mode, args.max_new, args.n_samples)
        results[mode] = r
        line = "  ".join(f"{t}: acc={s['fact_acc']:.3f} parse={s['parse_rate']:.2f}"
                         for t, s in sorted(r["summary"].items()))
        print(f"[gen-eval {mode:9s}] {line}")
    for t in sorted(results["correct"]["summary"]):
        gap = results["correct"]["summary"][t]["fact_acc"] - results["shuffled"]["summary"][t]["fact_acc"]
        print(f"   task={t:5s} grounding gap (correct-shuffled fact acc) = {gap:+.3f}")

    print("\n--- sample explanations (correct condition) ---")
    for s in results["correct"]["samples"]:
        print(f"[{s['task']}] GT={s['facts']}  ->  {s['generated'][:150]}")

    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "device": str(device), "runtime_sec": round(time.time() - t0, 1),
              "train_ce": hist,
              "gen_eval": {m: results[m]["summary"] for m in results},
              "samples": {m: results[m]["samples"] for m in results},
              "n_train": len(qa_train), "n_eval": len(qa_eval)}
    (args.output_dir / "mg_llm_explain_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    torch.save(adapter.state_dict(), args.output_dir / "adapter.pt")
    print(f"\nwrote {args.output_dir / 'mg_llm_explain_report.json'}")


if __name__ == "__main__":
    main()
