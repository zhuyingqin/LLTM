"""Phase-2 Adapter — Frozen-LLM Event-Memory Bridge (MODEL_ARCHITECTURE.md §2), FULL version.

Goal (Q2): can LAFR's latent temporal event tokens be transferred to a FROZEN LLM so it
answers structured temporal questions it could not answer from text alone?

This is the CHANNEL-ADDRESSED full version. The minimal pilot grounded GLOBAL properties
(change-point) but failed channel-pair questions because the LLM had no way to bind
"sensor B" in the question to any token. Fix = the adapter now emits THREE token groups:

  * event tokens   v_k  : one per LAFR event (time/global info)        — §2.3.3 Strategy A
  * channel tokens c_m  : one per channel slot, carrying that channel's summary + a learned
                          slot-ID; the prompt writes "A=<c_0> B=<c_1> ..." so the question's
                          sensor letters are BOUND to specific tokens.
  * pair tokens   r_mn  : one per slot pair, carrying [sigma(tau_ij), G_ij (CONDITIONAL
                          dependency, the CORE), ||A_ij||] + a learned pair-ID; the prompt
                          writes "A-B=<r_01> B-C=<r_12> ..." so RELATIONS are addressable.

Only the adapter trains. LAFR and the LLM are frozen at all times (seam-gate G4, unit-tested).

GROUNDING TRAP (memory: describe-probe-prior-trap): every eval reports three conditions and
the claim holds only if correct >> the controls:
  (a) correct  : the window's real LAFR tokens
  (b) shuffled : another window's tokens (same distribution, wrong content)
  (c) no_token : text-only (no tokens) — the pure LLM/text prior
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lafr_eval as E                       # noqa: E402  CARE loaders + GT injectors
from lafr_encoder import LAFR, LAFROutput   # noqa: E402

LAG_BINS = [0, 1, 2, 3, 4, 6, 8]
LETTERS = "ABCDEFGH"


# ---------------------------------------------------------------------------
# Adapter (§2.3.3 Strategy A + channel addressing)
# ---------------------------------------------------------------------------
class TemporalAdapter(nn.Module):
    def __init__(self, event_dim: int, d_model: int, d_llm: int, n_patches: int,
                 m_slots: int = 4):
        super().__init__()
        self.d_llm = d_llm
        self.n_patches = n_patches
        self.m_slots = m_slots
        # event tokens
        self.W_e = nn.Linear(event_dim, d_llm)
        self.evt_ln = nn.LayerNorm(d_llm)
        self.r_evt = nn.Parameter(torch.zeros(d_llm))
        self.time_mlp = nn.Sequential(nn.Linear(1, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        # channel tokens (binding): per-channel summary + learned slot id
        self.W_ch = nn.Linear(d_model, d_llm)
        self.ch_ln = nn.LayerNorm(d_llm)
        self.slot_id = nn.Embedding(m_slots, d_llm)
        # pair tokens (relations): [sigma(tau), G_conditional, strength] + learned pair id
        pairs = [(i, j) for i in range(m_slots) for j in range(i + 1, m_slots)]
        self.pair_labels = [f"{LETTERS[i]}-{LETTERS[j]}" for i, j in pairs]
        self.register_buffer("pair_i", torch.tensor([i for i, _ in pairs]))
        self.register_buffer("pair_j", torch.tensor([j for _, j in pairs]))
        self.pair_mlp = nn.Sequential(nn.Linear(3, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        self.pair_id = nn.Embedding(len(pairs), d_llm)
        self.slot_letters = list(LETTERS[:m_slots])

    def forward(self, out: LAFROutput) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = self.W_e.weight.dtype
        # --- event tokens (global / time) ---
        e = out.event_embeddings.to(dt)                                  # (B,K,d_e)
        v = self.evt_ln(self.W_e(e)) + self.r_evt
        t = (out.soft_boundary_positions / max(self.n_patches - 1, 1)).clamp(0, 1).to(dt)
        v = v + self.time_mlp(t.unsqueeze(-1))                           # (B,K,d_llm)
        # --- channel tokens (binding) ---
        s = out.step_hidden.mean(dim=1).to(dt)[:, :self.m_slots]         # (B,M,d_model)
        ch = self.ch_ln(self.W_ch(s)) + self.slot_id.weight[None]       # (B,M,d_llm)
        # --- pair tokens (relations, incl. the CORE conditional dependency G) ---
        ii, jj = self.pair_i, self.pair_j
        G = out.dependency_graph[:, ii, jj].to(dt)                       # (B,Pn)
        tau = out.lag_pred[:, ii, jj].to(dt)
        st = out.relation_strength[:, ii, jj].to(dt)
        feat = torch.stack([torch.sigmoid(tau), G, st], dim=-1)          # (B,Pn,3)
        pair = self.pair_mlp(feat) + self.pair_id.weight[None]          # (B,Pn,d_llm)
        return v, ch, pair


# ---------------------------------------------------------------------------
# Frozen LLM wrapper
# ---------------------------------------------------------------------------
class FrozenLLM:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = tokenizer
        self.device = device
        self.embed = self.model.get_input_embeddings()
        self.d_llm = self.model.config.hidden_size

    def text_embeds(self, text: str) -> torch.Tensor:
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        return self.embed(ids)[0]

    def label_token_ids(self, letters: Sequence[str]) -> Dict[str, int]:
        out = {}
        for L in letters:
            out[L] = self.tok(L, add_special_tokens=False).input_ids[-1]
        return out


@dataclass
class QAExample:
    window: np.ndarray
    question: str
    answer: str
    choices: Tuple[str, str]
    task: str


# ---------------------------------------------------------------------------
# QA generation from injected ground truth (known answers)
# ---------------------------------------------------------------------------
LEAD_Q = ("Two sensors move together with a time delay. Does sensor A lead sensor B, "
          "or does sensor B lead sensor A? Answer 'A' if A leads, 'B' if B leads.")
CONF_Q = ("Sensor B and sensor C are both correlated with a driver. Is the link between "
          "B and C a DIRECT dependency, or only a CONFOUNDED one through the driver? "
          "Answer 'D' for direct, 'C' for confounded.")
CP_Q = ("Did the process switch to a new operating state at some point during this window? "
        "Answer 'Y' for yes (a regime change), 'N' for no (stable).")


def make_qa(base_windows: np.ndarray, rng: np.random.Generator, patch: int,
            lag_choices: Sequence[int]) -> List[QAExample]:
    qa: List[QAExample] = []
    for w0 in base_windows:
        # lead/lag: ch0 leads ch1; randomize which slot is "A"
        w, _ = E.make_lag(w0, rng, patch, lag_choices)
        if rng.random() < 0.5:
            qa.append(QAExample(w.copy(), LEAD_Q, "A", ("A", "B"), "lead"))
        else:
            w2 = w.copy(); w2[:, [0, 1]] = w2[:, [1, 0]]
            qa.append(QAExample(w2, LEAD_Q, "B", ("A", "B"), "lead"))
        # confounder: driver(0)->{1,2}; ask B-C (confounded) or make B-C direct
        wc, _ = E.make_confounder(w0, rng, patch)
        if rng.random() < 0.5:
            qa.append(QAExample(wc.copy(), CONF_Q, "C", ("D", "C"), "conf"))
        else:
            wd = wc.copy(); wd[:, [1, 2]] = wd[:, [0, 1]]      # B<-driver -> B-C now DIRECT
            qa.append(QAExample(wd, CONF_Q, "D", ("D", "C"), "conf"))
        # change-point: global property
        if rng.random() < 0.5:
            wcp, _ = E.make_changepoint(w0, rng, patch)
            qa.append(QAExample(wcp, CP_Q, "Y", ("Y", "N"), "cp"))
        else:
            qa.append(QAExample(w0.copy(), CP_Q, "N", ("Y", "N"), "cp"))
    return qa


# ---------------------------------------------------------------------------
# Sequence assembly (interleave text + token vectors)
# ---------------------------------------------------------------------------
def assemble(llm: FrozenLLM, segments: List, answer: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    edt = llm.embed.weight.dtype                           # LLM runs bf16; cast all tokens to it
    embs, labels = [], []
    for seg in segments:
        e = llm.text_embeds(seg) if isinstance(seg, str) else seg
        embs.append(e.to(edt))
        labels.append(torch.full((e.shape[0],), -100, dtype=torch.long, device=llm.device))
    if answer is not None:
        ans = llm.tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(llm.device)
        embs.append(llm.embed(ans)[0].to(edt)); labels.append(ans[0])
    return torch.cat(embs, 0), torch.cat(labels, 0)


def build_seq(llm, adapter, v, ch, pair, question, answer, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """mode in {correct, shuffled (caller passes another window's v/ch/pair), no_token}."""
    if mode == "no_token":
        return assemble(llm, ["Sensor event memory. ", f"Question: {question}\nAnswer:"], answer)
    segs: List = ["Sensor event memory: ", v]
    segs.append(" Sensors:")
    for k, L in enumerate(adapter.slot_letters):
        segs += [f" {L}=", ch[k:k + 1]]
    segs.append(" Relations:")
    for m, lab in enumerate(adapter.pair_labels):
        segs += [f" {lab}=", pair[m:m + 1]]
    segs.append(f"\nQuestion: {question}\nAnswer:")
    return assemble(llm, segs, answer)


def encode_lafr(lafr: LAFR, windows: np.ndarray, device, normalize_per_window=True) -> LAFROutput:
    x = windows.astype(np.float32)
    if normalize_per_window:
        m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True) + 1e-6
        x = (x - m) / s
    with torch.no_grad():
        return lafr(torch.tensor(x, device=device))


def pad_batch(llm, seqs: List[Tuple[torch.Tensor, torch.Tensor]]):
    Lmax = max(s[0].shape[0] for s in seqs); d = seqs[0][0].shape[1]
    emb = torch.zeros(len(seqs), Lmax, d, device=llm.device, dtype=seqs[0][0].dtype)
    lab = torch.full((len(seqs), Lmax), -100, dtype=torch.long, device=llm.device)
    att = torch.zeros(len(seqs), Lmax, dtype=torch.long, device=llm.device)
    for i, (e, l) in enumerate(seqs):
        emb[i, -e.shape[0]:] = e; lab[i, -l.shape[0]:] = l; att[i, -e.shape[0]:] = 1
    return emb, lab, att


def ce_loss(llm, seqs) -> torch.Tensor:
    emb, lab, att = pad_batch(llm, seqs)
    logits = llm.model(inputs_embeds=emb, attention_mask=att).logits[:, :-1]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_adapter(llm, adapter, lafr, qa_train, device, epochs, lr, batch) -> List[float]:
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr)
    wins = np.stack([q.window for q in qa_train])
    hist = []
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(qa_train))
        tot = nb = 0
        for s in range(0, len(qa_train), batch):
            idx = perm[s:s + batch]
            out = encode_lafr(lafr, wins[idx], device)
            v, ch, pair = adapter(out)
            seqs = [build_seq(llm, adapter, v[k], ch[k], pair[k],
                              qa_train[i].question, qa_train[i].answer, "correct")
                    for k, i in enumerate(idx)]
            loss = ce_loss(llm, seqs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        hist.append(tot / max(nb, 1))
        print(f"  epoch {ep+1}/{epochs}  adapter CE={hist[-1]:.4f}")
    return hist


@torch.no_grad()
def answer_acc(llm, adapter, lafr, qa, device, mode, label_ids, batch=8) -> float:
    wins = np.stack([q.window for q in qa])
    out = encode_lafr(lafr, wins, device)
    v, ch, pair = adapter(out)
    order = np.random.default_rng(0).permutation(len(qa))
    correct = 0
    for s in range(0, len(qa), batch):
        chunk = list(range(s, min(s + batch, len(qa))))
        seqs = []
        for i in chunk:
            j = int(order[i]) if mode == "shuffled" else i
            emb, _ = build_seq(llm, adapter, v[j], ch[j], pair[j], qa[i].question, None, mode)
            seqs.append((emb, torch.zeros(emb.shape[0], dtype=torch.long, device=device)))
        emb, _, att = pad_batch(llm, seqs)
        logits = llm.model(inputs_embeds=emb, attention_mask=att).logits[:, -1]
        for k, i in enumerate(chunk):
            c0, c1 = qa[i].choices
            pick = c0 if logits[k, label_ids[c0]] >= logits[k, label_ids[c1]] else c1
            correct += int(pick == qa[i].answer)
    return correct / len(qa)


# ---------------------------------------------------------------------------
# Architecture self-test (in-process tiny Llama; no download)
# ---------------------------------------------------------------------------
def _smoke() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=1024)
    model = LlamaForCausalLM(cfg)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.resize_token_embeddings(len(tok))
    llm = FrozenLLM(model, tok, device)
    print(f"[OK] frozen LLM built: d_llm={llm.d_llm}, vocab={len(tok)}")

    C, T, P = 16, 72, 12
    lafr = LAFR(max_channels=C, d_model=64, relation_dim=32, max_events=6, patch=6,
                n_heads=4, n_layers=2, lag_bins=LAG_BINS).eval()
    for p in lafr.parameters():
        p.requires_grad_(False)
    adapter = TemporalAdapter(lafr.event_dim, 64, llm.d_llm, P, m_slots=4)

    out = encode_lafr(lafr, np.random.randn(4, T, C).astype(np.float32), device)
    v, ch, pair = adapter(out)
    assert v.shape[2] == llm.d_llm and ch.shape == (4, 4, llm.d_llm) and pair.shape == (4, 6, llm.d_llm)
    emb, lab = build_seq(llm, adapter, v[0], ch[0], pair[0], "does A lead B?", "A", "correct")
    assert emb.shape[1] == llm.d_llm and (lab != -100).sum() >= 1
    print(f"[OK] addressed sequence: len={emb.shape[0]}, channel+pair tokens bound to letters "
          f"({adapter.slot_letters}, pairs={adapter.pair_labels})")
    # no_token path
    e2, _ = build_seq(llm, adapter, v[0], ch[0], pair[0], "q?", "A", "no_token")
    assert e2.shape[0] < emb.shape[0]
    print("[OK] no_token control builds a shorter text-only sequence")

    # frozen discipline
    seqs = [build_seq(llm, adapter, v[i], ch[i], pair[i], "q?", "A", "correct") for i in range(4)]
    ce_loss(llm, seqs).backward()
    llm_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in llm.model.parameters())
    lafr_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in lafr.parameters())
    ad_tot = sum(p.numel() for p in adapter.parameters())
    ad_grad = sum(p.numel() for p in adapter.parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
    assert llm_grad == 0 and lafr_grad == 0, "LLM/LAFR must stay frozen"
    assert ad_grad / ad_tot > 0.5
    print(f"[OK] frozen discipline: LLM grad=0, LAFR grad=0, adapter grad-cov={ad_grad/ad_tot:.0%} "
          f"({ad_tot:,} params)")

    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    l0 = float(ce_loss(llm, [build_seq(llm, adapter, v[i], ch[i], pair[i], "q?", "A", "correct") for i in range(4)]))
    for _ in range(20):
        v2, ch2, pair2 = adapter(out)
        loss = ce_loss(llm, [build_seq(llm, adapter, v2[i], ch2[i], pair2[i], "q?", "A", "correct") for i in range(4)])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[OK] adapter trains the bridge: CE {l0:.3f} -> {float(loss):.3f} over 20 steps")
    assert float(loss) < l0
    print("[PASS] Phase-2 FULL adapter self-test")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lafr-ckpt", type=Path, default=Path("results/lafr_eval_full_fixed/lafr_encoder.pt"))
    ap.add_argument("--data-root", type=Path,
                    default=Path("data/care_to_compare_extracted/CARE_To_Compare/Wind Farm C"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/llm_adapter_full"))
    ap.add_argument("--max-files", type=int, default=24)
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--stride", type=int, default=72)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--m-slots", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=8)
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
                patch=args.patch, n_heads=4, n_layers=2, lag_bins=LAG_BINS).to(device).eval()
    lafr.load_state_dict(torch.load(args.lafr_ckpt, map_location=device))
    for p in lafr.parameters():
        p.requires_grad_(False)
    print(f"[lafr] loaded frozen encoder from {args.lafr_ckpt}")

    # bf16 LLM: the gradient must flow THROUGH the frozen LLM to reach the adapter, so the
    # full activation graph is stored — bf16 halves that memory + speeds matmuls (fp32 0.6B
    # nearly OOMs a 6 GB laptop). Adapter stays fp32; assemble() casts tokens to the LLM dtype.
    llm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=llm_dtype)
    llm = FrozenLLM(model, tok, device)
    label_ids = llm.label_token_ids(["A", "B", "C", "D", "Y", "N"])
    print(f"[llm] {args.llm} frozen: d_llm={llm.d_llm}, label_ids={label_ids}")

    files = E.discover_csvs(args.data_root)[: args.max_files]
    channels = E.choose_channels(files, args.channels)
    raw = E.load_windows(files, channels, args.window, args.max_rows, args.stride)
    rng = np.random.default_rng(args.seed)
    raw = raw[rng.permutation(len(raw))]
    n_base = (args.n_train + args.n_eval) // 3 + 4
    qa_all = make_qa(raw[:n_base], rng, args.patch, [1, 2, 3, 4])
    rng.shuffle(qa_all)
    qa_train, qa_eval = qa_all[: args.n_train], qa_all[args.n_train: args.n_train + args.n_eval]
    print(f"[data] {len(files)} files, {len(channels)} ch -> QA train={len(qa_train)} eval={len(qa_eval)}")

    adapter = TemporalAdapter(lafr.event_dim, 64, llm.d_llm, P, m_slots=args.m_slots).to(device)
    print(f"[adapter] params={sum(p.numel() for p in adapter.parameters()):,}  m_slots={args.m_slots}")

    def grounding(tag):
        res = {m: answer_acc(llm, adapter, lafr, qa_eval, device, m, label_ids, args.batch_size)
               for m in ("correct", "shuffled", "no_token")}
        print(f"[{tag}] correct={res['correct']:.3f}  shuffled={res['shuffled']:.3f}  "
              f"no_token={res['no_token']:.3f}  (gap={res['correct']-res['shuffled']:+.3f})")
        return res

    pre = grounding("pre-train")
    print("[train] adapter only (LAFR + LLM frozen) ...")
    hist = train_adapter(llm, adapter, lafr, qa_train, device, args.epochs, args.lr, args.batch_size)
    post = grounding("post-train")

    by_task = {}
    for t in sorted({q.task for q in qa_eval}):
        sub = [q for q in qa_eval if q.task == t]
        by_task[t] = {m: answer_acc(llm, adapter, lafr, sub, device, m, label_ids, args.batch_size)
                      for m in ("correct", "shuffled", "no_token")}
        g = by_task[t]
        print(f"   task={t:5s} n={len(sub):3d}  correct={g['correct']:.3f}  "
              f"shuffled={g['shuffled']:.3f}  no_token={g['no_token']:.3f}  gap={g['correct']-g['shuffled']:+.3f}")

    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "train_ce": hist, "pre_train": pre, "post_train": post, "by_task": by_task,
              "n_train": len(qa_train), "n_eval": len(qa_eval)}
    (args.output_dir / "llm_adapter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(adapter.state_dict(), args.output_dir / "adapter.pt")
    print(f"\nwrote {args.output_dir / 'llm_adapter_report.json'}")
    grounded = sum(1 for t, g in by_task.items() if g['correct'] - g['shuffled'] > 0.1)
    print(f"GROUNDING: {grounded}/{len(by_task)} tasks grounded (correct-shuffled > 0.1)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        _smoke()
    else:
        main()
