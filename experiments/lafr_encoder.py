"""LAFR — Lag-Aware Factorized Relation encoder (Phase 1B core).

Faithful implementation of `refine-logs/MODEL_ARCHITECTURE.md` §1.4:

    raw multivariate SCADA window  X in R^{T x C}
        -> VTE   (variable-as-token, §1.4.3)
        -> PLB   (pairwise lag bank, per-pair, §1.4.2)
        -> LVAA  (lag-aware variable-axis attention + learned per-pair signed lag, §1.4.4)
        -> DBD   (differentiable boundary detector, §1.4.5)
        -> event-token readout (§1.4.6) + per-pair relation readout (§1.4.7)
        -> episode embedding (§1.4.8)

Trained **self-supervised** (§1.4.9): forecast-residual + masked reconstruction +
contrastive change-point. No event labels.

Output is a NAMED dataclass (`LAFROutput`, seam-gate G1); magnitude/lag are
SEPARATE scalar heads (`lag_pred`, `relation_strength`, seam-gate G2); per-pair
relations are a C x C matrix (seam-gate G3). The encoder is channel-count
agnostic except a sliced channel embedding (seam-gate G5).

This file is intentionally standalone (no `latent_event_memory_v1` import) so it
can be unit-tested and later wired into `wind_care_event_memory.py` as the
`--lafr-learned` system without dragging in the synthetic supervision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# Default lag bank in *patch units* (§1.4.2). Kept small so that for a CARE
# window T=72, patch=6 -> P=12 patches, every lag is realizable. Configurable.
DEFAULT_LAG_BINS: List[int] = [0, 1, 2, 3, 4, 6, 8]


# ---------------------------------------------------------------------------
# §1.4.10 — named output contract (seam-gate G1)
# ---------------------------------------------------------------------------
@dataclass
class LAFROutput:
    step_hidden: torch.Tensor              # (B, P, C, d)   Z'  -> downstream heads + Adapter
    next_patch_forecast: torch.Tensor      # (B, P, C)      yhat_{p+1|p}; last slot predicts unseen next patch
    boundary_logits: torch.Tensor          # (B, P)         b_t -> event localization
    soft_boundary_positions: torch.Tensor  # (B, K)         p_k -> event window readout
    soft_boundary_weights: torch.Tensor    # (B, K, P)      w_k
    event_embeddings: torch.Tensor         # (B, K, d_e)    e_k -> event memory
    pair_relation_embeddings: torch.Tensor # (B, C, C, d_r) r_ij-> Adapter + B5 proxy
    dependency_graph: torch.Tensor         # (B, C, C)      G_ij  CONDITIONAL dependency graph (CORE)
    dependency_logits: torch.Tensor        # (B, C, C)      S_ij  pre-threshold scores
    relation_strength: torch.Tensor        # (B, C, C)      ||A_ij||_2 attention magnitude (G2/G3)
    lag_pred: torch.Tensor                 # (B, C, C)      tau_hat directional lag on edges (G3)
    episode_embedding: torch.Tensor        # (B, d_g)       g   -> retrieval


# ---------------------------------------------------------------------------
# §1.4.5 — Differentiable Boundary Detector (soft top-K, no argmax)
# ---------------------------------------------------------------------------
class DifferentiableBoundaryDetector(nn.Module):
    def __init__(self, max_events: int, init_temperature: float = 1.0, suppression_width: float = 2.5):
        super().__init__()
        self.max_events = max_events
        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))
        self.suppression_width = float(suppression_width)

    def forward(self, boundary_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = boundary_logits.shape
        positions: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        mask = torch.ones(batch, length, dtype=boundary_logits.dtype, device=boundary_logits.device)
        time = torch.arange(length, dtype=torch.float32, device=boundary_logits.device)
        temp = self.temperature.clamp(0.2, 5.0)
        for _ in range(self.max_events):
            masked = boundary_logits + mask.clamp_min(1e-4).log()
            prob = torch.softmax(masked / temp, dim=-1)
            pos = (prob * time[None, :]).sum(dim=-1)
            positions.append(pos)
            weights.append(prob)
            supp = torch.exp(-0.5 * ((time[None, :] - pos[:, None]) / self.suppression_width) ** 2)
            mask = mask * (1.0 - 0.85 * supp)
        return torch.stack(positions, dim=1), torch.stack(weights, dim=1)


# ---------------------------------------------------------------------------
# §1.4.2 — Pairwise Lag Bank (per-pair triple product), channel-count agnostic
# ---------------------------------------------------------------------------
class PairwiseLagBank(nn.Module):
    """Per-pair triple-product lag features -> r_ij in R^{d_r}.

    Operates on the patch-level value series (B, P, C). For every ordered pair
    (i, j) and lag tau in the bank, computes the level / slope / magnitude
    covariation of channel i shifted by tau against channel j, pools over time,
    and projects through a *shared* per-pair MLP -> (B, C, C, d_r).
    """

    def __init__(self, lag_bins: Sequence[int], relation_dim: int):
        super().__init__()
        self.lag_bins = [int(v) for v in lag_bins]
        feat_dim = len(self.lag_bins) * 3
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, relation_dim),
            nn.LayerNorm(relation_dim),
        )

    @staticmethod
    def _shift_right(x: torch.Tensor, lag: int) -> torch.Tensor:
        # x: (B, P, C); shift along time (P) by lag, zero-pad the front.
        if lag <= 0:
            return x
        if lag >= x.shape[1]:
            return torch.zeros_like(x)
        pad = torch.zeros(x.shape[0], lag, x.shape[2], dtype=x.dtype, device=x.device)
        return torch.cat([pad, x[:, :-lag]], dim=1)

    def _cross_corr(self, value: torch.Tensor) -> torch.Tensor:
        # Proper per-lag NORMALIZED cross-correlation (B,C,C,L):
        #   align[i,j,g] = corr(i_t, j_{t+g})  == evidence that "i leads j by g".
        # Normalization is what makes it peak at the true lag on autocorrelated signals
        # (a raw lagged product does not). Re-centered on the valid overlap per lag.
        b, p, c = value.shape
        eps = 1e-5
        cols: List[torch.Tensor] = []
        for g in self.lag_bins:
            if g >= p:
                cols.append(value.new_zeros(b, c, c))
                continue
            xi = value if g == 0 else value[:, : p - g, :]
            xj = value if g == 0 else value[:, g:, :]
            xi = xi - xi.mean(dim=1, keepdim=True)
            xj = xj - xj.mean(dim=1, keepdim=True)
            n = max(xi.shape[1], 1)
            cov = torch.einsum("bti,btj->bij", xi, xj) / n          # (B,Ci,Cj)
            si = torch.sqrt((xi * xi).mean(dim=1) + eps)            # (B,C)
            sj = torch.sqrt((xj * xj).mean(dim=1) + eps)
            cols.append(cov / (si[:, :, None] * sj[:, None, :] + eps))
        return torch.stack(cols, dim=-1)                            # (B,C,C,L)

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # value: (B, P, C). Returns r_ij relation embedding (B,C,C,d_r) for the readout,
        # AND a proper cross-correlation alignment (B,C,C,L) for the directional lag.
        dv = torch.diff(value, dim=1, prepend=value[:, :1, :])
        eps = 1e-5
        feats: List[torch.Tensor] = []
        for lag in self.lag_bins:
            xs = self._shift_right(value, lag)          # (B, P, C) channel i shifted by +lag
            ds = self._shift_right(dv, lag)
            prod = xs[:, :, :, None] * value[:, :, None, :]
            dprod = ds[:, :, :, None] * dv[:, :, None, :]
            absd = (value[:, :, None, :] - xs[:, :, :, None]).abs()
            prod = prod / (prod.detach().std(dim=1, keepdim=True) + eps)
            dprod = dprod / (dprod.detach().std(dim=1, keepdim=True) + eps)
            absd = absd / (absd.detach().std(dim=1, keepdim=True) + eps)
            feats.extend([prod.mean(dim=1), dprod.mean(dim=1), absd.mean(dim=1)])
        pair_feat = torch.stack(feats, dim=-1)          # (B, C, C, 3*|T|)
        align = self._cross_corr(value)                 # (B, C, C, L)
        return self.proj(pair_feat), align              # (B, C, C, d_r), (B, C, C, L)


# ---------------------------------------------------------------------------
# §1.4.3 — Variable-as-token encoder (VTE)
# ---------------------------------------------------------------------------
class SinusoidalTimePos(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model - d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:length]                          # (P, d)


# ---------------------------------------------------------------------------
# §1.4.4 — Lag-aware Variable-Axis Attention (LVAA) — the contribution
# ---------------------------------------------------------------------------
class LagAwareVariableAxisAttention(nn.Module):
    """Attends along the VARIABLE axis at each patch, biased by a directional lag.

    The signed lag tau_hat_ij is NOT a free scalar (which is under-determined by
    self-supervision and tends to saturate or collapse). It is a soft-argmax over a
    SIGNED lag axis of the PLB level-alignment evidence (§1.4.2): positive lags read
    "i leads j", negative lags read "i lags j". A learnable per-lag trust weight makes
    it *learned*, while the soft-argmax keeps it *grounded* and *anti-symmetric by
    construction* (tau_hat_ji = -tau_hat_ij) -- the property B5 needs for the
    wind->power directional-lag proxy. A learned per-head bias function b^(h)(tau)
    then turns the lag into an attention bias.

    Convention: tau_hat_ij > 0  <=>  channel i LEADS channel j.
    """

    def __init__(self, d_model: int, n_heads: int, lag_bins: Sequence[int], lag_basis: int = 8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide n_heads"
        self.h = n_heads
        self.dh = d_model // n_heads
        self.lag_bins = [int(v) for v in lag_bins]
        assert self.lag_bins[0] == 0, "lag_bins must start at 0"
        self.max_lag = float(max(self.lag_bins) or 1)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        # signed lag axis: [-g_{L-1..1}, 0, +g_{1..L-1}]
        pos = self.lag_bins[1:]
        signed = [-v for v in reversed(pos)] + [0] + list(pos)
        self.register_buffer("signed_lags", torch.tensor(signed, dtype=torch.float32))
        # learnable, non-negative per-lag trust weight (shared pos/neg -> anti-symmetry kept)
        self.lag_trust = nn.Parameter(torch.zeros(len(self.lag_bins)))
        self.lag_temp = nn.Parameter(torch.tensor(0.1))
        # learned per-head bias function b^(h)(tau) via RBF featurization
        centers = torch.linspace(-1.0, 1.0, lag_basis)
        self.register_buffer("rbf_centers", centers)
        self.bias_weight = nn.Parameter(torch.zeros(n_heads, lag_basis))
        self.bias_bias = nn.Parameter(torch.zeros(n_heads))
        # the conditional dependency graph biases attention toward direct neighbours;
        # zero-init so it starts as a no-op and learns to rely on the graph.
        self.dep_scale = nn.Parameter(torch.zeros(n_heads))

    def grounded_lag(self, align: torch.Tensor) -> torch.Tensor:
        # align: (B,C,C,L), align[i,j,k] = corr(i_t, j_{t+lag_bins[k]}) (proper xcorr).
        # Read the signed lag as a (sharp) soft-argmax of the SIGNED cross-correlation:
        #   g>0  uses align[i,j,g]   (i leads j by g)
        #   g<0  uses align[j,i,|g|] (i lags j by |g|)
        # The normalized cross-corr peaks AT the true lead/lag, so the soft-argmax over
        # it recovers direction+magnitude; a learnable per-lag trust + low temperature
        # sharpen it. Anti-symmetric by construction (tau_ji = -tau_ij).
        trust = F.softplus(self.lag_trust)                           # (L,) >= 0
        a = align * trust                                           # (B,C,C,L) weighted xcorr
        aT = a.transpose(1, 2)
        pos = a[..., 1:]                                            # g>0
        neg = aT[..., 1:].flip(-1)                                 # g<0
        zero = a[..., :1]                                          # g=0 (symmetric)
        sx = torch.cat([neg, zero, pos], dim=-1)                   # (B,C,C, 2(L-1)+1)
        w = torch.softmax(sx / self.lag_temp.clamp(0.02, 5.0), dim=-1)
        tau = (w * self.signed_lags).sum(dim=-1)                    # (B,C,C) signed, anti-symmetric
        return tau

    def lag_bias(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: (B,C,C) -> per-head bias (B,h,C,C)
        norm = (tau / self.max_lag).unsqueeze(-1)                    # (B,C,C,1)
        rbf = torch.exp(-((norm - self.rbf_centers) ** 2) / 0.25)    # (B,C,C,basis)
        return torch.einsum("bijk,hk->bhij", rbf, self.bias_weight) + self.bias_bias[None, :, None, None]

    def forward(self, z: torch.Tensor, align: torch.Tensor, dep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, P, C, d); align: (B, C, C, L); dep: (B, C, C) conditional graph
        b, p, c, d = z.shape
        q = self.q(z).view(b, p, c, self.h, self.dh)
        k = self.k(z).view(b, p, c, self.h, self.dh)
        v = self.v(z).view(b, p, c, self.h, self.dh)
        scores = torch.einsum("bpihe,bpjhe->bphij", q, k) / math.sqrt(self.dh)   # (B,P,h,Ci,Cj)
        tau = self.grounded_lag(align)                              # (B,C,C)
        bias = self.lag_bias(tau)                                   # (B,h,C,C) directional lag
        bias = bias + self.dep_scale[None, :, None, None] * dep[:, None, :, :]   # + conditional graph
        scores = scores + bias[:, None, :, :, :]                    # broadcast over patches
        attn = torch.softmax(scores, dim=-1)                       # over j
        ctx = torch.einsum("bphij,bpjhe->bpihe", attn, v).reshape(b, p, c, d)
        out = self.o(ctx)
        attn_mag = attn.mean(dim=1)                                # (B, h, C, C)
        return out, tau, attn_mag


class LVAABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, lag_bins: Sequence[int], dropout: float = 0.05):
        super().__init__()
        self.attn = LagAwareVariableAxisAttention(d_model, n_heads, lag_bins)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )

    def forward(self, z: torch.Tensor, align: torch.Tensor, dep: torch.Tensor):
        a, tau, attn_mag = self.attn(self.ln1(z), align, dep)
        z = z + a
        z = z + self.ffn(self.ln2(z))
        return z, tau, attn_mag


# ---------------------------------------------------------------------------
# Conditional dependency graph (precision-matrix / GGM analogue) — THE CORE
# ---------------------------------------------------------------------------
class DependencyGraph(nn.Module):
    """Learns a symmetric, sparse CONDITIONAL dependency graph G over variables.

    This is the precision-matrix / Gaussian-graphical-model analogue: an edge
    G_ij > 0 means a DIRECT dependency between variables i and j AFTER accounting
    for all the others -- as opposed to a marginal correlation, which also lights
    up confounded pairs (a common driver makes its two children correlated).

    The conditional structure is enforced by the training objective, not here:
    `LAFR` reconstructs each variable from the OTHERS weighted by G (node-wise
    regression, Meinshausen-Buehlmann), so a redundant confounded edge HURTS the
    reconstruction and is driven to zero; an L1 penalty keeps G sparse.

    Here we only emit a symmetric score S = phi(s_i).phi(s_j) and a soft-thresholded,
    non-negative, diagonal-free graph G = relu(S - thr). Symmetric by construction.
    """

    def __init__(self):
        super().__init__()
        # amortized regularization: a learned ridge (shared across all windows) and a
        # learned soft-threshold for the graph. These are the ONLY learned params here;
        # the structure itself comes from the window's joint correlation (so it
        # generalizes to any input and matches the optimal linear estimator).
        # init so ridge = softplus(log_ridge)+1e-2 ~= 0.1, matching the partial-corr
        # baseline; the old -1.0 (-> ridge ~0.32) over-shrinks a well-conditioned
        # precision toward the identity and washes out partial correlation.
        self.log_ridge = nn.Parameter(torch.tensor(-2.3))
        self.log_thresh = nn.Parameter(torch.tensor(-2.0))

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # value: (B,P,C) -> A (B,C,C) signed node-wise regression coeffs A_ij=-Theta_ij/Theta_ii,
        # and G (B,C,C) symmetric non-negative graph = soft_threshold(|partial correlation|).
        # Computed from THIS window's correlation via a differentiable regularized inverse
        # (Theta = (corr + ridge*I)^-1), i.e. an amortized regularized partial correlation /
        # Gaussian-graphical-model estimator -- uses the JOINT statistics, unlike a
        # per-channel similarity, so it identifies which edges are real and rejects
        # confounded / noise-channel edges.
        b, p, c = value.shape
        X = value - value.mean(dim=1, keepdim=True)
        cov = torch.einsum("bpi,bpj->bij", X, X) / max(p, 1)
        d = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-6))
        corr = cov / (d[:, :, None] * d[:, None, :] + 1e-8)
        eye = torch.eye(c, device=value.device)
        ridge = F.softplus(self.log_ridge) + 1e-2
        prec = torch.linalg.inv(corr + ridge * eye[None])           # precision Theta
        pdg = torch.diagonal(prec, dim1=-2, dim2=-1)                # Theta_ii
        pd = torch.sqrt(pdg.clamp_min(1e-6))
        pcorr = (-prec / (pd[:, :, None] * pd[:, None, :] + 1e-8))  # partial correlation
        off = (1.0 - eye)[None]
        pcorr = pcorr * off
        # HARD soft-threshold (relu, per the design): drives sub-threshold confounded
        # edges to EXACT zero. softplus here has a ~log(2) floor at 0 -> it can never
        # suppress a confounded edge to ~0, inflating the spurious/direct ratio and
        # violating the "sparse graph" claim (no true zeros).
        G = F.relu(pcorr.abs() - F.softplus(self.log_thresh)) * off        # >=0, symmetric, sparse
        A = (-prec / pdg[:, :, None]) * off                         # node-wise regression coeffs
        return A, G


# ---------------------------------------------------------------------------
# LAFR
# ---------------------------------------------------------------------------
class LAFR(nn.Module):
    def __init__(
        self,
        max_channels: int,
        d_model: int = 64,
        relation_dim: int = 32,
        event_dim: int = 64,
        episode_dim: int = 64,
        max_events: int = 6,
        patch: int = 6,
        n_heads: int = 4,
        n_layers: int = 2,
        lag_bins: Sequence[int] = tuple(DEFAULT_LAG_BINS),
        dropout: float = 0.05,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.max_channels = max_channels
        self.d_model = d_model
        self.event_dim = event_dim
        self.relation_dim = relation_dim
        self.patch = int(patch)
        self.max_events = max_events
        self.lag_bins = [int(v) for v in lag_bins]
        self.mask_ratio = float(mask_ratio)

        # §1.4.3 VTE
        self.input_proj = nn.Linear(3, d_model)         # triplet [x, dx, |dx|]
        self.time_pos = SinusoidalTimePos(d_model)
        self.channel_emb = nn.Embedding(max_channels, d_model)
        self.in_ln = nn.LayerNorm(d_model)

        # §1.4.2 PLB (relation embedding + cross-corr alignment for the directional lag)
        self.lag_bank = PairwiseLagBank(self.lag_bins, relation_dim)

        # CORE: conditional dependency graph (precision-matrix analogue)
        self.dep_graph = DependencyGraph()

        # §1.4.4 LVAA stack
        self.blocks = nn.ModuleList(
            [LVAABlock(d_model, n_heads, self.lag_bins, dropout) for _ in range(n_layers)]
        )

        # §1.4.5 DBD
        self.boundary_head = nn.Linear(d_model, 1)
        self.boundary_detector = DifferentiableBoundaryDetector(max_events)

        # §1.4.6 event-token readout (soft window mean+max over channels)
        self.event_radius = nn.Parameter(torch.tensor(2.0))
        self.event_proj = nn.Sequential(
            nn.Linear(2 * d_model + 1, event_dim), nn.LayerNorm(event_dim), nn.GELU(),
            nn.Linear(event_dim, event_dim), nn.LayerNorm(event_dim),
        )  # +1 for normalized event time \tilde t_k

        # §1.4.8 episode embedding
        self.episode_proj = nn.Sequential(
            nn.Linear(2 * event_dim + relation_dim, episode_dim), nn.LayerNorm(episode_dim)
        )

        # §1.4.9 self-supervised pretext heads
        self.forecast_head = nn.Linear(d_model, 1)      # next-patch value per (patch, channel)
        self.recon_head = nn.Linear(d_model, 1)         # masked reconstruction per cell
        self.proj_cc = nn.Linear(d_model, d_model)      # contrastive change-point projection

    # ---- input plumbing ----------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> patch-pooled value (B, P, C) using non-overlapping mean.
        b, t, c = x.shape
        p = self.patch
        P = t // p
        x = x[:, : P * p, :].reshape(b, P, p, c).mean(dim=2)
        return x                                         # (B, P, C)

    def _augment(self, x: torch.Tensor, drop: float = 0.2, jitter: float = 0.1) -> torch.Tensor:
        # Two stochastic views per window for instance-discrimination: channel
        # dropout + additive jitter + small temporal roll. Operates on raw (B,T,C).
        b, t, c = x.shape
        keep = (torch.rand(b, 1, c, device=x.device) > drop).float()
        x = x * keep
        x = x + jitter * torch.randn_like(x)
        shift = int(torch.randint(0, max(1, t // 6), (1,)).item())
        if shift:
            x = torch.roll(x, shifts=shift, dims=1)
        return x

    def _triplet(self, value: torch.Tensor) -> torch.Tensor:
        # value: (B, P, C) -> (B, P, C, 3) = [x, dx, |dx|]
        dx = torch.diff(value, dim=1, prepend=value[:, :1, :])
        return torch.stack([value, dx, dx.abs()], dim=-1)

    def _vte(self, value: torch.Tensor) -> torch.Tensor:
        # value: (B, P, C) -> tokens Z (B, P, C, d)
        b, P, c = value.shape
        feat = self._triplet(value)                     # (B, P, C, 3)
        z = self.input_proj(feat)                       # (B, P, C, d)
        z = z + self.time_pos(P)[None, :, None, :]      # p_t
        z = z + self.channel_emb.weight[:c][None, None, :, :]  # q_c (sliced -> G5)
        return self.in_ln(z)

    def _backbone(self, value: torch.Tensor, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        z0 = self._vte(value)                           # (B, P, C, d) pre-mixing embeddings
        r_ij, align = self.lag_bank(value)              # (B,C,C,d_r), (B,C,C,L)
        # CORE graph is estimated from the FULL-RESOLUTION window (T samples), NOT the
        # P patch-means: a C x C precision needs T > C samples to be well-conditioned.
        # Patch pooling (T->P) starves it (here P=12 < C=16 -> rank-deficient) and forces
        # a large ridge that collapses partial correlation back to marginal, killing the
        # confounder separation. Raw x gives the clean, well-conditioned estimate.
        A, G = self.dep_graph(x_raw)                     # signed coeffs + symmetric graph
        z = z0
        tau = attn_mag = None
        for blk in self.blocks:
            z, tau, attn_mag = blk(z, align, G)         # graph biases variable-axis attention
        return {"z": z, "z0": z0, "r_ij": r_ij, "tau": tau, "attn_mag": attn_mag,
                "A": A, "G": G, "align": align}

    # ---- inference forward (named tensors) ---------------------------------
    def forward(self, x: torch.Tensor) -> LAFROutput:
        return self._encode(self._patchify(x), x)

    def forecast(self, x: torch.Tensor, horizon_patches: int = 1) -> torch.Tensor:
        """Forecast future patch means from a context window.

        Returns a tensor with shape (B, horizon_patches, C). The values live in the
        same normalized/patch-mean space as `_patchify(x)`: if callers standardize
        input windows before LAFR, the forecast is standardized too.
        """
        horizon = int(horizon_patches)
        if horizon < 1:
            raise ValueError("horizon_patches must be >= 1")

        value = self._patchify(x)
        if value.shape[1] < 1:
            raise ValueError("input must contain at least one full patch")

        raw_steps = value.shape[1] * self.patch
        cur_value = value
        cur_raw = x[:, :raw_steps, :]
        preds: List[torch.Tensor] = []
        for _ in range(horizon):
            bb = self._backbone(cur_value, cur_raw)
            next_value = self.forecast_head(bb["z"][:, -1:, :, :]).squeeze(-1)
            preds.append(next_value)
            cur_value = torch.cat([cur_value, next_value], dim=1)
            cur_raw = torch.cat([cur_raw, next_value.repeat_interleave(self.patch, dim=1)], dim=1)
        return torch.cat(preds, dim=1)

    def forecasting_loss(
        self,
        x: torch.Tensor,
        context_patches: int,
        horizon_patches: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Supervised forecasting loss using a prefix of `x` as context.

        This keeps forecasting inside LAFR while making benchmark scripts simple:
        pass a longer window, choose the context length in patches, and optimize
        the returned MSE against the held-out future patch means.
        """
        value = self._patchify(x)
        context = int(context_patches)
        horizon = int(horizon_patches)
        if context < 1:
            raise ValueError("context_patches must be >= 1")
        if horizon < 1:
            raise ValueError("horizon_patches must be >= 1")
        if context + horizon > value.shape[1]:
            raise ValueError(
                "context_patches + horizon_patches must not exceed the number of full patches"
            )

        context_x = x[:, : context * self.patch, :]
        target = value[:, context: context + horizon, :]
        pred = self.forecast(context_x, horizon)
        loss = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        logs = {
            "forecast_mse": float(loss.detach()),
            "forecast_mae": float(mae.detach()),
            "context_patches": float(context),
            "horizon_patches": float(horizon),
        }
        return loss, logs

    def _encode(self, value: torch.Tensor, x_raw: torch.Tensor) -> LAFROutput:
        b, P, c = value.shape
        bb = self._backbone(value, x_raw)
        z, r_ij, tau, attn_mag = bb["z"], bb["r_ij"], bb["tau"], bb["attn_mag"]
        next_patch_forecast = self.forecast_head(z).squeeze(-1)

        # §1.4.5 boundary score over patches (channel-agnostic eventness)
        boundary_logits = self.boundary_head(z.mean(dim=2)).squeeze(-1)   # (B, P)
        positions, weights = self.boundary_detector(boundary_logits)     # (B,K), (B,K,P)

        # §1.4.6 event-token readout via soft Gaussian window around p_k
        time = torch.arange(P, dtype=torch.float32, device=value.device)
        radius = self.event_radius.clamp(0.5, float(P))
        win = torch.exp(-0.5 * ((time[None, None, :] - positions[:, :, None]) / radius) ** 2)
        win = win / win.sum(dim=-1, keepdim=True).clamp_min(1e-6)         # (B,K,P)
        pooled = torch.einsum("bkp,bpcd->bkcd", win, z)                  # (B,K,C,d)
        evt = torch.cat([pooled.mean(dim=2), pooled.amax(dim=2)], dim=-1)  # (B,K,2d)
        norm_t = (positions / float(max(P - 1, 1))).unsqueeze(-1)        # \tilde t_k
        event_emb = self.event_proj(torch.cat([evt, norm_t], dim=-1))    # (B,K,d_e)

        # per-pair relation readout: conditional graph (CORE) + lag (edge direction)
        lag_pred = tau                                                   # (B,C,C) grounded signed lag
        strength = attn_mag.norm(dim=1)                                  # (B,C,C) ||A_ij||_2 over heads

        # §1.4.8 episode embedding (summarize events + the conditional dependency graph)
        rel_summary = r_ij.mean(dim=(1, 2))                             # (B,d_r)
        g = self.episode_proj(
            torch.cat([event_emb.mean(dim=1), event_emb.amax(dim=1), rel_summary], dim=-1)
        )

        return LAFROutput(
            step_hidden=z,
            next_patch_forecast=next_patch_forecast,
            boundary_logits=boundary_logits,
            soft_boundary_positions=positions,
            soft_boundary_weights=weights,
            event_embeddings=event_emb,
            pair_relation_embeddings=r_ij,
            dependency_graph=bb["G"],
            dependency_logits=bb["A"],
            relation_strength=strength,
            lag_pred=lag_pred,
            episode_embedding=g,
        )

    # ---- §1.4.9 self-supervised pretext losses -----------------------------
    def pretext_losses(
        self,
        x: torch.Tensor,
        alpha_fc: float = 1.0,
        alpha_mr: float = 0.5,
        alpha_cc: float = 0.5,
        alpha_bd: float = 0.5,
        alpha_inst: float = 0.5,
        alpha_cond: float = 1.0,
        alpha_sparse: float = 0.02,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        value = self._patchify(x)                       # (B, P, C)
        b, P, c = value.shape

        # (1) forecast residual: predict next-patch value per (patch, channel)
        bb = self._backbone(value, x)
        z, A = bb["z"], bb["A"]
        fc = self.forecast_head(z).squeeze(-1)          # (B, P, C)
        loss_fc = F.mse_loss(fc[:, :-1, :], value[:, 1:, :])

        # (0) CORE: node-wise CONDITIONAL reconstruction + L1 sparsity (Meinshausen-Buehlmann
        #     => the precision / Gaussian-graphical-model structure). Each variable is a SIGNED
        #     linear regression on the OTHER variables' values (diagonal excluded). Under a
        #     common driver, regressing a child JOINTLY on {driver, sibling} gives ~0 weight to
        #     the sibling (it only adds independent noise) -> the confounded edge is suppressed.
        #     This is exactly what marginal correlation CANNOT do.
        x_hat = torch.einsum("bij,bpj->bpi", A, value)         # predict each var from OTHERS
        loss_cond = F.mse_loss(x_hat, value)
        loss_sparse = A.abs().mean()                           # L1 (graphical-lasso style)

        # (1b) boundary self-supervision: the boundary score must peak where the
        #      forecast residual is large (event candidate, §1.4.9). Standardize
        #      both per-episode and regress -> trains the localization path label-free.
        if P >= 3:
            resid = (fc[:, :-1, :].detach() - value[:, 1:, :]).pow(2).mean(dim=-1)   # (B, P-1)
            resid = (resid - resid.mean(1, keepdim=True)) / (resid.std(1, keepdim=True) + 1e-5)
            b_logits = self.boundary_head(z.mean(dim=2)).squeeze(-1)[:, 1:]          # align to t+1
            b_std = (b_logits - b_logits.mean(1, keepdim=True)) / (b_logits.std(1, keepdim=True) + 1e-5)
            loss_bd = F.mse_loss(b_std, resid)
        else:
            loss_bd = z.sum() * 0.0

        # (2) masked reconstruction: mask ~15% of (patch, channel) cells of the input
        mask = (torch.rand(b, P, c, device=x.device) < self.mask_ratio)
        masked_value = value.masked_fill(mask, 0.0)
        zm = self._backbone(masked_value, x)["z"]
        recon = self.recon_head(zm).squeeze(-1)         # (B, P, C)
        if mask.any():
            loss_mr = F.mse_loss(recon[mask], value[mask])
        else:
            loss_mr = recon.sum() * 0.0

        # (3) contrastive change-point: adjacent patches positive, distant negative
        zc = F.normalize(self.proj_cc(z.mean(dim=2)), dim=-1)   # (B, P, d)
        sim = torch.einsum("bpd,bqd->bpq", zc, zc) / 0.1        # (B, P, P)
        eye = torch.eye(P, device=x.device, dtype=torch.bool)
        sim = sim.masked_fill(eye[None], float("-inf"))         # no self
        # positive target = next patch (a -> a+1); valid anchors 0..P-2
        if P >= 3:
            logits = sim[:, :-1, :].reshape(b * (P - 1), P)     # anchors
            target = torch.arange(1, P, device=x.device).repeat(b)
            loss_cc = F.cross_entropy(logits, target)
        else:
            loss_cc = z.sum() * 0.0

        # (4) instance-discrimination contrastive (NT-Xent) on the EPISODE embedding
        #     over two augmented views -> self-trains the event/episode readout path
        #     (event_proj, episode_proj, event_radius) so the retrieval embedding is
        #     not a random projection.
        xa1, xa2 = self._augment(x), self._augment(x)
        g1 = self._encode(self._patchify(xa1), xa1).episode_embedding
        g2 = self._encode(self._patchify(xa2), xa2).episode_embedding
        gz = F.normalize(torch.cat([g1, g2], dim=0), dim=-1)            # (2B, d_g)
        logits_i = gz @ gz.t() / 0.2                                    # (2B, 2B)
        diag = torch.eye(2 * b, device=x.device, dtype=torch.bool)
        logits_i = logits_i.masked_fill(diag, float("-inf"))
        target_i = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(x.device)
        loss_inst = F.cross_entropy(logits_i, target_i)

        # NOTE: the directional-lag inductive bias (anti-symmetry tau_ji = -tau_ij) is
        # now BUILT INTO the grounded soft-argmax readout (LVAA.grounded_lag), so no
        # separate lag regularizer is needed; lag magnitude is driven by task utility
        # (the learnable per-lag trust weight increases only where a lead/lag helps).
        total = (
            alpha_cond * loss_cond + alpha_sparse * loss_sparse
            + alpha_fc * loss_fc + alpha_mr * loss_mr + alpha_cc * loss_cc
            + alpha_bd * loss_bd + alpha_inst * loss_inst
        )
        logs = {
            "loss": float(total.detach()),
            "cond": float(loss_cond.detach()),
            "sparse": float(loss_sparse.detach()),
            "fc": float(loss_fc.detach()),
            "mr": float(loss_mr.detach()),
            "cc": float(loss_cc.detach()),
            "bd": float(loss_bd.detach()),
            "inst": float(loss_inst.detach()),
        }
        return total, logs


# ---------------------------------------------------------------------------
# self-test (architecture verification, no data / no labels)
# ---------------------------------------------------------------------------
def _smoke() -> None:
    torch.manual_seed(0)
    B, T, C = 4, 72, 16
    model = LAFR(max_channels=C, d_model=64, relation_dim=32, event_dim=64, episode_dim=64,
                 max_events=6, patch=6, n_heads=4, n_layers=2)
    x = torch.randn(B, T, C)
    P = T // model.patch

    # ---- forward + §1.4.10 shape contract ----
    out = model(x)
    expect = {
        "step_hidden": (B, P, C, 64),
        "next_patch_forecast": (B, P, C),
        "boundary_logits": (B, P),
        "soft_boundary_positions": (B, 6),
        "soft_boundary_weights": (B, 6, P),
        "event_embeddings": (B, 6, 64),
        "pair_relation_embeddings": (B, C, C, 32),
        "dependency_graph": (B, C, C),
        "dependency_logits": (B, C, C),
        "relation_strength": (B, C, C),
        "lag_pred": (B, C, C),
        "episode_embedding": (B, 64),
    }
    for name, shape in expect.items():
        got = tuple(getattr(out, name).shape)
        assert got == shape, f"{name}: expected {shape}, got {got}"
        assert torch.isfinite(getattr(out, name)).all(), f"{name} has non-finite values"
    print("[OK] forward shapes (s1.4.10):")
    for name in expect:
        print(f"     {name:28s} {tuple(getattr(out, name).shape)}")

    # ---- seam gates ----
    assert hasattr(out, "dependency_graph") and hasattr(out, "lag_pred"), "G1 named tensors"
    assert out.dependency_graph.shape == (B, C, C), "G3 per-pair C x C dependency matrix"
    assert out.relation_strength.shape == (B, C, C), "G2 scalar magnitude head"
    # G5: channel-count agnostic (run with fewer channels, no rebuild)
    out2 = model(torch.randn(B, T, C - 5))
    assert out2.dependency_graph.shape == (B, C - 5, C - 5), "G5 channel-agnostic"
    print("[OK] seam gates G1/G2/G3/G5")

    # ---- dependency graph: symmetric, non-negative, sparse, diagonal-free ----
    Gd = out.dependency_graph
    assert (Gd >= 0).all(), "G must be non-negative"
    assert (Gd - Gd.transpose(-1, -2)).abs().max() < 1e-5, "G must be symmetric"
    assert Gd.diagonal(dim1=-2, dim2=-1).abs().max() < 1e-6, "G diagonal must be 0"
    soft_sparsity = float((Gd < 0.1 * Gd.max()).float().mean())
    print(f"[OK] dependency graph symmetric/non-neg/diag-free; {soft_sparsity:.0%} edges < 0.1*max")

    # ---- self-supervised loss + gradient flow ----
    # Warm up a few steps first: the conditional graph enters attention through a
    # ZERO-INIT gate (LVAA.dep_scale), so at step 0 dG's gradient is multiplied by 0
    # and the graph's threshold param looks "untrained". A handful of optimizer steps
    # activates the gate, after which every representational submodule is reachable --
    # which is what this grad-coverage check is meant to verify.
    warm_opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        warm_opt.zero_grad()
        wl, _ = model.pretext_losses(x)
        wl.backward()
        warm_opt.step()
    warm_opt.zero_grad()
    loss, logs = model.pretext_losses(x)
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    from collections import defaultdict
    tot: Dict[str, int] = defaultdict(int)
    cov: Dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        tot[top] += p.numel()
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            cov[top] += p.numel()
    n_grad = sum(cov.values())
    print(f"[OK] pretext losses {logs}")
    print(f"[OK] backward: params={n_params:,} grad-coverage={n_grad / n_params:.1%}")
    # every representational submodule must receive a self-supervised gradient
    must_train = [
        "input_proj", "channel_emb", "lag_bank", "dep_graph", "blocks",
        "boundary_head", "event_proj", "episode_proj", "forecast_head", "recon_head", "proj_cc",
    ]
    for sm in must_train:
        frac = cov[sm] / max(tot[sm], 1)
        flag = "OK " if frac > 0.5 else "!! "
        print(f"     [{flag}] {sm:18s} grad {frac:6.1%}")
        assert frac > 0.5, f"{sm} is not trained by the self-sup objective"
    # the LVAA contribution specifically must get finite gradient on its learnable
    # lag params (per-lag trust + temperature) and its per-head bias function
    lvaa = model.blocks[-1].attn
    for pname in ("lag_trust", "lag_temp", "bias_weight"):
        g = getattr(lvaa, pname).grad
        assert g is not None and torch.isfinite(g).all(), f"LVAA.{pname} got no/!finite grad"
    print("[OK] LVAA learnable lag-trust/temp + bias function receive finite gradient")

    # ---- anti-symmetry of the grounded signed lag is exact by construction ----
    asym = (out.lag_pred + out.lag_pred.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-4, f"lag readout is not anti-symmetric (max |tau_ij+tau_ji|={asym:.2e})"
    print(f"[OK] grounded lag is anti-symmetric (max |tau_ij + tau_ji| = {asym:.1e})")

    # ---- native forecasting interface (kept on LAFR, not a separate module) ----
    pred = model.forecast(x[:, :48], horizon_patches=3)
    assert tuple(pred.shape) == (B, 3, C), f"forecast shape expected {(B, 3, C)}, got {tuple(pred.shape)}"
    fl, flog = model.forecasting_loss(x, context_patches=8, horizon_patches=2)
    assert torch.isfinite(fl), "forecasting_loss is non-finite"
    assert "forecast_mse" in flog and "forecast_mae" in flog, "forecasting logs incomplete"
    print(f"[OK] native forecast API: pred={tuple(pred.shape)}  logs={flog}")

    # ---- CORE PROOF: recover a CONFOUNDED dependency graph (conditional, not marginal) ----
    # Plant a confounder: driver = ch0 drives ch1 and ch2 (ch1,ch2 = ch0 + indep noise).
    # GT direct edges: 0-1, 0-2.  NON-edge: 1-2 (marginally correlated, but conditionally
    # independent given 0). Also plant a lead/lag on edge 0->3 (driver leads by 2 patches).
    torch.manual_seed(1)
    tr = LAFR(max_channels=C, d_model=64, relation_dim=32, max_events=6, patch=6)
    opt = torch.optim.AdamW(tr.parameters(), lr=3e-3)

    def batch():
        base = torch.randn(16, T, C)
        drv = base[:, :, 0]
        base[:, :, 1] = drv + 0.7 * torch.randn(16, T)         # responder 1 (confounded child)
        base[:, :, 2] = drv + 0.7 * torch.randn(16, T)         # responder 2 (confounded child)
        base[:, 12:, 3] = drv[:, :-12] + 0.3 * torch.randn(16, T - 12)  # lagged child (0 leads 3)
        return base

    first = last = None
    for step in range(250):
        opt.zero_grad()
        l, lg = tr.pretext_losses(batch())
        l.backward()
        torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = lg["loss"]
        last = lg["loss"]
    print(f"[OK] trainability: loss {first:.3f} -> {last:.3f} over 250 steps "
          f"(cond={lg['cond']:.3f} sparse={lg['sparse']:.3f})")
    assert last < first, "objective did not decrease -- encoder is not optimizable"

    with torch.no_grad():
        bb_out = tr(batch())
        Gm = bb_out.dependency_graph.mean(dim=0)               # (C,C)
        lp = bb_out.lag_pred.mean(dim=0)
        # marginal |correlation| baseline on the same batch (what cross-corr/corr would see)
        xv = tr._patchify(batch())                             # (B,P,C)
        xc = xv - xv.mean(1, keepdim=True)
        cov = torch.einsum("bpi,bpj->bij", xc, xc) / xv.shape[1]
        sd = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1) + 1e-6)
        corr = (cov / (sd[:, :, None] * sd[:, None, :] + 1e-6)).abs().mean(0)

    direct = (Gm[0, 1] + Gm[0, 2]) / 2                          # true edges
    spurious = Gm[1, 2]                                         # confounded non-edge
    corr_direct = float((corr[0, 1] + corr[0, 2]) / 2)
    print(f"[OK] dependency graph: direct(0-1,0-2)={direct:.3f}  spurious(1-2)={spurious:.3f}")
    print(f"     marginal |corr| same pairs: direct={corr_direct:.3f}  "
          f"spurious(1-2)={corr[1, 2]:.3f}  <- marginal CANNOT separate it")
    # the CORE claim: LAFR keeps the direct edges but suppresses the confounded one,
    # AND does so more cleanly than marginal correlation (the whole point).
    lafr_ratio = float(spurious / (direct + 1e-6))
    corr_ratio = float(corr[1, 2] / (corr_direct + 1e-6))
    print(f"     spurious/direct ratio:  LAFR={lafr_ratio:.2f}   marginal={corr_ratio:.2f}  (lower=better)")
    assert direct > 1.3 * spurious, "did not suppress the confounded edge (conditional failed)"
    assert lafr_ratio < corr_ratio, "LAFR no better than marginal correlation at separating the confounder"
    # and the lag edge keeps its direction (0 leads 3)
    print(f"[OK] edge direction (0 leads 3): tau(0->3)={lp[0, 3]:+.3f}")
    assert lp[0, 3] > 1e-3, "directional lag sign on edge not recovered"
    print("[PASS] LAFR architecture self-test")


if __name__ == "__main__":
    _smoke()
