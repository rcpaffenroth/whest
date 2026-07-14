# PLAN v1 — A UT-anchored recurrent corrector, built as one architecture with port ablations

*2026-07-13. Companion to `RCP_ideas_v1.md` (the thesis/principles) and
`experiments/README.md` §0 (the central insight). This file is the concrete
experimental plan, with the design decisions and their rationale logged so
another LLM can pick it up cold.*

---

## 0. The one-paragraph frame

`μ = E_{x~N(0,I)}[z_L]` is a **deterministic function of the weights**, `μ =
F(W)` (README §0). The unscented transform (UT) is one cheap approximation whose
error floors at its sampling variance; its residual `R(W) = F(W) − UT(W)` is also
deterministic in `W` and therefore learnable. We build **one** recurrent corrector
that adds a learned per-neuron correction to the UT's final-layer mean, and we
run **four ablations of that single architecture** to measure exactly what each
information source is worth. The eventual research payload — a large sparse
(Monarch) RNN — is deferred to v2 (§7); v1 proves the signal cheaply.

---

## 1. The four tracks (all one architecture, retrained per track)

| Track | Name | Cloud port (C2) | Weight port (C1 = `P(W_k)`) | What it measures |
|---|---|---|---|---|
| 1 | **UT base** | — | — | the champion reference (adj. 3.39e-7); its final mean anchors all others |
| 2 | **sampling→RNN** | on | off | value of the *intermediate UT clouds* alone (no weights) |
| 3 | **weights→RNN** | off | on | value of *`W` alone* on top of UT (deliberately weak — see §3) |
| 4 | **UT + `P(W_k)`** | on | on | the thesis: sampling supplies moments, `W` supplies exact alignment |

**Anchoring (do-no-harm by construction).** Track 1 runs the UT with a *fixed,
recorded* rotation set; its post-layer-32 mean is handed to Tracks 2–4, which all
output `UT_final_mean + correction`. A **zero correction reproduces Track 1**, so
no track can do worse than UT except by overfitting — which we guard against with
held-out seeds and a zero-initialized readout (§5).

**Ablation = retrain, not eval-mask.** Each track is a *separate fit of the same
architecture*, with the off ports masked to zero **during training** (not just at
eval), so every track is in-distribution for the ports it has. The Track-4-minus-
Track-2 and Track-4-minus-Track-3 gaps then honestly measure the interaction.

---

## 2. The architecture — one RNN, three channels

The corrector sweeps **depth** `k = 1..L`. At each layer it consumes three
channels and updates a hidden state; a readout at the end produces the per-neuron
final-layer correction.

```
Channel 1:  P(W_k)     — weight features (see §3)                    [ablatable]
Channel 2:  cloud_k    — the intermediate UT sigma-point cloud at layer k [ablatable]
Channel 3:  h_k        — hidden state, h_0 = 0                        [always on]

recurrence:  h_k = g( pool(P(W_k)), pool(cloud_k features), h_{k-1} )
readout:     correction_j = R( per-neuron final-layer features_j , h_L )
estimate:    μ̂_{L,j} = UT_final_mean_j + correction_j
```

### 2.1 Channel 3 is **global** (a single shared vector) in v1

The problem is invariant to **independent per-layer neuron relabeling**. That
symmetry admits exactly two ways to carry a hidden state across layers: *global /
pooled* (label-free) or *`W`-transported*. A positional per-neuron state
(`h_{k+1}[j] ← f(h_k[j],…)`) **breaks the symmetry** — it links indices that are
independently relabelable across layers — and would overfit each MLP's arbitrary
labeling. So v1 uses the **global** carrier: `h_k` is a single shared vector that
tracks the *depth-evolution of the distribution's aggregate non-Gaussianity*
(well-matched to §8.1: the leading, shared behavior is global; per-neuron
fluctuations are the deferred `W`-carrier's job).

`g` (the global recurrence map) and `R` (the readout) are **zero-initialized on
their output** so the correction starts at exactly 0 → Track 1.

### 2.2 Where the per-neuron signal enters

Because Channel 3 is global, the **per-neuron specificity comes from the readout's
per-neuron features**, not from the carrier. Those features are the last layer's
alignment and cloud statistics (§3). This is sound because §8.3 measured that,
*given accurate final-layer moments, the per-neuron mean is essentially a fixed
function of the alignment* (R²≈1.0) — so the per-neuron answer lives in the
final-layer features, and `h_L` supplies the depth-context correction for
accumulated closure drift.

---

## 3. `P(W_k)` — equivariance lives here (the "convolution" of this problem)

Design principle (the CNN analogy): **`P` carries the relabeling-equivariance so
the RNN can stay generic.** The equivariant primitive is the **inner product**,
which is invariant to relabeling the input (layer-`k`) neurons and equivariant in
the output neuron `j`. `P` has two pieces:

- **Pure-`W` invariants** (weak, but survive the cloud being off — this is all
  Track 3 has): per-column concentration stats, e.g. participation ratio
  `‖W_k[:,j]‖₂² / ‖W_k[:,j]‖₄²`. Expected weak (Gaussian columns concentrate;
  §5.3 of the ideas doc).
- **`W`×cloud alignment** (strong, symmetry-clean — the thesis signal): computed
  from `W_k` and the cloud's per-layer moments `μ_k, Σ_k`:
  ```
  alignment_j = ⟨ W_k[:,j], μ_k ⟩              # corr +0.86 with truth (§8.3)
  var_j       = W_k[:,j]ᵀ Σ_k W_k[:,j]          # the pre-activation std s_j²
  rect_j      = GaussianReLU(alignment_j, √var_j)   # exact rectification of the above
  ```

**These interaction features must be *precomputed*.** A generic RNN operating on
summary statistics cannot reconstruct a 256-dimensional dot product from a cloud's
mean and variance — so we hand it the alignment, exactly as one hands a CNN the
convolution rather than hoping an MLP rediscovers it. This is *not* a violation of
the bitter lesson: it is supplying the symmetry, not a heuristic.

**Consequence for the ablation ladder.** The strong feature is intrinsically a
`W`×cloud interaction, so it is present only when **both** ports are on (Track 4).
Track 3 (cloud off) keeps only the weak pure-`W` invariants — intentionally, so
its number reads as "what does `W` add with no propagated moments to align
against." The headline result is expected to be **Track 4 ≫ Track 2, Track 3** —
the interaction, which is the thesis.

---

## 4. Evaluation harness (shared by all four tracks)

- **Dataset:** the official public phase-1 set,
  `hf://aicrowd/arc-whestbench-public-2026@v1-phase1` — 100 real-generator MLPs
  shipping *with weights and ground-truth means*. No Monte-Carlo on our side; no
  `build_mlp` population mismatch (README §7.1).
- **Split:** 80 MLPs train / 20 held-out validation for the learned tracks; Track 1
  runs on all 100. (20 is thin given per-MLP MSE spans ~7×; bake more MLPs later if
  a ranking is close.)
- **Primary metric:** adjusted final-layer score (leaderboard metric), reported
  with raw final-layer MSE. Reference points: UT champion 3.39e-7; crude
  `cov_prop` 8.4e-6.
- **Diagnostics (for intuition, not scoring):** per-layer MSE and truth-vs-
  prediction correlation — the *shape* of the error matters for Tracks 2/3.

---

## 5. Training

- **Target:** `truth − UT_final_mean`, per neuron, final layer. (`truth` is the
  baked ground-truth mean; UT uses the frozen rotation set and the fixed radius.)
- **Loss:** MSE of `UT_final_mean + correction` vs `truth` on the final layer.
  *Optional* auxiliary term: supervise intermediate layers too (we have all-layer
  truth) to stabilize the depth recurrence — enable only if the final-layer-only
  fit is unstable.
- **Do-no-harm:** zero-initialize `R`'s output layer (correction ≡ 0 at init =
  Track 1), weight decay pulling the correction toward 0, validate only on
  held-out MLPs.
- **Frozen everything on the UT side:** the rotation set and count, and the radius
  `= E‖x‖ = √2·Γ((w+1)/2)/Γ(w/2)` (the champion fix), are identical in training and
  eval. The learned correction is defined *relative to this frozen sigma scheme*
  (RCP_ideas §5.2); changing it invalidates the fit.
- **Framework:** PyTorch.

---

## 6. Code structure (v1) — with the Monarch seam marked

```
experiments/
  corrector/
    features.py     # UT sweep that records, per layer: cloud moments (per-neuron
                    #   + pooled), and the P(W_k) features (pure-W invariants +
                    #   W×cloud alignment). Frozen rotations, radius = E||x||.
    dataset.py      # load HF phase-1 MLPs -> (features per track, target). 80/20 split.
    model.py        # the one architecture:
                    #   class GlobalRecurrence(nn.Module):  # <-- Channel 3 update g
                    #       # v1: a small GRU/Linear on the pooled channels.
                    #       # >>> MONARCH SEAM <<< when the global state goes large
                    #       # (v2), swap this Linear for MonarchLinear; keep the
                    #       # interface (in_dim, hidden_dim) identical.
                    #   class Readout(nn.Module):           # R: per-neuron, zero-init
                    #   class Corrector(nn.Module):         # sweep + readout + add UT
    train.py        # retrain-per-track loop (mask off-ports during training),
                    #   held-out eval, logs adj. score + raw MSE + per-layer/corr.
```

The **Monarch seam** is the single `Linear` inside `GlobalRecurrence` (and, at v2,
inside the per-neuron carrier). `MonarchLinear` lives at
`~/projects/2_research/iterativennsimple/iterativennsimple/MonarchLinear.py` and
is a drop-in `nn.Module` with a `Linear`-like interface; v1 keeps the plain
`Linear` so the seam is a one-line swap later.

---

## 7. Sequencing / milestones

- **v1 (this plan) — small, Monarch-free, proves the signal.** Build the four
  tracks, retrain each, report the ablation ladder. **Decisive question:** does
  Track 4 beat Track 1 (UT) on held-out MLPs, and does the `W`×cloud interaction
  (Track 4) beat cloud-only (Track 2)? If no, the corrector idea has no legs and
  we stop before building machinery. If yes, proceed.
- **v2 — the per-neuron `W`-transported carrier (Monarch's home).** Add the second
  Channel-3 sub-state that is transported across layers by `W_k` (the faithful
  learned moment propagation, symmetry-correct because `W` transforms correctly).
  This is where the state goes large and **Monarch becomes load-bearing** — the
  research payload. `W=0` ablation zeros the transport automatically.
- **v3 — leaderboard hardening** if warranted: verify via `whest run` on the HF
  set, check FLOP utilization < 10%, submit.

---

## 8. Decisions log (resolved forks + rationale)

1. **One shared harness** (HF phase-1, adj. final-layer score, 80/20). — Real
   generator, weights+truth free, methodology-compliant (README §7.1).
2. **UT-anchored corrector; 0-correction = Track 1.** — Do-no-harm by
   construction; every track stands on the champion.
3. **Sampling input = intermediate UT clouds; corrector is recurrent over depth.**
   — This is why an RNN: it consumes the depth sequence.
4. **Three channels; ablate C1/C2 by zeroing.** — Clean, comparable-by-construction
   ablations (RCP).
5. **Channel 3 global (v1); positional per-neuron forbidden.** — Symmetry admits
   only global or `W`-transported carriers; positional breaks per-layer relabeling
   and overfits labelings.
6. **Equivariance lives in `P`; RNN stays generic** (CNN analogy). — `P` built from
   inner products (relabeling-invariant); the alignment is the "convolution."
7. **`P` = pure-`W` invariants ⊕ `W`×cloud alignment, precomputed.** — The strong,
   symmetry-clean signal (§8.3, corr 0.86) is an interaction and cannot be
   reconstructed by a generic RNN from summaries.
8. **Retrain per track** (mask ports during training). — The only ablation whose
   gaps mean what we claim.
9. **Sequencing (a): small Monarch-free v1 first; Monarch at v2.** — Monarch earns
   its keep only for the large per-neuron carrier; prove signal first. Code leaves
   a marked seam for the swap.

---

## 9. What would kill it (honest)

- **Track 4 ≈ Track 1.** If the interaction doesn't beat UT on held-out MLPs, the
  residual `R(W)` is not cheaply representable from these features and the whole
  corrector line is a dead end (consistent with the possibility, from
  RCP_ideas §9.4, that UT + radius fix already extracts nearly all the signal).
- **Track 4 ≈ Track 2.** If adding `W` buys nothing over the clouds alone, then the
  clouds already contain the alignment well enough and `P` is redundant — a clean,
  publishable negative result about white-box value on top of a good sampler.
- **Overfit to 20 held-out MLPs.** Guard with zero-init/weight-decay and, if any
  ranking is close, bake more MLPs before believing it.
