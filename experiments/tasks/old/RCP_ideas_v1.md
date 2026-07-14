# RCP Ideas v1 — A Learned Recurrent Corrector for the Unscented Transform

*Working notes, 2026-07-11. Author: RCP, with Claude. This is a research
direction, not a spec — it records the thesis, the math that makes it
principled, the open decisions, and the honest risks.*

---

## 0. One-paragraph thesis

The Unscented Transform (UT) has extracted essentially all of the *closed-form*
analysis available in the whest problem. What it leaves on the table is the
**higher-moment (Edgeworth) correction** that a degree-3 moment-matching rule
structurally cannot represent. That residual has no clean analytic form — which
is why the white-box / moment-closure attempts stalled — but it is a smooth,
structured function of the propagated state. So we learn it, with a
**recurrent corrector** whose hidden state carries exactly the higher-moment
information UT discards. We favor a **large-scale sparse RNN** (block-diagonal +
random-permutation factorization, à la `MonarchLinear`) as the parameterization,
both because it is the research direction of interest and because it keeps a
high-capacity map cheap enough to be free against the FLOP budget.

---

## 1. Problem, in our notation

Fixed, given MLP; random input. For a single network:

```
z_0 = x ~ N(0, I_width)
z_{l+1} = ReLU(W_l^T z_l),   l = 0 .. depth-1
```

Phase 1: `width = 256`, `depth = 32`, He init `W_l ~ N(0, 2/width)` i.i.d.
entries. Target: per-neuron mean activation `mu_l = E_x[z_l]`, shape
`(depth, width)`. Scored metric is `adjusted_final_layer_score`:

```
score = mean_m [ final_layer_mse_m * max(0.1, C_m / B) ]
```

i.e. **only the last layer's MSE is scored**, and compute below 10% of budget
gets no multiplier reward. Two consequences we should exploit:

- The corrector only has to be right at **layer `depth`**. Intermediate layers
  matter only insofar as they feed the last one.
- The UT already sits near the 10% floor, so **the corrector is free**: any MSE
  it removes is pure score reduction, with no multiplier cost.

---

## 2. Why this is principled (the Edgeworth picture)

Let `a_j = w_j^T z_l` be the pre-activation of neuron `j` at layer `l`
(`w_j` = the j-th column/row of `W_l`).

- **Layer 1: the true mean is exact, but the UT estimate is not.** `z_0 = x ~
  N(0, I)` ⇒ `a_j = w_j^T x ~ N(0, ||w_j||^2)` is *exactly* Gaussian, so the
  *true* mean `E[ReLU(a_j)] = ||w_j|| / sqrt(2*pi)` has a closed form. **But the
  UT *estimate* is NOT exact here** — see §9: because its sigma points sit on the
  shell `||x|| = sqrt(n)` while the Gaussian's mean radius is `E||x|| ~
  sqrt(n - 1/2)`, and ReLU is degree-1 homogeneous, UT overestimates by the
  radial factor `sqrt(n)/E||x|| ~ 1 + 1/(4n)`. This is a scalar quadrature
  artifact, not an Edgeworth effect, and it is present (and compounding) from
  layer 0 on. It is removable for free by rescaling the shell radius (§9).
- **Depth is where the signal is.** For `l >= 2`, `z_l` is non-Gaussian
  (rectified, coordinate-dependent). `a_j` is a sum of `width` weakly-dependent
  terms ⇒ *approximately* Gaussian by a CLT argument — which is exactly why UT
  works and why the residual looks small and "isotropic." But it is not zero.
  The deviation is the **Edgeworth correction**: skew, kurtosis, and
  cross-coordinate dependence of `z_l`, propagated through `w_j` and the ReLU.
- **The correction accumulates.** UT re-Gaussianizes at every layer — it keeps
  only the mean and covariance implied by `2n` sigma points and discards the
  rest. The discarded higher moments compound with depth, so the residual grows
  toward the deep layers — precisely where the score lives.

**The RNN is a learned moment-closure.** The closed-form closure has no clean
form past low order (this is what stalled `old/whest_moment_closure.py`). A
recurrent map with a hidden state sidesteps the wall. The hidden state `h_k` has
a clean reading: *the higher-moment sufficient statistic of `z_l` that UT throws
away.* Weight-sharing `g` across layers is the right inductive bias — every
layer has the same statistical structure.

---

## 3. The architecture (rough)

Base estimate stays the randomized UT. The corrector adds a learned residual to
the **final-layer** mean:

```
h_{k+1} = g( phi(sigma-cloud_k) [, P(W_k)] , h_k )        # recurrent state update
mu_hat_L = f(sigma-cloud_L)  +  f_h(h_L)                  # UT mean + learned correction
```

- `f(sigma-cloud_L)` = the current UT estimate (per-layer empirical mean of the
  propagated sigma points).
- `phi(sigma-cloud_k)` = **cheap per-neuron moment features** read directly off
  the propagated cloud: pre- and post-activation std, skewness, kurtosis, a few
  quantiles. These are free (`O(points * width)`) and already contain the image
  of `W_k` acting on the distribution.
- `g`, `f_h` = the learned maps, parameterized with `MonarchLinear`
  (block-diagonal dense blocks + random permutations) so a wide hidden state
  stays cheap.
- `P(W_k)` = **optional** featurization of the raw weights (see §5 — this is the
  open question we discuss next).

Training is fully offline and amortized: sample many MLPs via `build_mlp(seed)`,
compute Monte-Carlo ground truth, regress the corrector on the UT residual.
**Unlimited free training data from the exact test distribution** is the single
biggest asset here.

---

## 4. Why it can beat the white-box correctors that failed

The prior attempts appear to have died on **isotropy** — they sought a
*global/directional* correction (a fixed shift or preferred axis), which
averages to zero when the residual has no preferred direction. The RNN does
something categorically different: a **per-neuron, conditional** correction —
predicting neuron `j`'s residual from *neuron `j`'s own moment trajectory*.
Isotropy across neurons says nothing about whether that conditional map exists;
the Edgeworth picture says it should. **This is the bet.** It remains a bet, so
the first move is to *measure* that conditional signal before building a large
model on top of it.

---

## 5. Open decision: does the corrector need `P(W_k)`?

Two readings, and they are separable:

- **Weights implicit (moments-only).** The propagated sigma cloud already *is*
  `W_k` acting on the distribution. A small RNN over `phi(sigma-cloud)` may
  capture most of the correction. Cheap, low parameter count. *Claude's prior:
  this gets most of the win.*
- **Weights explicit (`P(W_k)`).** Feed a Monarch-compressed representation of
  `W_k` alongside the cloud. Higher capacity, the natural home for a genuinely
  large sparse RNN, but more cost/params for possibly marginal accuracy gain.

**Honest framing:** "large-scale sparse RNN over the weights" is partly a
*research-direction* choice, not a pure performance necessity. Both goals —
win the challenge, and explore Monarch-style efficient RNNs — are legitimate and
worth pursuing; we should just stay clear-eyed about which knob buys accuracy and
which buys research interest. Clean way to honor both: parameterize `g`, `f_h`
with `MonarchLinear` (the research payload) but start the *input* at the cheap
moment cloud, adding `P(W_k)` only if the residual measurement demands it.

> **Next conversation:** nail down what `P(W_k)` actually is — what information
> about `W_k`, at what cost, in what invariant form (it should respect the
> permutation/relabeling symmetry of neurons within a layer).

### 5.1 Consensus principle — ReLU, singular vectors, and what `P(W_k)` is for

Established by RMT + the non-rotational-invariance of ReLU (agreed 2026-07-11):

- **Singular values are stable across `k`.** Each `W_k` is a square Ginibre
  matrix (`i.i.d. N(0, 2/n)`, `n = 256`); its singular-value spectrum converges
  to a fixed quarter-circle law on `[0, 2*sqrt(2)]` (operator norm → `2*sqrt(2)
  ≈ 2.83`), the same for every `k` up to `O(n^{-1/2})` bulk fluctuations.
- **Singular vectors are Haar-random and independent across `k`.** `U_k`, `V_k`
  are independent Haar orthogonal bases, independent of the singular values —
  essentially a random orthonormal basis of `R^256` per layer, with no
  persistent frame across layers.
- **ReLU is not rotationally symmetric** — equivariant only to coordinate
  permutations and positive scalings. So the per-neuron mean is *not* a function
  of the singular values alone; the **orientation** of `W_k` matters. This is
  the real content.

Three refinements that make this precise:

- **(a) It is a three-way interaction; the state is the third party.** The
  singular vectors of `W_k` do not meet ReLU in a vacuum — they meet it *through
  the geometry of the current state*. `a_j = w_j^T z_l`, and `E[ReLU(a_j)]`
  depends on how `W_k`'s orientation aligns with the covariance and higher
  moments of `z_l` **in the standard basis**. The essential object is
  (ReLU axes) ↔ (`W_k` orientation) ↔ (the coordinate-aligned anisotropy ReLU
  has imprinted on `z_l`). `W_k`'s singular vectors alone are not the signal;
  their *coupling to the state* is.
- **(b) The interaction turns on at depth ≥ 2.** At layer 1 the input is
  isotropic Gaussian (rotationally symmetric), so `a_j ~ N(0, ||w_j||^2)`
  regardless of the singular vectors — they are literally irrelevant, only the
  singular-value quantity `||w_j||` matters. The vector/ReLU interaction is
  *created* by the first ReLU making `z_1` anisotropic and non-Gaussian, and it
  compounds with depth. This is exactly the Edgeworth picture of §2. (Caveat:
  "singular vectors irrelevant at layer 1" is a statement about the *true* mean;
  the UT *estimate* still has a scalar radial bias at layer 1 — see §2 and §9 —
  but that bias is orientation-free, so it does not involve the singular vectors.)
- **(c) The second-moment slice is already free.** The UT cloud propagated
  through `W_k` already computes the coupling of `W_k`'s orientation to the
  state's *first two moments* — that is precisely `ReLU(W_k^T · cloud)`. So the
  singular-vector/ReLU interaction, to second order, is *not* missing; UT has it.
  What UT cannot see is the coupling to the *higher* moments of `z_l`.

**Consensus statement.** The non-rotational-invariance of ReLU makes the
*orientation* of each `W_k` essential (not just its spectrum), but this only
bites at depth ≥ 2, it is fundamentally a coupling between `W_k`'s orientation
and the state's anisotropy, and its second-moment part is already captured by
the UT cloud — so **the specific job of an explicit `P(W_k)` is to encode how
`W_k`'s orientation couples to the higher moments of `z_l` that the cloud cannot
represent.**

---

### 5.2 Second principle — the residual is safe by construction; `P(W_k)` need only be *useful*

Agreed 2026-07-11.

- **The correction is an additive residual, and `f_h ≡ 0` is already the
  champion.** `mu_hat_L = f(cloud_L) + f_h(h_L)` reduces to the current UT
  estimator when `f_h` vanishes. So the corrector can only *add* signal on top of
  a working baseline — the same reason boosting and residual nets are safe. We
  are never risking the base solution, only trying to improve it.
- **"Useful, not perfect."** The exact target (the full Edgeworth closure) has
  no clean closed form (§2), so a *perfect* `P(W_k)` is unreachable and chasing
  it is a known failure mode (this is where the previous effort derailed). The
  only bar that matters is: **does adding `P(W_k)` lower held-out MSE versus the
  ablation without it?** Infinite training data makes that bar cheap to test, so
  we let the data — not analysis — decide whether a given `P(W_k)` earns its
  place. Start plain; add fanciness only when an ablation pays for it.
- **"Do no harm" must be made structural.** `f_h ≡ 0` is safe, but a *fitted*
  `f_h` can overfit and raise held-out error. So bias the default toward doing
  nothing: initialize `f_h` near the zero function (small final-layer weights),
  optionally gate it with a learned scalar, add weight decay pulling it toward
  zero, and **validate only on held-out seeds** (never training seeds).
- **Frozen rotations are essential.** The propagated sigma cloud is a
  deterministic transform of the sigma-point set; its higher-order content is an
  artifact of that specific (rotation-averaged) construction, not a clean
  estimate of the true state distribution. `g`/`f_h` therefore learn a
  correction *relative to a fixed sigma scheme*. The sigma-point construction —
  radius, rotation scheme and its RNG, and the count policy given the budget —
  **must be identical in training and at inference**, and is effectively part of
  the model. Changing the rotation scheme invalidates the learned correction and
  every usefulness ablation must hold it fixed.

### 5.3 First candidate `P(W_k)` — weight concentration, and why to expect it weak

Considered 2026-07-11. A natural *simple* `P(W_k)`: a per-neuron measure of how
concentrated neuron `j`'s incoming weights are. The intuition is sound — a
neuron whose weights spread across many coordinates has a pre-activation
`a_j = w_{·j}^T z_l` that is a sum of many weakly-dependent terms (≈ Gaussian by
CLT, small residual); a neuron whose weights concentrate on a few coordinates
inherits their post-ReLU non-Gaussianity (large residual). So concentration
predicts *where* the correction is needed.

Two corrections to the first draft of this idea:

- **Measure it on `W_k`'s columns, not on `U_k` / `V_k^T`.** ReLU acts in the
  standard basis, so what matters is the weight vector *in that basis* (column
  `w_{·j}` of `W_k`). The largest entry of a row of `V_k^T` lives in abstract
  singular-coordinate space — Haar noise with no persistent frame (§5.1) — and
  because the spectrum is flat (no spectral gap), a large `V_k^T` entry does not
  imply a spiky `w_{·j}`. Use a direct concentration statistic instead, e.g. the
  participation ratio `||w_{·j}||_2^2 / ||w_{·j}||_4^2` or `||w_{·j}||_∞ /
  ||w_{·j}||_2`. These are per-neuron, permutation-equivariant, and need **no
  SVD** (the SVD costs `O(n^3)` ≈ ¼ of a UT rotation over 32 layers and only
  buys a noisier version of the same signal).

- **Expect it weak, for two reasons.** (i) With `i.i.d. N(0, 2/n)` entries,
  every column's concentration statistic concentrates over its 256 entries, so
  it is nearly *constant across neurons* (`O(1/sqrt(n))` per-neuron variation) —
  recall even `||w_{·j}||`, the only thing that matters at the exact layer 1,
  barely varies. (ii) More fundamentally, by the §5.1 consensus a **state-free
  `P(W_k)` cannot capture the signal**: the residual is the *coupling* of `W_k`'s
  orientation to the higher moments of `z_l`, a function of both operands. A
  `P(W_k)` that never looks at the cloud is at best a weak marginal prior on
  "which neurons tend to be hard," not the coupling itself.

**Takeaway.** Concentration-of-`W_k`-columns is a fine *first probe* under
"useful, not perfect" (§5.2) — cheap, easy to ablate — but predicted to move
held-out MSE only marginally. The higher-value first `P(W_k)` is a `W_k` × cloud
*interaction* feature (looks at both operands, per §5.1), e.g. per output neuron
`j`, the cloud-measured variance of `a_j` versus the Gaussian prediction.

## 6. FLOP budget reality

One UT rotation at (256, 32) ≈ `4 * depth * width^3 ≈ 2.1e9` FLOPs; we spend
~10% of budget on rotations. A Monarch RNN sweeping 32 layers with a width-256
state and a few blocks costs `~1e6–1e7` FLOPs total — **< 1% of a single
rotation.** The multiplier floors at 0.1 regardless. So the corrector is
effectively free, and (§1) its only effect on the score is downward. The one
real cost is generating MC ground truth for training (expensive, but
embarrassingly parallel on GPU).

---

## 7. Risks / things that could kill it

1. **No conditional signal.** If the per-neuron residual genuinely isn't
   predictable from cheap features, we're chasing irreducible variance. →
   Measure first (§4).
2. **Overfitting the seed distribution.** Mitigated: the test set *is* the same
   distribution (He, 256×32). Still, hold out seeds and watch generalization.
3. **MC-label noise floor.** The ground truth is itself a finite-sample MC
   estimate; the corrector can't beat that noise. Budget enough MC samples that
   label noise << current UT MSE.
4. **`P(W_k)` cost blow-up.** Feeding raw weights densely could cost more than
   the UT itself. Monarch mitigates; moments-only sidesteps entirely.
5. **Depth generalization.** If we ever train at one depth and test at another,
   the shared-`g` recurrence should help — but Phase 1 is fixed at 32, so not an
   immediate concern.

---

## 8. Immediate next steps (proposed)

- [x] **Characterize the residual — is the UT error bias or variance?** Done
      2026-07-11 (`experiments/whest_bias_variance.py`). Result in §9: the bias
      is real and substantial (~43% of operating-point MSE), so a corrector is
      justified; but a large chunk of it is a *radial quadrature artifact*, not
      Edgeworth.
- [ ] **Rescale the shell radius and re-measure the bias floor.**
      (`experiments/whest_radius.py`) Quantify how much of the bias is the free
      radial fix vs the true Edgeworth residual; then tune the scalar radius on
      held-out seeds. Whatever survives is the RNN's actual target.
- [ ] **Define `P(W_k)`** (later discussion) — invariant, cheap, and only if the
      residual that survives the radius fix still shows learnable structure.
- [ ] **Stand up the training scaffold.** Monarch-parameterized `g`/`f_h`,
      offline regression on the *post-radius-fix* UT residual, held-out seeds.

---

## 9. Experimental findings

### 9.1 UT error is ~43% bias, and rotations are not a free lever (2026-07-11)

`experiments/whest_bias_variance.py`, 6 seeds, 2e6 MC samples (MC noise `v ~
1.5e-8`, ~100x below the signal — trustworthy). Decompose `UT_R = (UT_R -
UT_inf) [variance] + (UT_inf - truth) [bias]`. Final-layer, width 256, depth 32:

- **Bias floor (attackable) ≈ 1.9e-6.** MSE at the operating point (R=16) ≈
  4.4e-6 = **bias 1.9e-6 + variance 2.5e-6**, i.e. **~43% of the error is
  attackable bias.** (An earlier 2-seed peek wrongly suggested "variance-
  dominated, 15-20%" — it hit two low-bias seeds; seed-to-seed bias varies ~4x.)
- **Rotations are a wash, not a competing lever.** Buying down the 2.5e-6
  variance needs more rotations, which raises the FLOP multiplier ~linearly
  (R=16→32 nearly halves MSE but doubles the multiplier). So reducing MSE *at
  fixed compute* — attacking the bias — is essentially the only free lever. The
  corrector program is justified.
- **The bias is NOT purely deep-layer Edgeworth.** Layer 0 already carries ~half
  the deep-layer bias (~1e-6 vs ~2e-6), rising mildly and non-monotonically with
  depth. Layer-0 bias cannot be Edgeworth (the pre-activation is exactly
  Gaussian there).

### 9.2 The layer-0 bias is a radial quadrature artifact (provable)

ReLU is positively homogeneous of degree 1: `ReLU(w^T x) = ||x|| ReLU(w^T
x_hat)`. The randomized (Haar-averaged) UT places every sigma point on the shell
`||x|| = sqrt(n)`, so as rotations → ∞ the directional integral is exact but the
radius is fixed at `sqrt(n)`:

```
UT_inf = sqrt(n) * E_dir[ReLU(w^T x_hat)]     truth = E||x|| * E_dir[ReLU(w^T x_hat)]
```

So at layer 0 the entire bias is the radial factor `sqrt(n)/E||x|| = 16/15.985 ≈
1.00096` — a systematic **~0.1% overestimate**, present at every layer and
compounding with depth. Root cause: the base UT sets the radius to match
`E[rho^2] = n` (the covariance), but the mean of a degree-1-homogeneous map is
controlled by the *first* radial moment `E[rho] = E||x||`.

**Free fix (untried in our champion):** keep ONE shell, move it to radius `alpha
* sqrt(n)` with `alpha = E||x||/sqrt(n)`. This zeroes the layer-0 bias exactly,
with no extra points and no stolen rotations. `alpha` is a single scalar we can
tune on infinite data to minimize the *final-layer* bias (the input rescaling
propagates nonlinearly, so the depth-32 optimum may drift from the layer-0 one).

**Distinct from `old/whest_two_radius.py`.** The previous effort found the radial
bias but "fixed" it with a 2-point Gauss quadrature on the `chi^2_n` radial
measure — *two* shells = 4n points/rotation, forcing rotations to be halved at
fixed budget. Given §9.1 (variance ≈ bias), halving rotations roughly doubles
the angular variance and eats the gain. That is the "perfect not useful" trap
(§5.2): the single-scalar radius rescale is the useful version — same point
budget, strictly fewer sources of error.

### 9.3 Radius rescale result — the free fix removes ~87%+ of the bias (2026-07-11)

`experiments/whest_radius.py`, 6 seeds, 2e6 MC, `r_inf=256`. Radius = `alpha *
sqrt(n)`; final-layer bias floor vs `alpha`:

```
   alpha   radius       bias^2
    0.95   15.200    3.588e-03
    0.97   15.520    1.255e-03
    0.99   15.840    1.202e-04
 0.99902   15.984    2.410e-07   <- alpha_opt = E||x||/sqrt(n)  (theory)
     1.0   16.000    1.890e-06   <- base UT (sqrt n)
    1.01   16.160    1.831e-04
    1.02   16.320    6.639e-04
```

- **The theory value nails it.** `alpha = E||x||/sqrt(n) = 0.99902` (the layer-0
  optimum from §9.2) is *also* the final-layer optimum — the nonlinear
  depth-propagation does **not** drift it. No tuning needed: the radius is
  `E||x|| = sqrt(2)Γ((n+1)/2)/Γ(n/2)`, full stop. The curve is a razor-sharp
  minimum (α off by 0.01 in either direction is 50-500x worse), a clean
  confirmation of the homogeneity analysis.
- **87% of the bias removed for free:** final-layer bias floor `1.89e-6 →
  2.41e-7`, a 7.8x reduction, at identical point budget. This is a bankable
  improvement to the champion — change one constant (`radius = sqrt(width)` →
  `E||x||`) in `estimator.py`. At the operating point it should cut MSE from
  ~4.4e-6 toward ~2.7e-6 (≈38%), since the optimal α for bias is R-independent.

- **The surviving 2.4e-7 is mostly variance, not Edgeworth — the RNN's target is
  small.** Two tells: (i) at `r_inf=256` there is still ~`var@R16 * 16/256 ≈
  1.8e-7` of residual angular variance, comparable to the whole 2.4e-7 floor;
  (ii) the per-layer floor at optimal α **decreases** with depth (layer 0
  ~6.8e-7 → layer 31 ~2.4e-7), the *opposite* of the accumulating non-Gaussianity
  an Edgeworth residual would produce — so it reads as finite-rotation variance
  (which falls with depth as ReLU contracts), not higher-moment bias. **The true
  post-fix Edgeworth bias is therefore ≤ 2.4e-7 and plausibly much smaller; it is
  buried below the r_inf=256 variance floor and cannot be quantified from this
  run.**

### 9.4 Implications for the program (revised thesis)

- **Bank the radius fix now** — it is a large, free, provable win independent of
  everything else. Verify via the real eval path (whestbench baked sets + AIcrowd
  API) and submit.
- **The RNN corrector thesis is substantially deflated by this.** The "~43%
  attackable bias" of §9.1 was almost entirely the trivial radius artifact, not
  the deep higher-moment coupling the whole `P(W_k)`/RNN design was built to
  capture. After the free fix, the Edgeworth residual — the RNN's actual target —
  looks small (≤ 2.4e-7, likely less) and may sit below the MC/rotation noise we
  can currently resolve.
- **Before investing in the RNN, measure what actually survives.** Re-run the
  bias floor at the *fixed* radius with a much larger `r_inf` (e.g. 1024-4096) so
  the angular variance drops well below the residual bias. Only if a clear,
  depth-*increasing* residual emerges above the noise floor is there an Edgeworth
  target worth a learned corrector. If not, the honest conclusion is that UT +
  radius fix has extracted essentially all the available signal, and the RNN
  direction should be reconsidered (or pursued as a research exercise with eyes
  open about a small payoff).
