# Sampling-Based Estimation for the ARC White-Box Challenge

Research log and submission guide for our work on the
[ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026).
This folder holds our experiments; the repo root is a fork of the official
[starter kit](https://github.com/AIcrowd/whest-starterkit) (its `README.md` is
preserved).

---

## 1. The competition in one page

**The research question (from ARC):** *Can you predict a neural network's
behavior by analyzing its weights, instead of running it on many inputs?*

**The task.** You are given one randomly-initialized ReLU MLP — square,
**width 256, depth 32** in Phase 1, He-initialized weights `W ~ N(0, 2/width)`,
input `x ~ N(0, I)`. Each layer computes `y = ReLU(Wᵀx)`. You must predict the
**per-neuron mean post-ReLU activation at every layer** — a `(depth, width)`
array. Ground truth is a large Monte-Carlo average.

**Scoring** (lower is better), leaderboard metric `adjusted_final_layer_score`:

```
                              1   M
adjusted_final_layer_score = ─── ∑  final_layer_mse_m × max(0.1, C_m / B)
                              M  m=1
```

with `C_m` = effective compute (analytic FLOPs + a small residual-wall-time
penalty `λ·R_m`, `λ ≈ 1e11` FLOP/s) and `B` = FLOP budget (`≈ 2.72e11`). **Only
the final layer is scored**; all-layers MSE is a diagnostic. Exceeding budget
zeroes that MLP *and* forces the multiplier to 1.0 (strictly worse than any valid
run).

### The two facts that shaped everything

1. **Below 10% of budget, compute is effectively free.** The multiplier floors
   at `0.1` for any usage `≤ 10%·B` — about **~6,500 forward passes** at
   width 256. Below the floor only accuracy matters, so for a sampling method the
   whole problem collapses to: *estimate `E_{x~N(0,I)}[activations]` as accurately
   as possible with a ~6,500-point budget.* That is a **quadrature/cubature
   problem.**
2. **Only the final layer counts, and depth 32 is where the analytic
   approximations break.** Correlations build with depth and ReLU couples the
   joint distribution, so the independence/Gaussian assumptions degrade exactly
   in the scored regime.

---

## 2. Toy cases — isolating the two analytic approximations

| File | Shape | Point |
|---|---|---|
| `whest_1x1.py` | scalar chain | Width 1 ⇒ **no correlations**, so mean- and covariance-propagation are the *same* algorithm. The only error is the **Gaussian-pre-activation assumption**, which compounds with depth (analytic systematically *over*-predicts: a Gaussian puts less mass at exactly 0 than the true rectified distribution). *Aside:* in 1D a single negative downstream weight kills the chain to exactly 0 forever — most seeds give a dead chain; seed 56 keeps it alive. |
| `whest_2x2.py` | 2 neurons | Smallest case with off-diagonal covariance. The two methods **agree at layer 0** (input cov = I) and **diverge from layer 1 on** — the challenge thesis in miniature. |
| `whest_8x8.py` | 8 neurons | **Covariance propagation is a coin flip vs mean propagation here** (won 6/10 seeds, sometimes 2–3× worse). The covariance advantage is a **large-width phenomenon**: the crude gain-rule ReLU covariance update `Φ(αᵢ)Φ(αⱼ)Σ[i,j]` adds error comparable to the correlation signal when width is small. |

Analytic baselines (raw final-layer MSE at 256×32): zeros `~0.91`,
`mean_propagation ~9.5e-4`, `covariance_propagation ~8.4e-5` (adjusted score
`~8.4e-6`). Our clean re-implementation reproduces `cov_prop` (~1.1e-4 over 4
MLPs), confirming correctness.

---

## 3. Method comparison — `whest_harness.py`

Compares sampling methods and analytic baselines at matched compute, by
final-layer MSE vs a large-N MC ground truth. **Key result at 256×32, matched
FLOPs (`k=512 ≈ cov_prop`'s `O(depth·width³)` cost):**

| method @ k=512 | final-layer MSE | vs `cov_prop` (1.08e-4) |
|---|---|---|
| `random_mc` | 1.89e-4 | worse |
| `control_variate` | 9.97e-5 | ~tie |
| `antithetic_mc` | 9.82e-5 | ~tie |
| `sobol_qmc` | 5.85e-5 | 1.8× better |
| `cubature_ut` | 4.90e-5 | 2.2× better |

At the free-budget operating point (~6,500 points) the best samplers reach
`~5e-6`, roughly **20× better than `cov_prop`** on the scored metric (both at the
same floored 0.1 multiplier). Ordering: **structured samplers (UT, QMC) beat MC**;
control-variate and antithetic give little; the analytic methods are best seen as
cheap baselines to beat. **Conclusion: the sampling approach wins at competition
scale.**

---

## 4. The mathematics of the samplers

### 4.1 Unscented transform (UT) — `whest_ut.py`, `../estimator.py`

Represent `N(0, I_n)` by the deterministic **sigma-point set `±√n·eᵢ`**
(`i = 1..n`). These `2n` points reproduce the input's mean (`0`) and covariance
(`I`) *exactly* — the **degree-3 spherical-radial cubature rule**. Apply a
**Haar-random rotation** and average several rotations: rotation preserves the
exact moment-matching while averaging out the fixed-axis bias (the "randomized"
UT). Push all points through the network, average per layer (equal weights ⇒ the
UT expectation is the plain mean). More rotations ⇒ more points ⇒ lower error.

Why UT over the analytic route: it makes **no Gaussian-closure and no gain-rule
approximation**, targets the mean directly, and gives every layer for free.

#### The even/odd (symmetry) structure — this drives the CV result below

The UT point set is **inherently ± symmetric** (every `ξ` comes with `−ξ`). For
any function,

```
½ ( f(ξ) + f(−ξ) ) = f_even(ξ),
```

so the UT estimate **only ever sees the even part of `f`; the odd part is
annihilated exactly.** Combined with the `√n` radial scaling (which reproduces
`E[xxᵀ]=I`), the rule integrates **all polynomials up to degree 3, and *every*
odd-degree term, exactly.** Its residual error is therefore **only the even
degrees ≥ 4** — the `|·|`-like part of ReLU that survives symmetrization
(`ReLU(t) = t/2 + |t|/2`; the even `|t|/2` is what remains).

**Consequence:** a *linear* control `Z = Ax` is odd, so the UT integrates it
*exactly* — `Z̄ = E[Z] = 0` identically over the sigma points, and the control-
variate correction `β(Z̄ − τ)` vanishes. **A linear control variate does nothing
for the UT.** More generally, *any* polynomial control of degree ≤ 3 is
redundant with the UT. To help the UT you would need a control capturing the
even, degree-≥4 rectification tail — which is exactly the hard, nonlinear content
cheap surrogates lack. (This is the open direction we deferred.)

### 4.2 Latin hypercube / stratified sampling (LHS) — `estimator_lhs.py`

Sample `N(0, I)` by a Latin hypercube: stratify each of the `width` coordinates
into `k` equal-probability bins, visit each bin once (a random permutation of
strata per coordinate), place the point at a random offset inside its bin, and
map to Gaussian by the inverse normal CDF. This removes the **per-coordinate
(main-effect) sampling variance** that plain MC carries, at one forward pass per
point (same as MC). Whether that stratification beats the UT's exact cancellation
is exactly the question the next subsection settles.

### 4.3 Which rule should win here — a Hermite-variance view

The proposal to try LHS/Sobol was, honestly, driven by (flawed) benchmark numbers
rather than theory (§7.1). The right way to compare the rules is a variance
decomposition, and it turns out to *favor the UT* — for the opposite reason to the
intuition that motivated the detour.

The integrand is `f(x)` = a final-layer activation as a function of the input
`x ~ N(0, I_d)` (`d = width`). Expand it in the **Hermite basis** — the
polynomials `H_α` orthonormal with respect to the Gaussian:

```
f = Σ_α c_α H_α ,        E[f] = c_0   (the quantity we want).
```

Every quadrature estimator of `c_0` has variance `Σ_{α≠0} c_α² · r(α)`, where
`r(α)` measures how badly the rule handles mode `α`. Plain MC handles nothing
specially: `r(α) = 1/k` for all `α`, so the variance is `Var(f)/k`. The
structured rules instead *zero out* whole subspaces of modes (`r(α) = 0` there):

- **UT** (± symmetric, degree-3 exact) zeros **all odd `|α|`, and all `|α| ≤ 3`.**
  Residual variance lives in **even total degree ≥ 4.**
- **LHS** (marginal stratification) zeros **all main effects** — modes `H_α`
  supported on a single coordinate, at *any* degree. Residual: all *interaction*
  modes.

The decisive comparison is the *unique* kills:

| rule | uniquely removes |
|---|---|
| UT | **low-degree interactions** — `H_{e_i+e_j}` (i≠j, degree 2), degree-3 cross terms, all odd interactions |
| LHS | **high-degree main effects** — `H_{4·e_i}, H_{5·e_i}, …` (single coordinate, degree ≥ 4) |

And here is the point, which ties directly to the challenge's own framing: **in a
deep ReLU network the nonlinearity shows up predominantly as low-degree
*interactions*.** The between-neuron covariance that accumulates with depth — the
very structure that makes the problem hard and that `cov_prop` tries to track — is
exactly degree-2 interaction content `H_{e_i+e_j}`. That sits **inside the UT's
kill set and inside LHS's blind spot.** The modes LHS uniquely handles (high-degree
content of a *single* input coordinate) are comparatively small. So the
decomposition predicts **UT beats LHS here** — as the leaderboard confirmed (§7.1).

This also corrects the intuition that motivated the LHS detour — "depth 32 is so
nonlinear that a degree-3 cancellation captures too little." The nonlinearity is
not high-degree, it is *interaction*-degree, which is precisely the UT's sweet
spot, not its weakness. And there is **no hard degree-3 floor** to escape:
fixed-axis cubature is biased by the degree-≥4 modes, but the *random rotation*
turns that bias into variance that averages down as `~1/rotations`, so the
randomized UT converges like any unbiased estimator. (The "plateau" seen at
width 32 was ground-truth noise, not saturation.)

**When LHS/Sobol would win instead:** when the variance lives in high-degree
*main effects* (strong univariate nonlinearity, weak coupling — LHS), or when `f`
is smooth with low effective dimension (Sobol). This problem is the opposite on
both counts: the coupling *is* the signal, and ReLU's kink makes `f` non-smooth,
which defeats Sobol's high-order convergence. Neither method's theoretical
strength is engaged.

**Corollary for improving the UT:** its *only* remaining error is the
even-degree-≥4 subspace (`H_{2·e_i}`-type curvature and quartic interactions). A
control variate aimed at those modes attacks exactly what the UT leaves on the
table — and, unlike a linear control (odd, degree 1, annihilated by the symmetry;
§5), it would not be redundant. That is the deferred "even-part control" of §9.

---

## 5. The hybrid investigation (route b) — a clean negative result

We tested "structured sampler + control variate," where the control is the
**mean-field-linearized network**: replace each ReLU by its expected derivative
`Φ(α_ℓ) = E[ReLU'] = P(active)` (from `cov_prop`), giving a linear map `Z = Ax`
with `E[Z] = 0` exactly. Control-variate estimator: `pred = Ȳ − β Z̄`,
`β = Cov(Y,Z)/Var(Z)`, variance reduced by `(1 − ρ²)`.

**It does not help, for two independent reasons — both measured.**

1. **Redundancy.** The control is linear (odd, degree 1). Any decent sampler
   already captures that: the UT annihilates it by symmetry (§4.1); LHS removes
   the same additive main-effect variance by stratification. So `ρ` between the
   control and the *sampler's residual error* is small — the CV corrects
   variance the sampler already removed.
2. **It costs a second forward pass.** Evaluating `Z` requires propagating every
   point through the linearized network too, so **under a fixed FLOP budget the
   CV variant gets half the points.**

**Matched-budget benchmark** (`bench2.py`, 256×32, ~6,144 forward-units, 4 MLPs ×
10 reps, raw final-layer MSE, mean ± std):

| method | MSE | note |
|---|---|---|
| **`lhs`** (pure) | **6.99e-6** ± 5.0e-6 | best, lowest variance |
| `ut` (12 rot) | 1.39e-5 ± 2.1e-5 | high variance at few rotations |
| `mc` | 1.44e-5 ± 1.2e-5 | |
| `lhs+cv` | 1.47e-5 ± 1.1e-5 | **worse than pure `lhs`** |
| `mc+cv` | 1.85e-5 ± 1.6e-5 | **worse than pure `mc`** |

So `sampler + CV` is dominated by `just spend the budget on the sampler`. Within
this benchmark LHS *looked* ~2× better than UT — **but that ranking did not hold
up on the leaderboard** (see §7.1). The `sampler + CV` negative result is solid;
the LHS-beats-UT claim was a small-sample / wrong-population artifact.

> Dependency note: the grader environment has **no `scipy`** (it was not present
> until we added it for local experiments), so submissions use only
> `flopscope`/`whestbench`. `flopscope` provides `flops.stats.norm.{cdf,pdf,ppf}`
> and a FLOP-counted `fnp.linalg.qr`, which is all the samplers need. There is no
> Sobol in `flopscope`; LHS is our pure-`fnp` QMC-family sampler.

---

## 6. Coding details of the submissions

Both submissions are single self-contained files (only `flopscope`, `whestbench`,
`math`). Both are **budget-aware**: they read `budget`, spend `TARGET_FRACTION`
of it, and size the point count from a closed-form cost.

**`../estimator.py` — UT.** `cost_per_rotation = 4·depth·width³`
(the `2·width` points of one rotation, `depth` matmuls of `~2·width²` each);
`rotations = int(TARGET·budget / cost_per_rotation)`. Sigma points are
`±√width · (columns of a random orthogonal Q)` via `fnp.linalg.qr`, concatenated,
propagated, averaged per layer.

**`estimator_lhs.py` — LHS.** `cost_per_point = 2·depth·width²` (one forward
pass); `k = int(TARGET·budget / cost_per_point)`. Points are generated
vectorized: `strata = argsort(argsort(uniform noise))` gives an independent
permutation `0..k-1` per column, `u = (strata + uniform)/k`, then
`flops.stats.norm.ppf(u)`.

**Tuning `TARGET_FRACTION`.** The multiplier is pinned at `0.1` for usage
`≤ 10%`, so we push toward the floor. UT uses `0.095` (≈9.7% effective). LHS uses
`0.09` (≈9.55% effective) — a larger margin because its point-generation
(`argsort`/`ppf`) adds residual wall-time (`λ·R`) on top of the raw FLOP count,
which pushes effective compute above the FLOP fraction. Verify the effective
`Mean Compute Utilization` stays `< 10%` after any change.

---

## 7. Submissions — verified end to end and submitted

Verification chain per estimator: `whest validate` (contract) → local
`BudgetContext` probe (FLOP fraction) → `whest run --runner subprocess` on a
locally-baked 256×32 dataset (real scoring: effective utilization, multiplier,
MSE, failures) → `whest validate-package`.

| # | Method | `TARGET` | points | local adj.* (3 MLP) | **leaderboard** | id |
|---|---|---|---|---|---|---|
| — | UT (pre-session) | 0.08 | 5,120 | 5.41e-7 | 4.29e-7 | (manual) |
| a | **UT** | 0.095 | 6,144 | 4.71e-7 | **3.99e-7 ← best** | 315205 |
| b | LHS | 0.09 | ~5,800 | 2.47e-7 (fluke) | 5.30e-7 (worse) | 315209 |

\* adjusted final-layer score on our **3-MLP** baked set (N=500k GT). All
submissions graded at multiplier ≈ 0.10 (floor), 0 failed MLPs. Reference:
`cov_prop` adjusted ≈ 8.4e-6 — the UT submissions are **~20× better**.

### 7.1 The LHS misranking — a methodology lesson

The 3-MLP local score said LHS (2.47e-7) beat UT (4.71e-7); **the leaderboard
said the opposite** (LHS 5.30e-7 vs UT 3.99e-7, both at multiplier 0.10 — verified
via the AIcrowd API, so *not* a compute-penalty artifact). LHS's raw MSE is simply
worse than UT's on the real MLP distribution. Two compounding evaluation errors:

- **Too few MLPs.** Per-MLP MSE spans ~7× and UT has high rep-to-rep variance, so
  ranking a ~30% difference needs many MLPs. The 3-MLP baked set caught a
  lucky-low LHS draw. Re-baking **8** whestbench MLPs already flips it to match
  the leaderboard: UT 4.91e-6 vs LHS 8.13e-6.
- **Wrong population.** The hand-rolled `build_mlp` used in `whest_harness.py`,
  `whest_ut.py`, and the `bench*` scripts gives UT a ~3× *higher* MSE than the
  competition's own generator (bench3 UT 14.9e-6 vs real ~4–5e-6). So those
  benchmarks measured a different distribution than the grader.

**Takeaway: rank methods only on a baked *whestbench* dataset with enough MLPs
(or the leaderboard itself), never on `build_mlp` MLPs or ≤ handful of MLPs.** The
harness rankings in §3 (sobol/cubature/etc.) were all on `build_mlp` and should be
re-validated this way before being trusted. The one robust, leaderboard-confirmed
result: **UT at `TARGET_FRACTION=0.095` (3.99e-7) is the best submission**, and UT
is machine-independent (pure FLOP-counted `matmul`+`QR`), unlike LHS.

### How to reproduce / submit

```bash
# contract + budget sanity
uv run whest validate --estimator experiments/estimator_lhs.py

# real scoring on a locally-baked dataset
DS=/tmp/whest_ds
uv run whest dataset bake --n-mlps 3 --width 256 --depth 32 \
    --n-samples 500000 --split mini --output "$DS"
uv run whest run --estimator experiments/estimator_lhs.py \
    --dataset "$DS" --split mini --runner subprocess
# confirm: Mean Compute Utilization < 0.10, Failed MLPs 0 of N

# submit (credentials via `whest login` once)
uv run whest submit --estimator experiments/estimator_lhs.py --watch
```

---

## 8. White-box structure — the weights, not just the map

Everything above (UT, LHS, CV) treats the MLP as a **black box**: a function into
which we push a Gaussian. Our best submission holds `mlp.weights` and never looks
at them — it uses each `W` only as a matmul operator. But the challenge is
*white-box*. This section asks what the weights tell us that sampling cannot, and
settles it with a probe.

### 8.1 The leading order is realization-independent (mean-field concentration)

Track one scalar per layer, the mean squared activation `q_l = (1/width)·E‖z_l‖²`.
For `W_ij ~ N(0, σ_w²/width)` the pre-activation `h = Wᵀz` is (large width)
zero-mean Gaussian with variance `σ_w² q_l`, so

```
q_{l+1} = E[ReLU(h)²] = ½ σ_w² q_l ,      m_{l+1} = E[ReLU(h)] = √(σ_w² q_l / 2π).
```

He init picks `σ_w² = 2`, giving `q_{l+1} = q_l` — **length preservation,
criticality.** From `q_0 = 1`, every layer has `q_l ≡ 1` and mean activation
`m_l = √(1/π) ≈ 0.564`. So **to leading order in width, every neuron's mean
activation is the same known constant `0.564`, independent of the realization of
`W`.** This is why `zeros` scores `~0.91` (the variance about 0) while
`mean_prop`/`cov_prop`, which just track `q` and its refinement, already reach
`~1e-4`. The realization-specific part of the answer lives entirely in the
**finite-width fluctuations** — the `O(1/√width)` corrections concentration washes
out. Those are what a white-box method must predict.

### 8.2 ReLU is not rotationally invariant — so the spectrum is useless

The natural low-dimensional summary of a random matrix is its **singular-value
spectrum** (Marchenko–Pastur); any function invariant under `W ↦ UWV`
(orthogonal `U,V`) depends only on it. But ReLU acts coordinatewise — the problem
is defined in the **standard basis**, and `G(UW) ≠ G(W)`. So the spectrum is
**not** a sufficient statistic: the informative quantities are **basis-aligned**
(how each weight column sits relative to the standard-basis activation vector), not
its rotation-invariant magnitude. This is the point the whole discussion kept
circling; the probe makes it quantitative.

### 8.3 Probe — what actually predicts the per-neuron final mean

For output neuron `j`, `E[z_{L,j}] = E[ReLU(h_j)]` with `h_j = ⟨W[:,j], z_prev⟩`,
so the pre-activation has mean `μ_j = ⟨W[:,j], μ_prev⟩` (the **alignment**) and std
`s_j`, `s_j² = W[:,j]ᵀ Σ_prev W[:,j]`. Measured over **6 real whestbench MLPs** at
256×32 (`N=1e6` ground truth; `whest_probe.py`):

| quantity | result | reading |
|---|---|---|
| corr( truth, alignment `⟨W[:,j],μ_prev⟩` ) | **+0.86** (0.85–0.87, rock-steady) | the alignment *is* the signal |
| corr( truth, column norm `‖W[:,j]‖²` ) | **0.00** | rotation-invariant magnitude is **inert** |
| R²( truth │ alignment, colnorm ), linear | 0.74 | one statistic, minus the CDF nonlinearity |
| R²( truth │ exact Gaussian-ReLU formula, *true* moments ) | **≈1.000** | the final rectification is essentially exact |
| non-Gaussian residual explained by weight stats | 0.05 | final-step non-Gaussianity is negligible & structureless |
| UT bias fraction of its MSE | ~0.35 (0.0–0.75, MLP-dependent) | UT is ~⅓ bias, ⅔ sampling noise |

Three conclusions:

1. **The per-neuron answer is a scalar function of one basis-aligned statistic** —
   the pre-activation mean `⟨W[:,j], μ_prev⟩` — because `s_j` concentrates (column
   norm barely varies at width 256), so the answer is `≈ E[ReLU(N(μ_j, s))]` with
   `s` nearly constant. Column norm, the rotation-invariant part, carries **no**
   per-neuron information — §8.2 confirmed.
2. **The bottleneck is moment propagation, not the final nonlinearity.** Given the
   *true* `μ_prev, Σ_prev`, the Gaussian rectification reproduces the truth
   (R²≈1.0): the final pre-activation is Gaussian by CLT over 256 terms. So real
   `cov_prop`'s `8.4e-5` is **accumulated moment error over 31 layers**, not
   final-step closure. A corrector should fix the *moment recursion across depth*.
3. **A UT-residual patch has limited headroom:** only ~⅓ of the UT's error is
   deterministic (weight-predictable) bias; the rest is irreducible sampling noise
   on a fixed cloud. The deterministic lever — correcting the moment recursion — is
   the larger and fully-learnable target.

*(The `R²≈1.0` uses true moments from a 1M-sample MC — it isolates the final
closure step, showing where the error is* not.*)*

### 8.4 Consequence for pooling

The predictive statistic is **per-neuron and basis-aligned**; the rotation-invariant
pooled magnitude is inert. A permutation-*invariant* sum-pool (→ scalar) recovers
the concentrated order parameters `m_l, q_l` — the part that is *already known* —
and discards the neuron identity where the fluctuation signal lives. A corrector
therefore needs **permutation-equivariant per-neuron features** (each neuron reading
its own column's alignment with the propagated moments), not a global pool.
Randomness-first remains fine — but pointed at the moment statistics and pooled
equivariantly.

*(What we built on these statistics — and why none of it beat UT — is §8.5.)*

### 8.5 The corrector — a principled negative result

Acting on §8.1–8.4 we built and tested four correctors, each at the **same
~6,144-point budget** as the UT submission, on real whestbench MLPs (8 MLPs ×
8 rotation seeds, `N=500k` ground truth). All four **fail to beat base UT**, and
*why* they fail is the useful part.

| corrector (`file`) | idea | vs base UT | why it fails |
|---|---|---|---|
| exact Gaussian closure (`whest_moment_closure.py`) | replace `cov_prop`'s gain rule by the exact per-layer ReLU covariance (MC) | **~100× worse** (`4.3e-4` vs `~4e-6`) | any per-layer re-Gaussianization drifts over depth; UT carries the true non-Gaussian joint |
| analytic last layer (`whest_analytic_last.py`) | feed UT's penultimate cloud moments through the exact rectification `σ·R(α)` | **~1.15× worse** | penultimate `Σ` from 6k points is too noisy; UT's direct average is already optimal |
| learned bias corrector (`whest_corrector.py`) | ridge on per-neuron weight features (alignment, `σ`, `α`, `Ψ`), train/test split across MLPs | **1.08×, unreliable** (1 of 6 test MLPs worse) | only ~⅓ of UT's error is deterministic bias; the rest is sampling noise a feature model cannot predict |
| two-radius rule (`whest_two_radius.py`) | 2-point Gauss radial quadrature to kill UT's degree-4 radial bias | **0.64× worse** | halving rotations to add a radius costs more angular variance than the radial bias it removes |

The four triangulate one conclusion: **UT's error is dominated by angular
sampling variance, and the best use of a fixed point budget is maximal angular
coverage at a single radius — exactly what the submission does.** The two-radius
experiment is decisive: even correcting UT's *only* deterministic bias (the
single-radius `E‖x‖⁴ = n²` vs the true `n(n+2)` mismatch) is a net loss, because
the angular coverage it sacrifices matters more.

**Why the weights don't help — the isotropy argument.** Angular variance is
driven by how `f` varies over the input sphere. To beat it *using the weights* we
would need weight-derived **anisotropy** — preferred input directions to sample
more densely. But at criticality with He-random weights the network is
**statistically isotropic in its input** (`x ~ N(0,I)`, first layer random
Gaussian ⇒ no preferred direction) — the same concentration as §8.1. So there is
no weight-derived angular structure for a smarter sampler to exploit: isotropic UT
is already matched to the problem's symmetry. The weights *do* fix the per-neuron
answer (via the last-layer alignment, §8.3), but that information is **already
fully contained in the propagated UT cloud**; extracting it separately only
re-introduces estimation noise — which is why the analytic-last-layer and learned
correctors cannot win.

**Conclusion.** The white-box thesis holds for the *answer* (a deterministic
function of the weights) but not for *improving the estimator*: black-box UT
already reads that function near-optimally, because its randomness is matched to
the isotropy the weights induce. For the scored final-layer mean of a *critical
random* ReLU net, structure buys almost nothing over good isotropic quadrature.
**The UT submission (`../estimator.py`, `3.99e-7`) stands as the champion.**

---

## 9. Open directions

- **Even-part control for the UT** *(deferred — longer discussion planned).* The
  UT's residual error is exactly the even, degree-≥4 subspace (§4.1, §4.3). A
  control variate capturing that (an `|·|`- or second-moment-based surrogate)
  would be the one thing that is *not* redundant with the UT's symmetry — the
  genuine way to improve it.
- **Push LHS further:** orthogonal-array / maximin LHS, or a real randomized
  Sobol/lattice (needs a pure-`fnp` implementation) for below-`1/√k` convergence.
- **Sampler + analytic control done right:** a control must be nonlinear and
  analytically integrable to beat a structured sampler — the hard open problem.
