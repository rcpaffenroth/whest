# The ARC White-Box Challenge: the answer is a deterministic function of the weights

Research log for our work on the
[ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026).
This folder holds our experiments; the repo root is a fork of the official
[starter kit](https://github.com/AIcrowd/whest-starterkit) (its `README.md` is
preserved).

---

## 0. The central insight — read this first

**The quantity we must predict is a deterministic function of the weights.**

The target is `μ = E_{x~N(0,I)}[z_L]`, the per-neuron mean activation. Once the
weights `W = (W_0,…,W_{L-1})` are fixed, the input distribution is *integrated
out* — there is no randomness left. So

```
μ = F(W)          — a fixed, deterministic function of the weights.
```

A genie holding `W` with unlimited compute computes `F(W)` exactly and gets
**zero error**. The competition is therefore not "estimate a random quantity"; it
is **"compute the deterministic function `F(W)` as cheaply as possible."** Every
method here is just a different cheap approximation of the *same* `F(W)`.

This single fact reorganizes everything, and earlier versions of this log missed
it. Three consequences to keep in mind throughout:

1. **Two classes of method, not one.**
   - *Sampling / quadrature* (Monte-Carlo, UT, LHS, Sobol): approximate the
     integral `F(W)=∫ z_L dN(0,I)` by pushing points through the network. Every
     such method has an **irreducible sampling-variance floor** at a fixed point
     budget — it is estimating a deterministic number with a random rule.
   - *Analytic / white-box* (mean- and covariance-propagation, and learned
     refinements of them): compute `F(W)` **directly from the weights** by
     propagating the distribution's moments through the layers. **No sampling,
     no variance floor**

2. **The residual of any estimator is also deterministic in `W`.** For a base
   estimator `Ê(W)`, the error `R(W) = F(W) − Ê(W)` is a fixed function of the
   weights. A *corrector* that learns `R(W)` and adds it back can, in principle,
   reach zero error — **provided `R(W)` is a cheaper function than `F(W)` itself.**
   That is the whole bet of a correction approach: you stand on a base estimator's
   shoulders and learn only the (hopefully smooth, low-complexity) residual, not
   `F` from scratch. Whether `R(W)` is cheaply representable within the FLOP
   budget is *the* open research question — and where structured-sparse models
   (block-diagonal + permutation, "Monarch") earn their place.

3. **Sampling-variance is a wall on *samplers*, not on the problem.** A recurring
   error below was to prove "no isotropic sampler can do better here" and conclude
   "nothing can do better." Those are different statements. The sampling floor
   says nothing about the analytic route, which has no variance at all. The
   leaderboard confirms this by a floor argument: any pure sampler bottoms out at
   its own directional variance, so scores well below that floor can only come
   from **exploiting `W` directly** (analytic moment propagation) — i.e. from
   leaving the sampling class.

**How to read the rest of this log.** §2–§6 characterize the *sampling class* and
find its best member (a randomized unscented transform, our current champion).
That work is correct and useful — but it is a statement about one class of
methods, and its "we are near-optimal" conclusions are bounded by the sampling
floor, not by `F(W)`. §7 is the submission record. §8 is the white-box structure:
a probe shows the answer is a scalar function of one **basis-aligned** weight
statistic, that the bottleneck is **moment propagation across depth** (not the
final nonlinearity), and — corrected here — that this is precisely the analytic
route with the higher ceiling. §9 sets the direction: a learned corrector of the
deterministic moment recursion.

---

## 1. The competition in one page

**The research question (from ARC):** *Can you predict a neural network's
behavior by analyzing its weights, instead of running it on many inputs?* Read
through §0: the honest answer the challenge is fishing for is that the behavior
(`μ`) *is* a function of the weights, and the sport is computing it cheaply.

**The task.** You are given one randomly-initialized ReLU MLP — square,
**width 256, depth 32** in Phase 1, He-initialized weights `W ~ N(0, 2/width)`,
input `x ~ N(0, I)`. Each layer computes `y = ReLU(Wᵀx)`. You must predict the
**per-neuron mean post-ReLU activation at every layer** — a `(depth, width)`
array `= F(W)`. Ground truth is a large Monte-Carlo average (itself a finite-N
estimate of the deterministic `F(W)`).

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
   width 256. Below the floor only accuracy matters. *If you choose to sample*,
   the problem collapses to a **quadrature problem**: estimate `F(W)` with a
   ~6,500-point budget. But note the "if": an analytic method spends its FLOPs
   propagating moments, not sampling, and lives under the same floor.
2. **Only the final layer counts, and depth 32 is where the analytic closures
   break.** Correlations build with depth and ReLU couples the joint
   distribution, so the Gaussian-closure assumptions in the analytic route
   degrade exactly in the scored regime. This is a statement about *today's crude
   closures*, not a barrier — §8 shows the closure error is deterministic and
   learnable.

---

## 2. Toy cases — isolating the two analytic approximations

These isolate where the *analytic* route's error comes from — useful because that
route, not the sampler, is where the headroom is (§8).

| File | Shape | Point |
|---|---|---|
| `whest_1x1.py` | scalar chain | Width 1 ⇒ **no correlations**, so mean- and covariance-propagation are the *same* algorithm. The only error is the **Gaussian-pre-activation assumption**, which compounds with depth (analytic systematically *over*-predicts: a Gaussian puts less mass at exactly 0 than the true rectified distribution). *Aside:* in 1D a single negative downstream weight kills the chain to exactly 0 forever — most seeds give a dead chain; seed 56 keeps it alive. |
| `whest_2x2.py` | 2 neurons | Smallest case with off-diagonal covariance. The two methods **agree at layer 0** (input cov = I) and **diverge from layer 1 on** — the closure-drift thesis in miniature. |
| `whest_8x8.py` | 8 neurons | **Covariance propagation is a coin flip vs mean propagation here** (won 6/10 seeds, sometimes 2–3× worse). The covariance advantage is a **large-width phenomenon**: the crude gain-rule ReLU covariance update `Φ(αᵢ)Φ(αⱼ)Σ[i,j]` adds error comparable to the correlation signal when width is small. |

Analytic baselines (raw final-layer MSE at 256×32): zeros `~0.91`,
`mean_propagation ~9.5e-4`, `covariance_propagation ~8.4e-5` (adjusted score
`~8.4e-6`). Our clean re-implementation reproduces `cov_prop` (~1.1e-4 over 4
MLPs), confirming correctness. **These are the *crude* analytic closures**; §8.3
shows that with accurate moments the same rectification step is essentially
exact, so `cov_prop`'s error is fixable closure drift, not a floor.

---

## 3. Method comparison within the sampling class — `whest_harness.py`

Compares sampling methods (and the crude analytic baselines) at matched compute,
by final-layer MSE vs a large-N MC ground truth. **Key result at 256×32, matched
FLOPs (`k=512 ≈ cov_prop`'s `O(depth·width³)` cost):**

| method @ k=512 | final-layer MSE | vs `cov_prop` (1.08e-4) |
|---|---|---|
| `random_mc` | 1.89e-4 | worse |
| `control_variate` | 9.97e-5 | ~tie |
| `antithetic_mc` | 9.82e-5 | ~tie |
| `sobol_qmc` | 5.85e-5 | 1.8× better |
| `cubature_ut` | 4.90e-5 | 2.2× better |

At the free-budget operating point (~6,500 points) the best samplers reach
`~5e-6`, roughly **20× better than *crude* `cov_prop`** on the scored metric
(both at the floored 0.1 multiplier). **Within the sampling class:** structured
samplers (UT, QMC) beat MC; control-variate and antithetic give little.

**Read this conclusion carefully.** "Sampling beats `cov_prop` at competition
scale" is true *against today's crude closure*, and it is **not** a claim that
sampling beats the analytic route in general. `cov_prop` loses because its
gain-rule Gaussian closure drifts over 31 layers, not because the analytic idea
is weak — §8.3 shows that route's ceiling is ~0. The correct summary: *among
methods we have actually built, the best is a sampler; the analytic route has the
higher ceiling and is the open direction.* (All harness numbers are on the
hand-rolled `build_mlp` generator — see §7.1 for why those rankings must be
re-validated on real whestbench MLPs before being trusted.)

---

## 4. The mathematics of the samplers

Characterizing the best sampler and *its* error subspace — which then tells us
exactly what a corrector must attack.

### 4.1 Unscented transform (UT) — `whest_ut.py`, `../estimator.py`

Represent `N(0, I_n)` by the deterministic **sigma-point set `±r·eᵢ`**
(`i = 1..n`). With `r = √n` these `2n` points reproduce the input's mean (`0`) and
covariance (`I`) exactly — the **degree-3 spherical-radial cubature rule**. Apply
a **Haar-random rotation** and average several rotations: rotation preserves the
exact moment-matching while averaging out the fixed-axis bias (the "randomized"
UT). Push all points through the network, average per layer (equal weights ⇒ the
UT expectation is the plain mean). More rotations ⇒ more points ⇒ lower variance.

Why UT over the crude analytic route: it makes **no Gaussian-closure and no
gain-rule approximation**, targets the mean directly, and gives every layer for
free — at the cost of a sampling-variance floor the analytic route does not have.

**The radius correction (a free win, and a cautionary tale).** The base rule sets
`r = √n` to match `E[‖x‖²] = n`. But ReLU is positively homogeneous of degree 1,
so `ReLU(wᵀx) = ‖x‖·ReLU(wᵀx̂)`, and the *mean* is controlled by the **first**
radial moment `E‖x‖ = √2·Γ((n+1)/2)/Γ(n/2) ≈ √(n−½)`, not `√n`. A single shell at
`√n` therefore overestimates by `√n/E‖x‖ ≈ 1 + 1/(4n)` — a deterministic radial
bias, present from layer 0 and compounding with depth. **Moving the single shell
to radius `E‖x‖` removes it at the same point budget**, cutting the final-layer
bias floor `1.9e-6 → 2.4e-7` (measured in `whest_radius.py`) and improving the
leaderboard submission (§7). This is the first clean case where correcting a
*deterministic* piece of the error won — and it directly overturns the old
"two-radius" negative result (§8.5).

#### The even/odd (symmetry) structure — bounds what any UT correction can do

The UT point set is **inherently ± symmetric** (every `ξ` comes with `−ξ`), so
the estimate sees only the **even part** of the integrand; the odd part is
annihilated exactly. Combined with the radial scaling it integrates **all
polynomials up to degree 3, and every odd-degree term, exactly.** Its residual
error is therefore **only the even degrees ≥ 4** — the `|·|`-like part of ReLU
that survives symmetrization (`ReLU(t) = t/2 + |t|/2`; the even `|t|/2` remains).

**Consequence:** a *linear* control `Z = Ax` is odd, so the UT integrates it
exactly and the control-variate correction vanishes. **A linear control variate
does nothing for the UT**; more generally any polynomial control of degree ≤ 3 is
redundant. To help the *sampler* you would need a control capturing the even,
degree-≥4 rectification tail — the hard nonlinear content cheap surrogates lack.

### 4.2 Latin hypercube / stratified sampling (LHS) — `estimator_lhs.py`

Sample `N(0, I)` by a Latin hypercube: stratify each of the `width` coordinates
into `k` equal-probability bins, visit each once, place a random offset inside the
bin, map to Gaussian by the inverse normal CDF. This removes **per-coordinate
(main-effect) sampling variance** at one forward pass per point. Whether that
beats the UT's exact cancellation is settled next.

### 4.3 Which sampling rule should win here — a Hermite-variance view

The right way to compare sampling rules is a variance decomposition, and it
favors the UT. Expand the integrand `f(x)` (a final-layer activation) in the
**Hermite basis** orthonormal to the Gaussian:

```
f = Σ_α c_α H_α ,        E[f] = c_0   (the quantity we want).
```

Every quadrature estimator of `c_0` has variance `Σ_{α≠0} c_α² · r(α)`, where
`r(α)` measures how badly the rule handles mode `α`. Plain MC: `r(α) = 1/k` for
all `α`. The structured rules *zero out* whole subspaces:

- **UT** (± symmetric, degree-3 exact) zeros **all odd `|α|`, and all `|α| ≤ 3`.**
  Residual variance lives in **even total degree ≥ 4.**
- **LHS** (marginal stratification) zeros **all main effects** — modes on a single
  coordinate, at any degree. Residual: all *interaction* modes.

The decisive comparison is the *unique* kills:

| rule | uniquely removes |
|---|---|
| UT | **low-degree interactions** — `H_{e_i+e_j}` (i≠j, degree 2), degree-3 cross terms, all odd interactions |
| LHS | **high-degree main effects** — `H_{4·e_i}, H_{5·e_i}, …` (single coordinate, degree ≥ 4) |

**In a deep ReLU network the nonlinearity shows up predominantly as low-degree
*interactions*.** The between-neuron covariance that accumulates with depth — the
structure that makes the problem hard — is exactly degree-2 interaction content
`H_{e_i+e_j}`, **inside the UT's kill set and inside LHS's blind spot.** So the
decomposition predicts **UT beats LHS here**, as the leaderboard confirmed (§7.1).
This also corrects the intuition that motivated the LHS detour ("depth 32 is so
nonlinear a degree-3 rule captures too little"): the nonlinearity is
*interaction*-degree, the UT's sweet spot. There is **no hard degree-3 floor** —
the random rotation turns fixed-axis degree-≥4 bias into variance that averages
as `~1/rotations`.

**Corollary.** Within the sampling class, the UT's only remaining error is the
even-degree-≥4 subspace. But note the framing of §0: escaping that subspace with a
better *sampler* is a smaller prize than leaving the sampling class entirely for
the analytic route (§8).

---

## 5. The hybrid investigation (route b) — a clean negative result *for samplers*

We tested "structured sampler + control variate," the control being the
**mean-field-linearized network** (`Z = Ax`, `E[Z]=0`). It **does not help**, for
two measured reasons: (1) **redundancy** — the control is linear/odd, which the UT
annihilates by symmetry and LHS removes by stratification, so it corrects variance
the sampler already removed; (2) **it costs a second forward pass**, so at fixed
budget the CV variant gets half the points.

**Matched-budget benchmark** (`bench2.py`, 256×32, ~6,144 forward-units, 4 MLPs ×
10 reps, raw final-layer MSE, mean ± std):

| method | MSE | note |
|---|---|---|
| **`lhs`** (pure) | **6.99e-6** ± 5.0e-6 | best *in this build_mlp benchmark* |
| `ut` (12 rot) | 1.39e-5 ± 2.1e-5 | high variance at few rotations |
| `mc` | 1.44e-5 ± 1.2e-5 | |
| `lhs+cv` | 1.47e-5 ± 1.1e-5 | **worse than pure `lhs`** |
| `mc+cv` | 1.85e-5 ± 1.6e-5 | **worse than pure `mc`** |

`sampler + CV` is dominated by `just spend the budget on the sampler`. (The
LHS-beats-UT ordering here is a `build_mlp` / small-sample artifact that did *not*
hold on the leaderboard — §7.1.) This is a solid negative result **about linear
controls for samplers**; it says nothing about the analytic route.

> Dependency note: the grader environment has **no `scipy`**, so submissions use
> only `flopscope`/`whestbench`. `flopscope` provides
> `flops.stats.norm.{cdf,pdf,ppf}` and a FLOP-counted `fnp.linalg.qr`. There is no
> Sobol in `flopscope`; LHS is our pure-`fnp` QMC-family sampler.

---

## 6. Coding details of the submissions

Both submissions are single self-contained files (only `flopscope`, `whestbench`,
`math`), **budget-aware**: they read `budget`, spend `TARGET_FRACTION` of it, and
size the point count from a closed-form cost.

**`../estimator.py` — UT.** `cost_per_rotation = 4·depth·width³`;
`rotations = int(TARGET·budget / cost_per_rotation)`. Sigma points are
`±E‖x‖ · (columns of a random orthogonal Q)` via `fnp.linalg.qr` — **note the
radius is `E‖x‖`, not `√width`** (§4.1), computed as
`√2·exp(lgamma((w+1)/2) − lgamma(w/2))`. Points are concatenated, propagated,
averaged per layer.

**`estimator_lhs.py` — LHS.** `cost_per_point = 2·depth·width²`;
`k = int(TARGET·budget / cost_per_point)`. Points via
`strata = argsort(argsort(uniform))` (independent permutation per column),
`u = (strata + uniform)/k`, then `flops.stats.norm.ppf(u)`.

**Tuning `TARGET_FRACTION`.** The multiplier is pinned at `0.1` for usage `≤ 10%`,
so we push toward the floor. UT uses `0.095` (≈9.7% effective); LHS uses `0.09`
(its `argsort`/`ppf` add residual wall-time on top of raw FLOPs). Verify effective
`Mean Compute Utilization < 10%` after any change.

---

## 7. Submissions — verified end to end

Verification chain: `whest validate` (contract) → `whest run` on a baked
whestbench dataset (real scoring) → `whest validate-package`.

| # | Method | `TARGET` | **leaderboard adj.** | id |
|---|---|---|---|---|
| — | UT (pre-session) | 0.08 | 4.29e-7 | (manual) |
| a | UT (radius `√n`) | 0.095 | 3.99e-7 | 315205 |
| b | LHS | 0.09 | 5.30e-7 (worse) | 315209 |
| c | **UT + radius fix (`E‖x‖`)** | 0.095 | **3.39e-7 ← champion** | 316178 |

The radius fix (§4.1) is a provably-correct, FLOP-neutral change that took the
best submission from 3.99e-7 to **3.39e-7**. Reference: crude `cov_prop` adjusted
≈ 8.4e-6.

### 7.1 The LHS misranking — a methodology lesson

A 3-MLP local score said LHS beat UT; **the leaderboard said the opposite** (LHS
5.30e-7 vs UT 3.99e-7, both at multiplier 0.10 — verified via the AIcrowd API, so
not a compute artifact). Two compounding errors: **(1) too few MLPs** — per-MLP
MSE spans ~7× and UT has high rep variance, so ranking a ~30% difference needs
many MLPs (re-baking 8 whestbench MLPs already flips it to match the leaderboard:
UT 4.91e-6 vs LHS 8.13e-6); **(2) wrong population** — the hand-rolled `build_mlp`
gives UT a ~3× *higher* MSE than the competition generator, so `build_mlp`
benchmarks measure a different distribution than the grader.

**Takeaway: rank methods only on a baked *whestbench* dataset with enough MLPs (or
the leaderboard), never on `build_mlp` or a handful of MLPs.** The harness
rankings in §3 were all on `build_mlp` and must be re-validated this way.

### How to reproduce / submit

```bash
uv run whest validate --estimator estimator.py
# real scoring on the official public generator (weights + ground truth):
uv run whest run --estimator estimator.py \
    --dataset "hf://aicrowd/arc-whestbench-public-2026@v1-phase1" \
    --streaming --n-mlps 8 --json
uv run whest submit --estimator estimator.py --watch   # credentials via `whest login`
```

---

## 8. White-box structure — where the deterministic `F(W)` actually lives

§2–§7 treat the MLP as a **black box**: a function we push a Gaussian into. But `μ
= F(W)` is a function *of the weights*, and this section pins down what part of the
weights carries the answer — and, corrected here, shows it points straight at the
analytic route of §0, not at "UT is unbeatable."

### 8.1 The leading order is realization-independent (mean-field concentration)

Track `q_l = (1/width)·E‖z_l‖²`. For `W_ij ~ N(0, σ_w²/width)` the pre-activation
`h = Wᵀz` is (large width) zero-mean Gaussian with variance `σ_w² q_l`, so

```
q_{l+1} = ½ σ_w² q_l ,      m_{l+1} = √(σ_w² q_l / 2π).
```

He init (`σ_w² = 2`) gives `q_{l+1} = q_l` — **length preservation, criticality.**
From `q_0 = 1`, every layer has `q_l ≡ 1` and mean activation `m_l = √(1/π) ≈
0.564`. So **to leading order in width, every neuron's mean is the same known
constant `0.564`, independent of `W`.** (This is why `zeros` scores `~0.91` while
`mean_prop`/`cov_prop` reach `~1e-4`.) The realization-specific answer lives in the
`O(1/√width)` **finite-width fluctuations** — that is what any white-box method
must predict.

### 8.2 ReLU is not rotationally invariant — so the spectrum is useless

Any function invariant under `W ↦ UWV` depends only on the singular-value
spectrum. But ReLU acts coordinatewise — the problem is in the **standard basis**,
`G(UW) ≠ G(W)` — so the spectrum is **not** a sufficient statistic. The
informative quantities are **basis-aligned** (how each weight column sits relative
to the standard-basis activation vector), not the rotation-invariant magnitude.

### 8.3 Probe — what actually predicts the per-neuron final mean (`whest_probe.py`)

For output neuron `j`, `E[z_{L,j}] = E[ReLU(h_j)]`, `h_j = ⟨W[:,j], z_prev⟩`, so
the pre-activation mean is `μ_j = ⟨W[:,j], μ_prev⟩` (the **alignment**) with std
`s_j² = W[:,j]ᵀ Σ_prev W[:,j]`. Over **6 real whestbench MLPs** (`N=1e6` GT):

| quantity | result | reading |
|---|---|---|
| corr( truth, alignment `⟨W[:,j],μ_prev⟩` ) | **+0.86** (rock-steady) | the alignment *is* the signal |
| corr( truth, column norm `‖W[:,j]‖²` ) | **0.00** | rotation-invariant magnitude is **inert** (confirms §8.2) |
| R²( truth │ **exact Gaussian-ReLU formula, true moments** ) | **≈1.000** | given accurate moments, the rectification is essentially exact |
| non-Gaussian residual explained by weight stats | 0.05 | final-step non-Gaussianity is negligible |

**This is the most important measurement in the log, and it points at the analytic
route:**

1. **The per-neuron answer is a scalar function of one basis-aligned statistic**,
   the pre-activation mean `⟨W[:,j], μ_prev⟩` (since `s_j` concentrates at width
   256). Rotation-invariant magnitude carries **no** per-neuron information.
2. **The bottleneck is moment propagation across depth, not the final
   nonlinearity.** Given the *true* `μ_prev, Σ_prev`, the Gaussian rectification
   reproduces the truth (R²≈1.0) — the final pre-activation is Gaussian by CLT over
   256 terms. So crude `cov_prop`'s `8.4e-5` is **accumulated closure drift over 31
   layers**, and its ceiling with an accurate recursion is `≈ 0`. **This is the
   deterministic `F(W)` route of §0, and it has essentially unlimited headroom.**

### 8.4 Consequence for a corrector's architecture

The predictive statistic is **per-neuron and basis-aligned**; a permutation-
*invariant* pool (→ scalar) only recovers the already-known order parameters
`m_l, q_l` and discards the neuron identity where the fluctuation signal lives. So
a corrector needs **permutation-equivariant per-neuron features** — each neuron
reading its own column's alignment with the propagated moments — not a global
pool. (Note the symmetry precisely: permuting neurons within a layer permutes the
answer the same way, so features must be equivariant to that relabeling.)

### 8.5 The earlier "UT is unbeatable" conclusion was a category error

A previous version of this log built four correctors on top of the UT — exact
Gaussian closure (`whest_moment_closure.py`), analytic last layer
(`whest_analytic_last.py`), a learned ridge bias corrector (`whest_corrector.py`),
and a two-radius rule (`whest_two_radius.py`) — found that **none beat the base
UT**, and concluded "structure buys almost nothing; the UT is the champion; the
white-box thesis holds for the *answer* but not for *improving the estimator*."

**That conclusion was wrong, and it is exactly the error §0 warns about.** What the
four correctors actually established is narrower and correct: *you cannot cheaply
improve a good **isotropic sampler** by re-extracting, from a noisy finite cloud,
information the cloud already contains.* The "isotropy argument" (He-random critical
nets are statistically isotropic in their input, so no weight-derived anisotropy
lets a *sampler* place points better) is a valid statement **about samplers**. It
does **not** bound the problem, because the analytic route does not sample and has
no angular variance to beat.

Three concrete corrections:

- **The two-radius negative result was a method artifact, not a law.** It failed
  because adding a second shell *halved the rotations* at fixed budget, trading
  angular variance for a small radial-bias fix. The **single-shell radius rescale**
  (§4.1) corrects the *same* deterministic radial bias for **free** — and it *won*
  (§7, 3.99e-7 → 3.39e-7). Correcting a deterministic error is not a net loss; the
  earlier experiment just paid for it the wrong way.
- **"The alignment info is already in the UT cloud, so weights don't help" is
  backwards.** The cloud is a *noisy, finite-sample* estimate of `⟨W[:,j],
  μ_prev⟩`; the weights give it **exactly**. The right move is not to re-read the
  alignment off the cloud (which re-introduces noise) but to **propagate the
  moments analytically from the weights**, which §8.3 shows is exact in the limit.
- **UT's residual is mostly variance *precisely because* the radius fix removed
  most of its deterministic bias.** That confirms the sampling floor is real — for
  *samplers*. It says nothing about the analytic route.

**Corrected conclusion.** The white-box thesis holds for both the answer *and* the
estimator; you just have to leave the sampling class to exploit it. The current
champion is the best *sampler* (`../estimator.py`, UT + radius fix, `3.39e-7`), and
the higher-ceiling direction is a learned correction of the **deterministic moment
recursion** (§9).

---

## 9. Open direction — a learned corrector of the deterministic moment recursion

Everything above converges on one program. `μ = F(W)` is deterministic (§0); the
answer is a per-neuron scalar function of a **basis-aligned** weight statistic
(§8.3); the true bottleneck is **closure drift in the moment recursion across
depth** (§8.3.2), whose ceiling is `≈ 0` (R²≈1.0 with accurate moments). So:

- **Base estimator: the deterministic moment recursion**, not (only) the sampler.
  Its residual `R(W) = F(W) − Ê(W)` is deterministic — **no irreducible sampling
  noise** — hence *fully* learnable, unlike a corrector on the UT (whose residual is
  mostly variance). This is the higher-ceiling target.
- **Corrector form: a per-layer recurrence with weight-dependence.** A learned map
  `g` that refines the propagated moments layer-to-layer, reading each layer's
  weights `W_k`, with a per-neuron equivariant readout (§8.4). This is a *learned
  moment-closure*: the closed form has no clean expression past low order, so we
  learn the correction the crude gain-rule misses. The hidden state carries the
  higher-moment information the Gaussian closure discards.
- **Why weight-dependence is the substrate, not a garnish (§0).** `F(W)` is a
  function of `W`; the residual is too; a genie with `W` reaches zero error. The
  research question is only whether `R(W)` is **cheaply** representable within the
  FLOP budget — which is why the map should be a large but **structured-sparse**
  (block-diagonal + permutation, "Monarch") function of the weights: high capacity
  at low FLOP cost. An ablation with weight-dependence switched off is the *control*
  that measures what the weights are worth — expected weak, not a gate.

Design notes and the running argument live in `experiments/tasks/RCP_ideas_v1.md`;
the bias/variance and radius experiments that grounded the reframe are
`whest_bias_variance.py` and `whest_radius.py`.

*Deferred sampler-side idea (lower ceiling): an even-degree-≥4 control variate for
the UT — the one control not redundant with its symmetry (§4.1, §4.3). Kept for
completeness; the analytic route above is the priority.*
