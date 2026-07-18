# The ARC White-Box Challenge: F(W) is a dynamical system, learned from the weights

Research log for our work on the
[ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026).
This folder holds our experiments; the repo root is a fork of the official
[starter kit](https://github.com/AIcrowd/whest-starterkit) (its `README.md` is
preserved).

---

## 0. Philosophy & standing orders — read this first, do not re-litigate

This section exists because RCP has had the **same conversation with multiple
LLMs** and is tired of it. Internalize it before proposing anything. It is written
in certain terms on purpose.

**The framing: this is a dynamical system, and we learn it from the weights.**
The forward pass `z_{l+1} = ReLU(W_lᵀ z_l)` is a **random dynamical system** — a
product of random matrices acting on a distribution. The target
`μ = F(W) = E_{x~N(0,I)}[z_L]` is a *deterministic functional of that system's
terminal distribution* (the input is integrated out). `F`'s heavy tail across
random `W` is **not noise** — it is the **large-deviation statistics of the
finite-time Lyapunov exponent** (some `W` sit in the expanding/chaotic regime,
some contracting). **The research goal is to LEARN `F(W)` directly from `W`**,
treating a recurrent model that scans `W_1,…,W_L` as a *learned simulator of the
distribution's dynamics*. That is the interesting problem, and the one RCP cares
about (dynamical systems). Everything else is scaffolding.

**Standing orders:**

1. **Learn from `W` with recurrent / sequence models — this is the priority.** The
   layers are an ordered sequence; an RNN/LSTM/GRU (or a structured recurrence)
   over them is the native tool. Push it *hard* — scale data and architecture —
   before concluding anything about what is or isn't learnable.

2. **Sampling (MC/UT/QMC) and analytic moment-propagation are EFFECTIVE BUT
   BORING — a red herring for the research goal.** They are cheap only because the
   FLOP budget is generous, not because they are the point. The unscented
   transform is merely our current *leaderboard submission* (§7): a solved
   baseline, not a research frontier. **When asked about learning from `W`, do not
   reach for sampling/moments and pronounce the ML route unnecessary or dead.**
   That reflex is the exact thing RCP is tired of.

3. **Never declare `F(W)` (or any target) "unlearnable" from one architecture or
   one data scale.** Escalate architecture and data first. This has already been
   wrong *twice in a single session:*

   | claim an LLM made | what actually happened |
   |---|---|
   | "flat MLP on flatten(W) plateaus at R²≈0.26 → `F` isn't black-box-learnable; a representation limit" | **wrong.** A GRU over the *ordered layers* → R²=0.69; an LSTM → **0.86**; and it *climbs with data* (0.48→0.65→0.83 at 100k→300k→600k nets, still rising). A plateau of one architecture ≠ a property of the problem. |
   | "the residual `R=F−UT` is unlearnable from `W`" | half-right, and again the fix was *more information, not surrender*: `W` alone → `R` gives R²≈0.03 (5% of UT's error), but feeding the RNN the **per-layer UT moments** as input takes it to R²=0.19 (20% removed, §10). The residual is learnable given the right features. |

4. **The likely endgame is a HYBRID — a learned `G(W-features, cheap-estimator-features)`** —
   not pure learning-from-`W` *or* pure sampling. What is deprioritized is only the *naive
   additive* corrector `F ≈ UT + learned(R)` and the *hand-crafted* moment-closure (§9):
   moment-propagation with a patch. But a model that *jointly* learns from the weights **and**
   cheap estimator outputs (UT per-layer moments as input features) is very much live — it is
   exactly the `W+UT` experiment (§10), and feeding UT features unlocked the residual the
   additive form could not. **Use cheap sampling/moments as *features* for a weight-learner,
   never as the answer.**

**Communicating with RCP.** RCP is an applied mathematician fluent in linear
algebra (SVD-native). Explain architectures and ideas in **linear-algebra terms**
— matrices, bases, projections, orthogonal/permutation-group actions, Hadamard and
matrix products — and **minimize per-"neuron" language**. E.g. the equivariant
model's state is *a `d×c` matrix that co-transforms with the activation vector
under the layer's basis permutation* (§10.1), not "per-neuron features."

**Setup facts that bite (they are dynamical-systems artifacts).** Width-2 depth-32
ReLU nets are **100% dead** (`F≡0`): the surviving input fraction halves every ~2
layers, and because ReLU is positively homogeneous this is *scale-invariant* — no
initialization fixes it; **width** is what keeps deep ReLU nets alive. The toy
therefore uses **width 8, depth 8** (`whest_toy.py`, `F: R⁵¹²→R⁸`), and the
recurrent experiments live in `rnn_harness.py`.

**How to read the rest.** §1 is the competition spec. §2–§8 are the **sampling +
analytic record** — correct, useful, the source of the current champion
submission, but *the boring part*: skim it, don't innovate there. §9 is the
deprioritized moment-corrector idea. **§10 is the live direction and its results.**

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

## 9. Deprioritized — learned corrector of the moment recursion (kept for the record)

**RCP finds moment/corrector methods boring; this is NOT the direction (see §0).**
The idea: learn the residual of the deterministic moment recursion — its error is
closure drift across depth (§8.3), deterministic, ceiling ≈0. Attractive on paper
(no sampling noise; per-neuron equivariant readout per §8.4; a Monarch-sparse map
for FLOP-cheap capacity). But it is moment-propagation with a learned patch — the
boring class. Note the *additive* form (learn `F−UT`) fails from `W` alone; but feeding UT
moments as *features* to a weight-learner does help (§10) — so the live version is the hybrid
`G(W,UT)` of §0.4, not this hand-crafted closure. Recorded so the next reader doesn't
re-derive it. (Also deferred, lower ceiling: an even-degree-≥4 control variate for the UT,
§4.1/§4.3.)

---

## 10. The live direction — learn F(W) directly with recurrent models

The forward pass is a random dynamical system (§0); an RNN scanning `W_1,…,W_L` is
a learned simulator of it. Harness: **`rnn_harness.py`** — one `make_core` factory
swaps `nn.RNN/LSTM/GRU`; `W` is fed as `K` layer-tokens of `d²` features; target
`log1p(F)`; R² reported in log-space on leakage-proof val/test; 600k-network pool
cached. Toy geometry `whest_toy.py`, width 8 / depth 8. Design notes:
`tasks/RCP-ideas-v2.md`.

**Results so far (held-out R², log-space):**

- **Architecture ≫ size.** Flat MLP on flatten(W) tops out ≈0.26; a residual MLP
  *climbs* with data (→0.36 at 600k, no plateau); respecting the ordered layers
  jumps far higher. *The flat-MLP "plateau" is not a property of `F`.*
- **Recurrent comparison** (600k nets, hidden 256, 2 layers): **LSTM 0.863 > GRU
  0.827 > vanilla RNN 0.663** (params 858k / 644k / 216k — capacity confounded
  with gating; param-matched sweep still TODO).
- **Data scaling (GRU): 0.48 → 0.65 → 0.83** at 100k → 300k → 600k, **still
  rising.** Learning from `W` is data-limited here, not capability-limited.
- **UT is a superb *feature*; the hybrid beats additive correction** (`exp_ut_features.py`).
  One **fixed** 16-point UT (no rotations → deterministic label) explains **99.4%** of Var(F).
  Feed the RNN the per-layer UT moments (mean + cov-diag) as input: `UT`-features→`F`
  R²=**0.996**; `W+UT`→`F` 0.995 (≈ `UT` — raw `W` adds little *at 300k*); and the residual,
  dead from `W` alone (0.03), becomes learnable given UT features (**0.19**, 20% of UT's error
  removed). This is the hybrid `G(W,UT)` of §0.4; the additive `F−UT` was the wrong form.
- **Width → the RMT limit changes the problem** (`exp_width_scale.py`, depth 8, `d`=8/16/32).
  Concentration confirmed (mean `F` 0.30→0.42→0.49 → mean-field 0.564; tail p99/med 9.6→2.7).
  **`UT`-features → sufficiency** (R² 0.995→0.998) while **`R²(W)` *falls*** at fixed data
  (0.46→0.31→0.27): the raw-weight token is `d²`-dim and lacks neuron symmetry.
- **But raw-`W` is data-limited, not dead** (`exp_w_at_scale.py`). Width 8: `R²(W)` climbs
  0.60→0.85→**0.97** at 300k→600k→1.2M, chasing UT's 0.995 — the `W`-vs-`UT` gap is a *data*
  gap. Width 32: stalls (0.27→0.30 to 400k) — data alone doesn't rescue it → the missing
  ingredient is Rung 1.
- **Equivariance breaks the width-32 wall** (`equivariant.py`, §10.1; width 32, N=400k). The
  matrix-state equivariant model reaches **R²(log)=0.982 with 7.4k params** vs the flat LSTM's
  0.27–0.30 with 858k — 30–40× fewer params, 3× R², from raw `W`. It is effectively a *learned
  moment-propagator*. Sweep + 3-seed confirm (`equiv_confirm.py`): **residual row-cell wins,
  R²=0.981±0.000, 7.4k params — gating is unneeded/harmful because `W_lH` already carries the state**
  (flips the flat-model "gating helps"; gru/mean 0.977±0.001 close 2nd). **Mean-pool suffices**: for
  residual mean=mean2 (both 0.981); for gru, mean 0.977 ≫ mean2 0.852 — the 2nd-moment (`q_l`) pool
  never helps, **contradicting §10.2's prediction** (the row-cell reconstructs `q_l` from the mean
  channel). attention (0.50) and lstm (0.44) hurt; hybrid `W+UT` 0.92 (pure-`W` already near
  UT-sufficiency, so UT adds little). Winner: **residual + mean pool, pure-W**.
- **It scales to competition width — R² RISES** (`exp_width_scale_equiv.py`, depth 8, residual/mean,
  **5.3k params at EVERY width**): R²(log) **0.983 → 0.991 → 0.995 → 0.993** at d = 32/64/128/**256**,
  while N *shrinks* 400k→200k→50k→**12k**. Concentration makes the problem cleaner toward the
  RMT/competition regime and the width-independent equivariant model exploits it: **at competition
  width 256, R²=0.993 from 12k nets with a 5.3k-param model.** The wall is gone — learning-`F`-from-`W`
  works at scale, dense (12–24 min/width, Monarch not yet needed).
- **Depth-scaling softens but holds** (`exp_depth_scale_equiv.py`, width 64, residual/mean, 5.3k params
  at every depth): R²(log) 0.991 → 0.987 → 0.981 → **0.951** at K = 8/16/24/**32**, N shrinking
  200k→50k. Unlike width (which *raised* R²), depth *lowers* it — depth is where Gaussian-closure drift
  and inter-layer correlations compound (§1, §8.3). **At competition depth 32: R²=0.951** (50k nets,
  5.3k params). Confounded with the N drop (4× less data at K=32) → partly data-limited; untested
  whether more data recovers it. *Remaining gaps: the JOINT competition geometry (width 256 AND depth
  32) is untested — storage is `N·K·d²`, this is where Monarch/more-data enters; the two 1D sweeps
  bracket it ~0.95–0.99, depth being the limiting axis. And R² is log-space, not the final-layer MSE.*

**The ladder to climb (increasing dynamical-systems structure):**
1. **Exhaust the generic recurrent model** — scale data (1–2M), bigger/deeper/
   bidirectional, tune. Find its ceiling; nobody has.
2. **Neuron-permutation equivariance — now the critical path** (the width-32 wall above). The
   `d` hidden neurons are exchangeable, so `F` is *equivariant* to relabeling them — a
   per-boundary gauge symmetry coupling `W_l`'s rows to `W_{l+1}`'s columns (weight-space /
   NFN-style symmetry). §8.2/§8.4 already fixed the feature space: **`flatten(W_l)` is the wrong
   basis** — rotation-invariant magnitude is **inert** (the spectrum by the §8.2 argument;
   empirically the column norm, corr 0.00, §8.3); the signal is **basis-aligned** (singular
   *vectors* / per-neuron column alignment with the propagated moments).
   So encode each layer with a permutation-**equivariant**, basis-aligned per-neuron encoder
   (Deep Sets / attention over neurons) — *not* a flatten, *not* a rotation-invariant or
   permutation-*invariant* pool.
3. **Structured moment / particle recurrence** — a hidden state that *is* the
   activation moments, or `m` learned latent "particles" propagated through the
   *real* `W_l` (a learned, depth-adaptive UT that uses `W` directly). The
   dynamical-systems-native model; a clean result here is interesting regardless
   of the leaderboard. (Feeding UT moments as features, above, is the cheap proxy for this.)

### 10.1 The equivariant model, in linear-algebra form (the concrete Rung-1/2 build)

ReLU commutes with permutation matrices but not general orthogonal ones (§8.2), so
the only residual symmetry after fixing the standard basis is `W_l ↦ P_l W_l P_{l-1}ᵀ`
(P permutation), under which `μ_l ↦ P_l μ_l`. (Full orthogonal invariance would leave
only the singular values `Σ` — inert, §8.3. We keep the basis-aligned part: the singular
*vectors'* standard-basis coordinates.) Build a model equivariant to exactly that:

- **State `H ∈ R^{d×c}`** — a matrix co-transforming with the activation vector
  (`H ↦ P_l H`); its `c` columns are `c` probe-vectors in the layer's coordinate space.
- **Transition = the real weights as the step-`l` operator** (no flattening): propagate
  `M1 = W_l H` (mean) and `M2 = W_l^{∘2} H` (variance; `∘2` = entrywise square), then a
  **shared row-wise cell** (a GRU/LSTMCell applied to each row of `[M1 | M2 | UT_l]`, tied
  across rows) → `H_l`. A Deep-Sets column-mean `g = 1_dᵀX/d` (row-permutation-invariant,
  = the mean-field order parameters `m_l,q_l`) is broadcast back as `1_d g`.
- **Readout `F̂ = H_K w`** (`w ∈ R^c`).

Equivariance is one line (P orthogonal, `PᵀP=I`): `W_l H ↦ (P_l W_l P_{l-1}ᵀ)(P_{l-1}H) =
P_l(W_l H)`; row-maps and column-means commute with P. **Cost `O(K d² c)`** (the `W_l H`
products — the net's own forward-pass order); **learned params `O(c²)`, INDEPENDENT of width
`d`** — so it should not hit the flat-LSTM's `d²`-token data wall (the width-32 stall, §10).
It is the LSTM with its vector state promoted to the matrix `H` and its learned input
projection replaced by left-multiplication by `W_l`. The columns of `H` are *learned probe
vectors* (UT's fixed sigma points made learnable); the hybrid `G(W,UT)` enters by stacking
the per-coordinate UT moments `UT_l` into the row inputs. Custom cell — a NEW model class,
not a `make_core` swap in `rnn_harness.py`. Monarch composes on `c` or the row-mixing.

### 10.2 The Deep-Sets pool, in linear algebra (not an add-on — the other allowed operator)

Which linear maps on the row index commute with `X ↦ PX` (P permutation)? The commutant of
the `S_d` permutation representation on `R^d` is only **2-dimensional: `span{I, 11ᵀ}`** (Schur:
`R^d` = trivial irrep `span(1)` ⊕ standard irrep `1^⊥`, so an equivariant operator is one scalar
per block, `aI + b·11ᵀ`). Hence the most general equivariant linear layer is
`L(X) = XΛ + (1/d)11ᵀX·Γ + 1βᵀ` — "mix each row identically" (`I`) **plus** "mean-pool rows and
broadcast back" (`11ᵀ`). **The pool IS the second allowed operator; omitting it discards half the
equivariant linear space.**

`P_sym = 11ᵀ/d` is the rank-1 orthogonal projector onto `span(1)`; the split `R^d = span(1) ⊕ 1^⊥`
simultaneously diagonalizes every equivariant operator (eigenvalue `a+bd` on `1`, `a` on `1^⊥`).
The two blocks are exactly the two physical scales of §8.1: **`P_sym H` = the mean-field order
parameters** (`m_l = 1ᵀμ_l/d`; with a 2nd-moment pool, `q_l = ‖z_l‖²/d`, recursion
`q_{l+1}=½σ²q_l`, `m_{l+1}=√(σ²q_l/2π)`), and **`(I−P_sym)H` = the fluctuation** where the
realization signal lives (`F = 0.564 universal + O(1/√d) fluctuation`). So: **pool = order-parameter
channel, rows = fluctuation channel.** Linear-only gives just the mean; include a 2nd-moment pool
`1ᵀX^{∘2}/d` (the `q` order parameter, don't make the nonlinearity reconstruct it) and let the
row-cell `Φ` build higher invariants. Self-attention among rows = the same object with the fixed
`11ᵀ/d` replaced by a learned input-dependent row-coupling; likely overkill here since §8.1 says the
global coupling runs through order parameters (symmetric-subspace scalars), not pairwise structure.
Default was **mean + 2nd-moment pool** — but the §10 sweep **overturned this**: mean-pool ALONE won
(2nd-moment pool neutral for the residual cell, harmful for gru; attention worse). The row-cell
reconstructs `q_l` from the mean channel, so forcing it as an explicit feature only adds noise —
the §8.1 "`q_l` must be an explicit feature" intuition was wrong. **Use mean-pool.**

**Status.** TeamChaotic ≈ #82 with plain randomized UT; leaderboard top ≈3e-8 is
pure quadrature at the 0.1-pinned multiplier — the boring frontier. The bet here
is scientific (dynamical systems), not leaderboard-chasing.
