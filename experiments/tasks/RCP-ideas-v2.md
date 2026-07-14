I want to start fresh looking at the 

[ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026).
This folder holds our experiments; the repo root is a fork of the official
[starter kit](https://github.com/AIcrowd/whest-starterkit) (its `README.md` is
preserved).

---

## 0. The central insights

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

So, let's start with the following smaller problem.  \forall i, let W_i \in R^{2\times2}, but keep 
the number of layers k=32.  In this case W \in R^{32 \times 2 \times 2} and if we flatten W into a vector
is x = flat(W) \in R^128, since the output dimension is 2 in this case, we have that 
F: R^128 \rightarrow R^2.

Now, we need a bit more notation to makes things clear.  F is defined as above, but let's also consider the function "f".
f : R^2 \rightarrow R^2 define by f(x) = ReLU(W_k ReLU(W_{k-1} ... ReLU(W_1 x) ... )).
So, in rough language, f takes a vector x and maps it forward to a vector \hat{x}.  If we happen to choose many x~N(0,1) then we 
get many \hat{x} which can be thought of as a distribution whose mean we are interested in.  F is a deterministic function which
produces the limit of this mean when "all of the randomness is integrated out".

I want to proceed in two steps:

1) First, I want a small python code to play with f directly.  Generate random W_i in R^{2\times2} given a random seed, and let
me play around with evaluating f.  I am particularily interested in the "smoothness" of f. I realize that f is C^0 because of the ReLUs,
but I want to understand how senstive f is to pertubations in W_i. Why?  I suspect that the sensitivity of f to pertubnations in W_i will lead to sensitivy
in F that I want to understand.
2) Second, I want an MLP to approximate F.  Something like 128 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 2 to start with a leaky relu 
activation function, but all such hyperparameters should be easily changable.  Note, training data will need to be generated using something like the f in step 1, and I am
interested in first knowing whether F can be approximated at all, even using a large training corpus and a large MLP, and secondly how efficiently F can be approximated,
using a smaller training corpus and/or a smaller MLP. Now, training data for F will need to be generated by doing many MC samples of f.

Important last note, do not me misled by the previous work in the directory! I want you to laser focus on the the deterministic version of the problem.  
Sampling ideas are for another day, in particular whest_2x2.py it *not* what we want to do here.
