"""Sampling-vs-analytic harness for white-box estimation of ReLU-MLP activations.

We estimate the per-neuron mean activation of the FINAL layer (the scored
quantity) under x ~ N(0, I), and compare, at matched compute:

  analytic (k-independent):
    mean_prop        -- diagonal-variance moment propagation
    cov_prop         -- full-covariance moment propagation (gain-rule ReLU update)

  sampling (k forward passes each -- this is the matched-compute axis):
    random_mc        -- plain Monte Carlo, the baseline to beat
    antithetic_mc    -- pair each x with -x (kills the odd part of the integrand)
    cubature_ut      -- degree-3 spherical-radial cubature (the unscented point set
                        for N(0,I): +-sqrt(n) e_i), RANDOMLY ROTATED and used as a
                        fixed deterministic sample set propagated end-to-end
    sobol_qmc        -- scrambled Sobol quasi-Monte-Carlo, mapped to N(0,I)
    control_variate  -- random_mc corrected by the mean-field-linearized network
                        (soft-masked by cov_prop's activation probabilities Phi(alpha),
                        which is exactly E[ReLU']); the control has mean 0 by
                        construction, so this is unbiased and removes the linear-
                        response variance.

All sampling methods reduce to "pick k points, propagate, average the last layer";
only the point set (and, for the control variate, a per-neuron regression) differs.
The reported number is final-layer MSE against a large-N Monte-Carlo ground truth,
averaged over several random MLPs and a few repeats.

Set WIDTH = 256 (keep DEPTH = 32) for the competition shape; 32x32 here is a fast
proxy that already exercises the cross-width behavior.
"""

import numpy as np
from scipy.stats import norm, qmc

# --- experiment configuration -------------------------------------------------
# Competition shape. Fast 32x32 proxy values (see previous runs):
#   WIDTH=32, N_MLPS=8, REPEATS=4, K_SWEEP=(256,512,1024,2048,4096,8192), GT_SAMPLES=2_000_000
WIDTH = 256
DEPTH = 32
N_MLPS = 4                                    # average MSE over this many random MLPs
REPEATS = 2                                   # repeats per (MLP, k) to smooth sampler noise
K_SWEEP = (512, 1024, 2048, 4096, 8192)       # forward-pass budgets (>=2*width for cubature)
# Ground truth must be far more accurate than the best method or its own MC noise
# (~Var/GT_SAMPLES) becomes the floor every method bottoms out against. Keep the
# largest K_SWEEP entry well below GT_SAMPLES.
GT_SAMPLES = 500_000                           # Monte-Carlo samples for ground truth
GT_CHUNK = 100_000                             # chunk size so ground truth fits in memory


def build_mlp(width, depth, seed):
    """List of `depth` He-initialized weight matrices W ~ N(0, 2/width), each (width, width)."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / width)
    return [(rng.standard_normal((width, width)) * scale).astype(np.float32) for _ in range(depth)]


def forward_final_mean(W, X):
    """Propagate inputs X (k, width) through the ReLU MLP; return final-layer mean (width,)."""
    h = X
    for w in W:
        h = np.maximum(h @ w, 0.0)           # (k, width)  z = ReLU(W^T x), batched
    return h.mean(axis=0, dtype=np.float64)  # per-neuron mean over the k samples


def ground_truth(W, width, n, rng):
    """Large-N Monte-Carlo final-layer mean, accumulated in chunks to bound memory."""
    total = np.zeros(width)
    done = 0
    while done < n:
        b = min(GT_CHUNK, n - done)
        h = rng.standard_normal((b, width)).astype(np.float32)
        for w in W:
            h = np.maximum(h @ w, 0.0)
        total += h.sum(axis=0, dtype=np.float64)
        done += b
    return total / n


# --- analytic moment propagation ---------------------------------------------
def relu_moments(mu, var):
    """Exact E[z], Var[z], gain Phi(alpha) for z = ReLU(pre), pre ~ N(mu, var); per neuron."""
    sigma = np.sqrt(var)
    a = mu / sigma
    Ez = mu * norm.cdf(a) + sigma * norm.pdf(a)
    Ez2 = (mu * mu + var) * norm.cdf(a) + mu * sigma * norm.pdf(a)
    return Ez, Ez2 - Ez * Ez, norm.cdf(a)


def mean_prop(W, width):
    mu, var = np.zeros(width), np.ones(width)
    for w in W:
        mu, var, _ = relu_moments(w.T @ mu, (w * w).T @ var)
    return mu                                 # final-layer mean


def cov_prop(W, width):
    mu, cov = np.zeros(width), np.eye(width)
    gains = []                                # per-layer Phi(alpha) = E[ReLU'] mean-field gate
    for w in W:
        cov_pre = w.T @ cov @ w
        mu, var_post, g = relu_moments(w.T @ mu, np.diag(cov_pre))
        cov = np.outer(g, g) * cov_pre
        np.fill_diagonal(cov, var_post)
        gains.append(g)
    return mu, gains                          # final-layer mean + gates used by the control variate


# --- point sets for the sampling estimators (each returns X of shape (~k, width)) ---
def mc_points(rng, k, width):
    return rng.standard_normal((k, width)).astype(np.float32)


def antithetic_points(rng, k, width):
    half = rng.standard_normal((k // 2, width)).astype(np.float32)
    return np.concatenate([half, -half], axis=0)          # sample mean is exactly 0


def cubature_points(rng, k, width):
    # Degree-3 spherical-radial cubature nodes for N(0,I): +-sqrt(width) * e_i, equal weight.
    # This set reproduces mean 0 and covariance I exactly. We randomly rotate it (Haar O(n))
    # and stack enough rotations to fill the k budget -- randomized cubature.
    reps = max(1, k // (2 * width))
    blocks = []
    for _ in range(reps):
        Q, _ = np.linalg.qr(rng.standard_normal((width, width)))   # columns orthonormal
        cols = np.sqrt(width) * Q.T                                # rows = +-directions
        blocks += [cols, -cols]
    return np.concatenate(blocks, axis=0).astype(np.float32)


def sobol_points(rng, k, width):
    engine = qmc.Sobol(d=width, scramble=True, seed=int(rng.integers(2**31)))
    u = np.clip(engine.random(k), 1e-9, 1 - 1e-9)         # (k, width) in (0,1), avoid +-inf
    return norm.ppf(u).astype(np.float32)                 # inverse-CDF map to N(0, I)


def control_variate(W, X, gains):
    """random_mc on X, corrected by the mean-field-linearized network as a control.

    Control per sample:  Z = linearized-net(X), where each ReLU is replaced by its
    expected derivative Phi(alpha) (a soft gate). Z is linear in X, so E[Z] = 0 exactly.
    Per neuron j:  pred_j = mean(Y_j) - beta_j * mean(Z_j),  beta_j = Cov(Y_j,Z_j)/Var(Z_j).
    """
    h = X
    for w in W:
        h = np.maximum(h @ w, 0.0)
    Y = h                                                 # (k, width) true final activations

    z = X
    for w, g in zip(W, gains):
        z = (z @ w) * g                                   # (k, width) mean-field linear response
    Z = z

    Yc, Zc = Y - Y.mean(0), Z - Z.mean(0)
    beta = (Yc * Zc).mean(0) / np.maximum((Zc * Zc).mean(0), 1e-30)   # per-neuron optimal coeff
    return Y.mean(0, dtype=np.float64) - beta * Z.mean(0, dtype=np.float64)


# --- run the comparison -------------------------------------------------------
samplers = {
    "random_mc": mc_points,
    "antithetic_mc": antithetic_points,
    "cubature_ut": cubature_points,
    "sobol_qmc": sobol_points,
}
sampling_mse = {name: {k: 0.0 for k in K_SWEEP} for name in [*samplers, "control_variate"]}
analytic_mse = {"mean_prop": 0.0, "cov_prop": 0.0}

for mlp_seed in range(N_MLPS):
    W = build_mlp(WIDTH, DEPTH, seed=mlp_seed)
    gt = ground_truth(W, WIDTH, GT_SAMPLES, rng=np.random.default_rng(1000 + mlp_seed))

    analytic_mse["mean_prop"] += np.mean((mean_prop(W, WIDTH) - gt) ** 2)
    cov_pred, gains = cov_prop(W, WIDTH)
    analytic_mse["cov_prop"] += np.mean((cov_pred - gt) ** 2)

    for k in K_SWEEP:
        for r in range(REPEATS):
            rng = np.random.default_rng((mlp_seed, k, r))         # distinct per repeat
            for name, sampler in samplers.items():
                pred = forward_final_mean(W, sampler(rng, k, WIDTH))
                sampling_mse[name][k] += np.mean((pred - gt) ** 2)
            cv_pred = control_variate(W, mc_points(rng, k, WIDTH), gains)
            sampling_mse["control_variate"][k] += np.mean((cv_pred - gt) ** 2)

# average over MLPs (and repeats for the samplers)
for name in sampling_mse:
    for k in K_SWEEP:
        sampling_mse[name][k] /= N_MLPS * REPEATS
for name in analytic_mse:
    analytic_mse[name] /= N_MLPS

# --- report -------------------------------------------------------------------
print(f"width={WIDTH} depth={DEPTH}, final-layer MSE (avg over {N_MLPS} MLPs x {REPEATS} repeats)")
print("k = number of forward passes = matched-compute axis\n")
header = f"{'method':>16} " + " ".join(f"{'k='+str(k):>11}" for k in K_SWEEP)
print(header)
print("-" * len(header))
for name in [*samplers, "control_variate"]:
    print(f"{name:>16} " + " ".join(f"{sampling_mse[name][k]:>11.3e}" for k in K_SWEEP))
# Analytic methods are not free: place them on the forward-pass axis for a fair
# compute comparison. mean_prop ~ O(depth*width^2) ~ 2 forward passes; cov_prop
# ~ O(depth*width^3) ~ 2*width forward passes.
analytic_fwd = {"mean_prop": 2, "cov_prop": 2 * WIDTH}
print("\n--- analytic (k-independent; shown with forward-pass-equivalent cost) ---")
for name in analytic_mse:
    print(f"{name:>16} {analytic_mse[name]:>11.3e}   (~{analytic_fwd[name]} fwd-pass equiv)")
