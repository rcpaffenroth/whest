"""Corrector v2: learned per-neuron correction of UT's deterministic bias.

UT's single-radius rule leaves a weight-determined degree->=4 radial bias (~1/3 of
its MSE per the probe). We test whether a model on cheap, inference-available
per-neuron features can predict the truth better than UT's raw output -- trained on
one set of MLPs and evaluated on DISJOINT held-out MLPs (no leakage).

Per output neuron j, features from the (budget) UT penultimate cloud + last weights:
    UT_j      : UT's raw final-layer average          (the thing we correct)
    align_j   : <W[:,j], mu_prev>                      (the dominant statistic)
    sigma_j   : sqrt(W[:,j]^T Sigma_prev W[:,j])       (full pre-activation std)
    alpha_j   : align_j / sigma_j
    Psi_j     : sigma_j * R(alpha_j)                   (analytic rectification)
Model: ridge regression on a quadratic feature map (flexible, closed-form, no deps).
Target: true final-layer mean.  Baseline to beat: UT_j itself.
"""
import numpy as np
from scipy.stats import norm
import whestbench as wb

WIDTH, DEPTH = 256, 32
TRAIN_SEEDS = list(range(10))
TEST_SEEDS = list(range(10, 16))
ROT_SEEDS = range(6)
ROTATIONS = 12
N_GT = 500_000
RIDGE = 1e-4

Phi, phi = norm.cdf, norm.pdf
R = lambda a: a * Phi(a) + phi(a)


def ut_cloud(W, rotations, rng):
    r = np.sqrt(WIDTH); blocks = []
    for _ in range(rotations):
        Q, _ = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
        blocks += [r * Q.T, -r * Q.T]
    h = np.concatenate(blocks, 0)
    for w in W:
        h = np.maximum(h @ w, 0.0)
    return h


def features(W, rotations, rng):
    cloud = ut_cloud(W[:-1], rotations, rng)
    Wl = W[-1]
    ut = np.maximum(cloud @ Wl, 0.0).mean(0)
    mu_prev = cloud.mean(0)
    align = Wl.T @ mu_prev
    Sig = np.cov(cloud, rowvar=False, bias=True)
    sigma = np.sqrt(np.maximum(np.einsum('ac,bc,ab->c', Wl, Wl, Sig), 1e-12))
    alpha = align / sigma
    psi = sigma * R(alpha)
    return ut, np.column_stack([ut, align, sigma, alpha, psi])


def quad_map(X):
    """[1, x_i, x_i^2] plus pairwise products -- a flexible closed-form basis."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    for i in range(d):
        for k in range(i, d):
            cols.append(X[:, i] * X[:, k])
    return np.column_stack(cols)


def gather(seeds):
    rows_X, rows_y, per_mlp = [], [], []
    for seed in seeds:
        mlp = wb.sample_mlp(width=WIDTH, depth=DEPTH, seed=seed)
        W = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
        truth = np.asarray(wb.sample_layer_statistics(mlp, N_GT)[0][-1], dtype=np.float64)
        for rs in ROT_SEEDS:
            rng = np.random.default_rng(1000 * seed + rs)
            ut, feat = features(W, ROTATIONS, rng)
            rows_X.append(feat); rows_y.append(truth)
            per_mlp.append((seed, rs, ut, feat, truth))
    return np.vstack(rows_X), np.concatenate(rows_y), per_mlp


print(f"whestbench {WIDTH}x{DEPTH}  rot={ROTATIONS}  train MLPs={TRAIN_SEEDS}  test MLPs={TEST_SEEDS}\n")
Xtr, ytr, _ = gather(TRAIN_SEEDS)
_, _, test = gather(TEST_SEEDS)

# standardize features, fit ridge on quadratic map
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
Ptr = quad_map((Xtr - mu) / sd)
A = Ptr.T @ Ptr + RIDGE * np.eye(Ptr.shape[1])
beta = np.linalg.solve(A, Ptr.T @ ytr)

ut_mses, model_mses = [], []
for seed, rs, ut, feat, truth in test:
    pred = quad_map((feat - mu) / sd) @ beta
    ut_mses.append(((ut - truth) ** 2).mean())
    model_mses.append(((pred - truth) ** 2).mean())

ut_m, mo_m = np.mean(ut_mses), np.mean(model_mses)
print(f"held-out UT      MSE: {ut_m:.3e}")
print(f"held-out model   MSE: {mo_m:.3e}")
print(f"improvement ratio   : {ut_m / mo_m:.2f}x  ({'better' if mo_m < ut_m else 'WORSE'})")
print(f"\nper test MLP (UT -> model):")
for seed in TEST_SEEDS:
    u = [m for (s, r, *_ ), m in zip(test, ut_mses) if s == seed]
    v = [m for (s, r, *_ ), m in zip(test, model_mses) if s == seed]
    print(f"  seed {seed}: {np.mean(u):.3e} -> {np.mean(v):.3e}  ({np.mean(u)/np.mean(v):.2f}x)")
