"""Validation: is cov_prop's ONLY real error its off-diagonal ReLU-covariance
closure (the gain rule)?  If so, an exact closure should recover the accuracy.

We propagate the activation moments (mean mu, covariance Sigma) layer by layer:
    pre-activation   h = W^T z:   mu_h = W^T mu,   Sigma_h = W^T Sigma W
    post-ReLU        z' = ReLU(h)
and compare three ways of turning (mu_h, Sigma_h) into (mu', Sigma'):

  mean_prop : diagonal only (ignore off-diagonal Sigma_h) + exact scalar ReLU moments
  cov_prop  : full Sigma_h, EXACT diagonal, gain-rule off-diagonal  Phi(a_i)Phi(a_j)Sigma_h_ij
  adf_exact : full Sigma_h, post-ReLU moments by sampling N(mu_h, Sigma_h) -- the exact
              Gaussian closure (up to MC noise), i.e. cov_prop with a perfect covariance rule

Report final-layer mean MSE vs true MC means over several real whestbench MLPs.
"""
import numpy as np
from scipy.stats import norm
import whestbench as wb

WIDTH, DEPTH = 256, 32
SEEDS = [0, 1, 2]
N_GT = 500_000
S_ADF = 20_000            # samples for the exact-closure step
EPS = 1e-12

Phi, phi = norm.cdf, norm.pdf
R = lambda a: a * Phi(a) + phi(a)                 # E[ReLU(N(a,1))]
E2 = lambda a: (1 + a * a) * Phi(a) + a * phi(a)  # E[ReLU(N(a,1))^2]


def relu_moments_diag(mu_h, var_h):
    """Exact per-neuron ReLU mean & variance from pre-activation mean & variance."""
    s = np.sqrt(np.maximum(var_h, EPS))
    a = mu_h / s
    mu = s * R(a)
    var = s * s * E2(a) - mu * mu
    return mu, np.maximum(var, 0.0), a, s


def prop_mean(W):
    mu = np.zeros(WIDTH); v = np.ones(WIDTH)                 # z_0 = x ~ N(0, I)
    for w in W:
        mu_h = w.T @ mu
        var_h = (w * w).T @ v                                # diagonal / independence
        mu, v, _, _ = relu_moments_diag(mu_h, var_h)
    return mu


def prop_cov(W):
    mu = np.zeros(WIDTH); Sig = np.eye(WIDTH)
    for w in W:
        mu_h = w.T @ mu
        Sig_h = w.T @ Sig @ w
        var_h = np.diag(Sig_h)
        mu, var, a, s = relu_moments_diag(mu_h, var_h)
        gain = Phi(a)                                        # gain-rule factor per neuron
        Sig = np.outer(gain, gain) * Sig_h                  # off-diagonal gain rule
        np.fill_diagonal(Sig, var)                          # exact diagonal
    return mu


def prop_adf(W, rng):
    mu = np.zeros(WIDTH); Sig = np.eye(WIDTH)
    for w in W:
        mu_h = w.T @ mu
        Sig_h = w.T @ Sig @ w
        # exact Gaussian closure: sample N(mu_h, Sig_h), apply ReLU, recompute moments
        vals, vecs = np.linalg.eigh(0.5 * (Sig_h + Sig_h.T))
        L = vecs * np.sqrt(np.maximum(vals, 0.0))            # PSD square root
        g = mu_h[:, None] + L @ rng.standard_normal((WIDTH, S_ADF))
        z = np.maximum(g, 0.0)
        mu = z.mean(1)
        Sig = np.cov(z, bias=True)
    return mu


print(f"whestbench {WIDTH}x{DEPTH}  GT N={N_GT:,}  ADF closure S={S_ADF:,}\n")
print(f"{'seed':>4} {'mean_prop':>11} {'cov_prop':>11} {'adf_exact':>11}")
agg = {"mean_prop": [], "cov_prop": [], "adf_exact": []}
for seed in SEEDS:
    mlp = wb.sample_mlp(width=WIDTH, depth=DEPTH, seed=seed)
    W = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    truth = np.asarray(wb.sample_layer_statistics(mlp, N_GT)[0][-1], dtype=np.float64)
    rng = np.random.default_rng(100 + seed)
    mse = {
        "mean_prop": ((prop_mean(W) - truth) ** 2).mean(),
        "cov_prop": ((prop_cov(W) - truth) ** 2).mean(),
        "adf_exact": ((prop_adf(W, rng) - truth) ** 2).mean(),
    }
    for k, v in mse.items():
        agg[k].append(v)
    print(f"{seed:>4} {mse['mean_prop']:>11.3e} {mse['cov_prop']:>11.3e} {mse['adf_exact']:>11.3e}")

print("\nmean over MLPs:")
for k in agg:
    print(f"  {k:11s} {np.mean(agg[k]):.3e}")
print("\nreference: UT submission ~4-5e-6")
