"""White-box estimation, 8x8 case: a depth-d MLP with 8 neurons per layer.

Each layer is   z = ReLU(W^T x),   with W an 8x8 matrix and x ~ N(0, I_8) at input.
Same two estimators as the 2x2 file -- mean propagation (diagonal variance) and
covariance propagation (full 8x8 covariance) -- checked against Monte Carlo.

At width 8 there are too many neurons to eyeball (8 x depth cells), so instead of
printing every activation we report the per-layer MSE of each method against MC.
MSE against MC ground truth is exactly the challenge's metric, and the final row
(last layer) is the quantity the leaderboard actually scores.
"""

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)

depth = 8
width = 8
scale = np.sqrt(2.0 / width)                       # He init:  W_ij ~ N(0, 2/width)
W = rng.standard_normal((depth, width, width)) * scale     # (depth, 8, 8)


def relu_moments(mu_pre, var_pre):
    """Exact E[z], Var[z], and gain Phi(alpha) for z = ReLU(pre), pre ~ N(mu, var).

    All arguments/returns are per-neuron vectors of shape (width,).
    """
    sigma = np.sqrt(var_pre)
    a = mu_pre / sigma                              # alpha = mu/sigma, per neuron
    Ez = mu_pre * norm.cdf(a) + sigma * norm.pdf(a)
    Ez2 = (mu_pre**2 + var_pre) * norm.cdf(a) + mu_pre * sigma * norm.pdf(a)
    return Ez, Ez2 - Ez**2, norm.cdf(a)

# --- Mean propagation (diagonal variance) -------------------------------------
mu = np.zeros(width)                                # (8,)   E[x] = 0
var = np.ones(width)                               # (8,)   diagonal of Cov[x] = I
mean_prop = []
for Wl in W:
    mu_pre = Wl.T @ mu                             # (8,)   E[W^T x] = W^T mu
    var_pre = (Wl**2).T @ var                      # (8,)   diag of W^T diag(var) W
    mu, var, _ = relu_moments(mu_pre, var_pre)
    mean_prop.append(mu)

# --- Covariance propagation (full 8x8) ----------------------------------------
mu = np.zeros(width)                                # (8,)
cov = np.eye(width)                                # (8, 8)   Cov[x] = I
cov_prop = []
for Wl in W:
    mu_pre = Wl.T @ mu                             # (8,)     E[W^T x] = W^T mu
    cov_pre = Wl.T @ cov @ Wl                      # (8, 8)   Cov[W^T x] = W^T Cov W  (exact)
    var_pre = np.diag(cov_pre)                     # (8,)
    mu, var_post, gain = relu_moments(mu_pre, var_pre)

    # Post-ReLU covariance (approx): scale off-diagonals by the per-neuron gains
    # Phi(alpha_i) Phi(alpha_j), then overwrite the diagonal with the exact
    # marginal variance from the ReLU moment formula.
    cov = np.outer(gain, gain) * cov_pre           # (8, 8)   approximate off-diagonals
    np.fill_diagonal(cov, var_post)                # exact diagonal
    cov_prop.append(mu)

# --- Monte Carlo ground truth -------------------------------------------------
n = 1_000_000
x = rng.standard_normal((n, width))                # (n, 8)  samples of x ~ N(0, I)
mc = []
for Wl in W:
    x = np.maximum(x @ Wl, 0.0)                    # (n, 8)  z = ReLU(W^T x), batched
    mc.append(x.mean(axis=0))

# --- Compare: per-layer MSE of each method against MC -------------------------
mean_prop, cov_prop, mc = map(np.array, (mean_prop, cov_prop, mc))   # (depth, width) each
mse_mean = ((mean_prop - mc) ** 2).mean(axis=1)    # (depth,)  MSE over the 8 neurons
mse_cov = ((cov_prop - mc) ** 2).mean(axis=1)      # (depth,)

print(f"{'layer':>5} {'mse mean_prop':>14} {'mse cov_prop':>14} {'ratio':>8}")
for layer in range(depth):
    print(f"{layer:>5} {mse_mean[layer]:>14.3e} {mse_cov[layer]:>14.3e} "
          f"{mse_mean[layer] / mse_cov[layer]:>8.2f}")
print(f"\nfinal-layer MSE (the scored quantity):  "
      f"mean_prop = {mse_mean[-1]:.3e}   cov_prop = {mse_cov[-1]:.3e}")
