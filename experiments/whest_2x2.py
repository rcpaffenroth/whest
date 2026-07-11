"""White-box estimation, 2x2 case: a depth-d MLP with 2 neurons per layer.

Each layer is   z = ReLU(W^T x),   with W a 2x2 matrix and x ~ N(0, I_2) at input.
Goal: predict E[z] per neuron after every layer two ways, and check against MC:

  * mean propagation  -- carry per-neuron mean + DIAGONAL variance, i.e. pretend
                         the two neurons stay independent.
  * covariance propagation -- carry the full 2x2 covariance, so the cross-neuron
                         correlations that dense W + ReLU create are kept.

Width 2 is the smallest case where correlations exist, so it is exactly where the
two methods diverge: the recorded means agree at layer 1 (input cov is I, so the
diagonals match) and split from layer 2 on, once off-diagonal covariance builds.
Compare the two columns against MC to see what mean propagation throws away.
"""

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)

depth = 8
width = 2
scale = np.sqrt(2.0 / width)                       # He init:  W_ij ~ N(0, 2/width)
W = rng.standard_normal((depth, width, width)) * scale     # (depth, 2, 2)


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
mu = np.zeros(width)                                # (2,)   E[x] = 0
var = np.ones(width)                               # (2,)   diagonal of Cov[x] = I
mean_prop = []
for Wl in W:
    mu_pre = Wl.T @ mu                             # (2,)   E[W^T x] = W^T mu
    var_pre = (Wl**2).T @ var                      # (2,)   diag of W^T diag(var) W
    mu, var, _ = relu_moments(mu_pre, var_pre)
    mean_prop.append(mu)

# --- Covariance propagation (full 2x2) ----------------------------------------
mu = np.zeros(width)                                # (2,)
cov = np.eye(width)                                # (2, 2)   Cov[x] = I
cov_prop = []
for Wl in W:
    mu_pre = Wl.T @ mu                             # (2,)     E[W^T x] = W^T mu
    cov_pre = Wl.T @ cov @ Wl                      # (2, 2)   Cov[W^T x] = W^T Cov W  (exact)
    var_pre = np.diag(cov_pre)                     # (2,)
    mu, var_post, gain = relu_moments(mu_pre, var_pre)

    # Post-ReLU covariance (approx): scale off-diagonals by the per-neuron gains
    # Phi(alpha_i) Phi(alpha_j), then overwrite the diagonal with the exact
    # marginal variance from the ReLU moment formula.
    cov = np.outer(gain, gain) * cov_pre           # (2, 2)   approximate off-diagonals
    np.fill_diagonal(cov, var_post)                # exact diagonal
    cov_prop.append(mu)

# --- Monte Carlo ground truth -------------------------------------------------
n = 1_000_000
x = rng.standard_normal((n, width))                # (n, 2)  samples of x ~ N(0, I)
mc = []
for Wl in W:
    x = np.maximum(x @ Wl, 0.0)                    # (n, 2)  z = ReLU(W^T x), batched
    mc.append(x.mean(axis=0))

# --- Compare (per neuron, each layer) -----------------------------------------
print(f"{'layer':>5} {'neuron':>7} {'mean_prop':>11} {'cov_prop':>11} {'MC':>11}")
for layer in range(depth):
    for i in range(width):
        print(f"{layer:>5} {i:>7} {mean_prop[layer][i]:>11.6f} "
              f"{cov_prop[layer][i]:>11.6f} {mc[layer][i]:>11.6f}")
