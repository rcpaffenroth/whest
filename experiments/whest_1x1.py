"""White-box estimation, 1x1 case: a depth-d chain of scalar ReLU units.

Each layer is   z = ReLU(w * x),   with w a scalar and x ~ N(0, 1) at the input.
Goal: predict E[z] after every layer analytically, and check against Monte Carlo.

Key point for the challenge: with width 1 there are NO cross-neuron correlations,
so "mean propagation" and "covariance propagation" are the *same* algorithm here
(covariance is just the scalar variance). The only source of error is the
modeling assumption that each pre-activation is Gaussian -- exact at layer 1
(w*x is Gaussian), approximate afterwards (the input to layer 2 is a *rectified*
Gaussian, not a Gaussian). This file isolates that single assumption.
"""

import numpy as np
from scipy.stats import norm

# Note on the seed: in the width-1 case the signal is >= 0 after the first ReLU,
# so a single negative weight downstream makes the pre-activation <= 0 and the
# chain dies to *exactly* 0 forever. That is generic for random 1D chains. Seed
# 56 is picked so w[1:] are all positive and the chain stays alive -- otherwise
# there is nothing to watch propagate. (Try seed 0 to see the dead chain.)
rng = np.random.default_rng(56)

depth = 8
scale = np.sqrt(2.0 / 1)                     # He init for width 1:  w ~ N(0, 2/width)
w = rng.standard_normal(depth) * scale       # (depth,)  one scalar weight per layer

# --- Analytic mean/variance propagation ---------------------------------------
# Carry the mean mu and variance var of the activation entering each layer.
# Input distribution:  x ~ N(0, 1).
mu, var = 0.0, 1.0
analytic = []
for wl in w:
    mu_pre = wl * mu                         # E[w x]      = w mu
    var_pre = wl**2 * var                    # Var[w x]    = w^2 var
    sigma = np.sqrt(var_pre)
    a = mu_pre / sigma                        # standardized threshold alpha = mu/sigma

    # Exact moments of z = ReLU(pre) for pre ~ N(mu_pre, sigma^2):
    #   E[z]   = mu_pre Phi(a) + sigma phi(a)
    #   E[z^2] = (mu_pre^2 + var_pre) Phi(a) + mu_pre sigma phi(a)
    Ez = mu_pre * norm.cdf(a) + sigma * norm.pdf(a)
    Ez2 = (mu_pre**2 + var_pre) * norm.cdf(a) + mu_pre * sigma * norm.pdf(a)

    mu, var = Ez, Ez2 - Ez**2                 # post-ReLU moments feed the next layer
    analytic.append(mu)

# --- Monte Carlo ground truth -------------------------------------------------
n = 1_000_000
x = rng.standard_normal(n)                    # (n,)  samples of x ~ N(0, 1)
mc = []
for wl in w:
    x = np.maximum(wl * x, 0.0)               # z = ReLU(w x), elementwise over samples
    mc.append(x.mean())

# --- Compare ------------------------------------------------------------------
print(f"{'layer':>5} {'analytic E[z]':>15} {'MC E[z]':>12} {'abs err':>10}")
for layer, (a_, m_) in enumerate(zip(analytic, mc)):
    print(f"{layer:>5} {a_:>15.6f} {m_:>12.6f} {abs(a_ - m_):>10.6f}")
