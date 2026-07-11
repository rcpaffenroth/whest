"""Latin-hypercube (randomized stratified / QMC-family) estimator.

Alternative submission to the unscented-transform estimator. Instead of the
symmetric sigma-point set, sample the input N(0, I) by a Latin hypercube: each of
the `width` input coordinates is stratified into `k` equal-probability bins and
visited exactly once, with a random offset inside each bin. This removes the
per-coordinate (main-effect) sampling variance that plain Monte Carlo carries,
at the same cost as MC (one forward pass per point) -- and, unlike the symmetric
UT, its accuracy keeps improving with more points rather than hitting a
degree-3 bias floor. See experiments/README.md for the budget comparison.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# See estimator.py: multiplier floors at 0.1 for usage <= 10% of budget, so we
# push toward the floor -- but leave margin for the LHS point-generation's
# residual wall-time (argsort/ppf), which pushes effective compute above the
# raw FLOP count. LHS spends one forward pass per point (no control-variate
# second pass -- that was measured to be a net loss under a FLOP budget).
TARGET_FRACTION = 0.09


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        rng = fnp.random.default_rng(mlp.seed)

        # One forward pass per point costs ~2*depth*width^2 FLOPs; take as many
        # points as the budget fraction allows.
        cost_per_point = 2 * depth * width**2
        k = max(1, int(TARGET_FRACTION * budget / cost_per_point))

        # --- Latin hypercube sample of N(0, I) (vectorized over coordinates) ---
        # For each coordinate (column): argsort-of-argsort of uniform noise gives
        # a random rank 0..k-1 per point -- a permutation, i.e. the LHS strata.
        # A uniform offset places the point inside its stratum; the inverse normal
        # CDF maps the stratified uniforms to Gaussians.
        noise = rng.random((k, width))
        strata = noise.argsort(axis=0).argsort(axis=0)        # (k, width) each column a permutation of 0..k-1
        u = (strata + rng.random((k, width))) / k             # stratified uniforms in (0,1)
        X = flops.stats.norm.ppf(u)                            # (k, width) inverse-CDF -> N(0,1)

        # --- propagate and average each layer ---
        h = X
        means = []
        for w in mlp.weights:
            h = fnp.maximum(h @ w, 0.0)          # z = ReLU(W^T x), batched over the LHS points
            means.append(h.mean(axis=0))         # per-layer mean estimate
        return fnp.stack(means, axis=0)          # (depth, width)
