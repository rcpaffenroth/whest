"""Unscented-transform (UT) estimator for the mean activations of a ReLU MLP.

We want  E[z_l]  -- the per-neuron mean activation at every layer l -- of
    z_0 = x,   z_{l+1} = ReLU(W_l^T z_l),   with input  x ~ N(0, I_width).

Idea (the unscented transform): instead of averaging thousands of RANDOM inputs
(Monte Carlo), choose a small DETERMINISTIC set of points -- the "sigma points" --
whose sample mean and covariance exactly equal those of the input N(0, I). Push
those points through the network and average. Because the points reproduce the
first two moments exactly, their average estimates E[ReLU(...)] to higher order
than random sampling, so far fewer points are needed for the same accuracy.

For N(0, I_n) the standard symmetric sigma-point set is
    +- sqrt(n) * e_i      (i = 1..n),   all with equal weight 1/(2n).
These 2n points have sample mean 0 and sample covariance I exactly (this is the
degree-3 spherical-radial cubature rule). We additionally apply a random rotation
and average several rotations: the rotation preserves the exact moment matching
while averaging away the directional bias of the fixed coordinate axes -- the
"randomized" UT. More rotations -> more points -> lower error.

Run at any size, e.g.:
    uv run python experiments/whest_ut.py --width 8 --depth 8 --rotations 8
    uv run python experiments/whest_ut.py --width 256 --depth 32 --rotations 13
"""

import argparse

import numpy as np


def build_mlp(width, depth, seed):
    """`depth` He-initialized weight matrices W ~ N(0, 2/width), each (width, width)."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / width)
    return [(rng.standard_normal((width, width)) * scale).astype(np.float32) for _ in range(depth)]


def sigma_points(width, rotations, rng):
    """Randomized UT sigma points for N(0, I_width): the +- sqrt(width) e_i set,
    each copy turned by a Haar-random rotation. Shape (2 * width * rotations, width)."""
    blocks = []
    for _ in range(rotations):
        Q, _ = np.linalg.qr(rng.standard_normal((width, width)))   # Q: orthonormal columns
        axes = np.sqrt(width) * Q.T                                # rows: orthonormal directions, radius sqrt(width)
        blocks += [axes, -axes]                                    # the +- symmetric set
    return np.concatenate(blocks, axis=0).astype(np.float32)


def unscented_means(W, X):
    """Propagate points X through the MLP; return per-layer means (depth, width).
    Equal-weight sigma points, so the UT expectation is just the plain average."""
    h = X
    means = []
    for w in W:
        h = np.maximum(h @ w, 0.0)          # z = ReLU(W^T x), batched over all sigma points
        means.append(h.mean(axis=0))        # UT estimate of E[z] at this layer
    return np.stack(means, axis=0)


def monte_carlo_means(W, width, n, rng):
    """Ground truth: average n random N(0, I) inputs, per layer. Shape (depth, width)."""
    h = rng.standard_normal((n, width)).astype(np.float32)
    means = []
    for w in W:
        h = np.maximum(h @ w, 0.0)
        means.append(h.mean(axis=0, dtype=np.float64))
    return np.stack(means, axis=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--rotations", type=int, default=13, help="UT uses 2 * width * rotations points")
    p.add_argument("--mc-samples", type=int, default=1_000_000, help="Monte-Carlo ground-truth samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    W = build_mlp(args.width, args.depth, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    X = sigma_points(args.width, args.rotations, rng)
    ut = unscented_means(W, X)
    gt = monte_carlo_means(W, args.width, args.mc_samples, rng)

    per_layer_mse = ((ut - gt) ** 2).mean(axis=1)        # (depth,)
    print(f"width={args.width} depth={args.depth}  "
          f"UT points={X.shape[0]} (= 2 * {args.width} * {args.rotations})  "
          f"MC samples={args.mc_samples:,}\n")
    print(f"{'layer':>5} {'UT mean[0]':>11} {'MC mean[0]':>11} {'layer MSE':>11}")
    for layer in range(args.depth):
        print(f"{layer:>5} {ut[layer, 0]:>11.6f} {gt[layer, 0]:>11.6f} {per_layer_mse[layer]:>11.3e}")
    print(f"\nfinal-layer MSE (the scored quantity): {per_layer_mse[-1]:.3e}")
