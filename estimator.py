"""Your estimator. Edit `predict()`. Run `python estimator.py` to iterate.

Stage 1 of the WhestBench ladder: just `flopscope` and the local engine. No CLI
knowledge required. Once `predict()` returns something interesting, climb to
Stage 2: `whest validate --estimator estimator.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# Fraction of the FLOP budget to spend. The score multiplier is max(0.1, C/B),
# so it floors at 0.1 for any usage <= 10% of budget: within that band more
# points strictly lowers MSE at no multiplier cost, so we push toward the floor,
# leaving a small margin for residual/QR overhead. (Even slightly over 10% is
# only a mild, linear multiplier penalty -- not the catastrophic budget-exceed.)
TARGET_FRACTION = 0.095


class Estimator(BaseEstimator):
    """Randomized unscented-transform (degree-3 spherical-radial cubature) estimator.

    Represents the input N(0, I) by the deterministic sigma-point set
    +- sqrt(width) e_i (which matches its mean and covariance exactly), turned by
    several Haar-random rotations, then propagates those points through the MLP and
    averages per layer. See experiments/whest_ut.py for the plain-numpy exposition.
    """

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        rng = fnp.random.default_rng(mlp.seed)

        # Cost is dominated by the batched matmuls: 2*width points per rotation,
        # each pushed through depth (width x width) layers. One (P, width)@(width,
        # width) matmul is ~2*P*width^2 FLOPs, so one rotation (P = 2*width) costs
        # ~4*depth*width^3. Take as many rotations as the budget fraction allows.
        cost_per_rotation = 4 * depth * width**3
        rotations = max(1, int(TARGET_FRACTION * budget / cost_per_rotation))

        # --- randomized UT sigma points for N(0, I): +- sqrt(width) e_i, rotated ---
        radius = math.sqrt(width)
        blocks = []
        for _ in range(rotations):
            Q, _ = fnp.linalg.qr(rng.standard_normal((width, width)))  # Q: orthonormal columns
            axes = radius * Q.T                                        # rows: directions at radius sqrt(width)
            blocks += [axes, -axes]                                    # the +- symmetric set
        X = fnp.concatenate(blocks, axis=0)                           # (2*width*rotations, width)

        # --- propagate the sigma points and average each layer (equal weights) ---
        h = X
        means = []
        for w in mlp.weights:
            h = fnp.maximum(h @ w, 0.0)          # z = ReLU(W^T x), batched over sigma points
            means.append(h.mean(axis=0))         # UT estimate of E[z] at this layer
        return fnp.stack(means, axis=0)          # (depth, width)


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)  # phase-1 competition shape (warmup was 8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
