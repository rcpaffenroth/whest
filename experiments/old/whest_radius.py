"""Does rescaling the single UT shell radius remove the radial bias?

Finding from whest_bias_variance.py: UT has a substantial deterministic bias
(~1.9e-6, ~43% of operating-point MSE), and a large chunk of it is present
already at layer 0 -- where the pre-activation is *exactly* Gaussian. That
layer-0 bias cannot be Edgeworth; it is a pure radial-quadrature artifact.

Why: ReLU is positively homogeneous of degree 1, so ReLU(w^T x) = ||x|| *
ReLU(w^T x_hat). The randomized (Haar-averaged) UT puts every sigma point on the
shell ||x|| = sqrt(n), so as rotations -> infinity it computes
    UT_inf = sqrt(n) * E_dir[ReLU(w^T x_hat)],
while the truth is
    truth  = E||x|| * E_dir[ReLU(w^T x_hat)].
The directional integral is identical, so at layer 0 the entire bias is the
radial factor sqrt(n) / E||x|| ~ 1 + 1/(4n). The base UT matches E[rho^2] = n
(radius sqrt(n)), but the *mean* of a degree-1-homogeneous map is controlled by
the FIRST radial moment E[rho] = E||x||, not the second.

Fix under test (cheap, "useful not perfect"): keep ONE shell, move it to radius
alpha * sqrt(n). No extra points, no stolen rotations (unlike old/whest_two_radius,
which used two shells and had to halve rotations). alpha = E||x||/sqrt(n) zeroes
the layer-0 bias exactly; we also sweep alpha to find the final-layer optimum,
since the input rescaling propagates nonlinearly through depth.

Run:
    uv run python experiments/whest_radius.py --seeds 6 --mc-samples 2000000
"""

import argparse
import math

import numpy as np

from whest_ut import build_mlp, unscented_means, monte_carlo_means


def sigma_points_scaled(width, rotations, rng, radius):
    """Randomized UT sigma points on the shell of the given radius (frozen scheme)."""
    blocks = []
    for _ in range(rotations):
        Q, _ = np.linalg.qr(rng.standard_normal((width, width)))
        axes = radius * Q.T
        blocks += [axes, -axes]
    return np.concatenate(blocks, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--mc-samples", type=int, default=2_000_000)
    p.add_argument("--r-inf", type=int, default=256,
                   help="rotations used as the R->infinity proxy for UT_inf")
    args = p.parse_args()

    width, depth = args.width, args.depth
    n = width

    # E||x|| for x ~ N(0, I_n) = sqrt(2) * Gamma((n+1)/2) / Gamma(n/2).
    Ex_norm = math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))
    alpha_opt = Ex_norm / math.sqrt(n)   # radius scale that makes layer 0 exact
    print(f"n={n}  sqrt(n)={math.sqrt(n):.4f}  E||x||={Ex_norm:.4f}  "
          f"alpha_opt(layer0)={alpha_opt:.5f}\n")

    # Radius scales to sweep. Wider than alpha_opt because the input rescaling
    # propagates nonlinearly, so the final-layer optimum may drift.
    alphas = sorted({0.95, 0.97, 0.99, round(alpha_opt, 5), 1.00, 1.01, 1.02})

    bias_final = {a: [] for a in alphas}       # final-layer bias^2 per seed
    layer_bias = {a: [] for a in alphas}       # (depth,) per-layer bias^2 per seed

    for s in range(args.seeds):
        W = build_mlp(width, depth, seed=s)
        truth_all = monte_carlo_means(W, width, args.mc_samples,
                                      np.random.default_rng(10_000 + s))   # (depth, width)

        for a in alphas:
            # Same rng seed across alphas => identical rotations; only the radius differs.
            X = sigma_points_scaled(width, args.r_inf,
                                    np.random.default_rng(30_000 + s), a * math.sqrt(n))
            ut_all = unscented_means(W, X)                                 # (depth, width)
            bias_final[a].append(((ut_all[-1] - truth_all[-1]) ** 2).mean())
            layer_bias[a].append(((ut_all - truth_all) ** 2).mean(axis=1))

        print(f"seed {s}: "
              + "  ".join(f"a={a}:{np.mean(bias_final[a][-1]):.2e}" for a in alphas))

    # ---- aggregate ----
    print(f"\n=== final-layer bias floor vs radius scale (mean over {args.seeds} seeds) ===")
    print(f"{'alpha':>8} {'radius':>8} {'bias^2':>12}")
    base = np.mean(bias_final[1.00])
    best_a, best_v = None, math.inf
    for a in alphas:
        v = np.mean(bias_final[a])
        if v < best_v:
            best_a, best_v = a, v
        tag = "  <- base (sqrt n)" if a == 1.00 else ("  <- alpha_opt(layer0)" if a == round(alpha_opt, 5) else "")
        print(f"{a:>8} {a*math.sqrt(n):>8.3f} {v:>12.3e}{tag}")
    print(f"\nbase (alpha=1.00) bias floor: {base:.3e}")
    print(f"best alpha={best_a}: bias floor {best_v:.3e}  ({base/best_v:.2f}x lower, "
          f"{100*(1-best_v/base):.0f}% of the bias removed for free)")

    print(f"\n=== per-layer bias^2: base (a=1.00) vs best (a={best_a}) ===")
    lb_base = np.stack(layer_bias[1.00]).mean(axis=0)
    lb_best = np.stack(layer_bias[best_a]).mean(axis=0)
    print(f"{'layer':>6} {'base':>12} {'best':>12}")
    for l in range(depth):
        print(f"{l:>6} {lb_base[l]:>12.3e} {lb_best[l]:>12.3e}")


if __name__ == "__main__":
    main()
