"""Is the UT error bias or variance? -- the experiment that gates the corrector program.

Decompose the finite-rotation UT estimate of the final-layer mean activation:

    UT_R = (UT_R - UT_inf)        [variance: -> 0 as rotations R -> infinity]
         + (UT_inf - truth)       [bias: the irreducible Edgeworth residual]

A learned corrector can ONLY attack the bias term; the variance term is killed
for free by adding rotations. So the question that decides whether the RNN
corrector is worth building is: at our operating rotation count, what fraction
of the final-layer MSE is bias (attackable) vs variance (just add rotations)?

We estimate:
  * truth       -- large Monte-Carlo average (with a second independent MC run
                   to confirm MC noise << signal),
  * UT_inf      -- UT at a large rotation count (proxy for R -> infinity),
  * MSE(R)      -- final-layer MSE of UT vs truth at increasing R.

Outputs (per seed, averaged over seeds):
  * MSE(R) vs R  -- plateau => bias floor; slope -1 to zero => pure variance,
  * bias floor / MSE(operating R) -- the attackable fraction,
  * bias^2 per layer -- to check the residual is concentrated in the deep layers.

Run:
    uv run python experiments/whest_bias_variance.py
    uv run python experiments/whest_bias_variance.py --seeds 8 --mc-samples 1000000
"""

import argparse

import numpy as np

from whest_ut import build_mlp, sigma_points, unscented_means, monte_carlo_means


def ut_final_means(W, width, rotations, rng):
    """Final-layer UT mean estimate, shape (width,), with the frozen sigma scheme."""
    X = sigma_points(width, rotations, rng)
    return unscented_means(W, X)[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--mc-samples", type=int, default=1_000_000)
    p.add_argument("--r-inf", type=int, default=256,
                   help="rotations used as the R->infinity proxy for UT_inf")
    p.add_argument("--operating-r", type=int, default=13,
                   help="rotation count representative of the ~10%% budget operating point")
    args = p.parse_args()

    width, depth = args.width, args.depth
    R_grid = [1, 2, 4, 8, 16, 32, 64, 128]

    # Per-seed accumulators.
    mse_of_R = {R: [] for R in R_grid}          # MSE(UT_R vs truth), final layer
    bias_floor = []                             # MSE(UT_inf vs truth), final layer
    mc_noise = []                               # MSE between two independent MC runs, final layer
    ut_var_at_op = []                           # Var of UT_operating across independent rotation draws
    layer_bias_sq = []                          # (depth,) per-layer bias^2

    for s in range(args.seeds):
        W = build_mlp(width, depth, seed=s)

        # --- truth (all layers) and an independent MC replicate (to bound MC's own noise) ---
        truth_all = monte_carlo_means(W, width, args.mc_samples,
                                      np.random.default_rng(10_000 + s))   # (depth, width)
        truth2 = monte_carlo_means(W, width, args.mc_samples,
                                   np.random.default_rng(20_000 + s))[-1]
        truth = truth_all[-1]
        mc_noise.append(((truth - truth2) ** 2).mean())

        # --- UT_inf (large-R proxy): bias floor (final) and per-layer bias^2 ---
        X_inf = sigma_points(width, args.r_inf, np.random.default_rng(30_000 + s))
        ut_inf_all = unscented_means(W, X_inf)                             # (depth, width)
        bias_floor.append(((ut_inf_all[-1] - truth) ** 2).mean())
        layer_bias_sq.append(((ut_inf_all - truth_all) ** 2).mean(axis=1))

        # --- MSE(R) sweep; each R gets its own frozen rng draw ---
        for R in R_grid:
            ut_R = ut_final_means(W, width, R, np.random.default_rng(40_000 + s))
            mse_of_R[R].append(((ut_R - truth) ** 2).mean())

        # --- variance of UT at the operating point across independent rotation draws ---
        draws = np.stack([
            ut_final_means(W, width, args.operating_r, np.random.default_rng(50_000 + s + 1000 * t))
            for t in range(8)
        ])                                          # (8, width)
        ut_var_at_op.append(draws.var(axis=0).mean())

        print(f"seed {s}: bias_floor={bias_floor[-1]:.3e}  "
              f"mc_noise={mc_noise[-1]:.3e}  ut_var@R{args.operating_r}={ut_var_at_op[-1]:.3e}")

    # ---- aggregate report ----
    bias_floor = np.array(bias_floor)
    mc_noise = np.array(mc_noise)
    ut_var_at_op = np.array(ut_var_at_op)
    layer_bias_sq = np.stack(layer_bias_sq).mean(axis=0)   # (depth,)

    print(f"\n=== final-layer MSE(UT_R vs truth), mean over {args.seeds} seeds ===")
    print(f"{'R':>6} {'MSE(R)':>12}")
    for R in R_grid:
        print(f"{R:>6} {np.mean(mse_of_R[R]):>12.3e}")
    print(f"{'inf(~'+str(args.r_inf)+')':>6} {bias_floor.mean():>12.3e}   <- bias floor")

    # MC noise adds independently to the measured bias floor: a single MC mean has
    # variance v per neuron, and MSE(truth vs truth2) = 2v, so v = mc_noise/2 and the
    # debiased bias floor is bias_floor - v. This is the honest attackable target.
    v = mc_noise.mean() / 2
    debiased_bias = bias_floor.mean() - v

    R_op = min(R_grid, key=lambda R: abs(R - args.operating_r))
    mse_op = np.mean(mse_of_R[R_op])
    print(f"\nMC noise floor 2v (indep MC runs):    {mc_noise.mean():.3e}"
          f"   (per-run noise v = {v:.3e})")
    print(f"  (measured bias floor must sit well above 2v to be trustworthy)")
    print(f"UT variance at operating R={args.operating_r}:      {ut_var_at_op.mean():.3e}")
    print(f"measured bias floor:                  {bias_floor.mean():.3e}")
    print(f"DEBIASED bias floor (attackable):     {debiased_bias:.3e}"
          f"   {'<- suspect, <= MC noise' if debiased_bias < v else ''}")
    print(f"\nMSE at operating R={R_op}:               {mse_op:.3e}"
          f"   (= bias {debiased_bias:.2e} + variance {mse_op - debiased_bias:.2e})")
    print(f"ATTACKABLE FRACTION = debiased_bias/MSE(op) = {debiased_bias / mse_op:.1%}")
    print(f"  -> high => corrector is the right lever; low => just add rotations.")

    print(f"\n=== per-layer bias^2 (is it concentrated deep?) ===")
    print(f"{'layer':>6} {'bias^2':>12}")
    for l in range(depth):
        bar = "#" * int(60 * layer_bias_sq[l] / (layer_bias_sq.max() + 1e-30))
        print(f"{l:>6} {layer_bias_sq[l]:>12.3e} {bar}")


if __name__ == "__main__":
    main()
