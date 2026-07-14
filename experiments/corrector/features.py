"""features.py -- UT sweep + feature extraction for the recurrent corrector (PLAN_v1).

Runs the frozen-rotation unscented transform (radius = E||x||, the champion fix) on a
given MLP and records, per layer, the two feature channels the corrector consumes.

  Channel 2 -- SAMPLING (no explicit W): moments of the POST-activation clouds z_l that
      the UT propagates -- mean, std, skewness, excess kurtosis -- per neuron (pooled
      over neurons downstream for the recurrence). This is what a "sampling-only"
      corrector (Track 2) may look at.

  Channel 1 -- P(W_k) (requires the explicit weight matrix):
      * pure-W invariant: per-column participation ratio ||w||_2^4 / ||w||_4^4
        (in [1, width]; higher = weights more spread). Survives the cloud being off,
        so it is all Track 3 has.
      * W x cloud interaction: the PRE-activation cloud A_{l+1} = z_l @ W_l gives, per
        output neuron j, the alignment <W_l[:,j], mu_l> (= its mean), std, skew, kurt,
        activation probability P(A>0), and the analytic Gaussian-ReLU mean of
        (alignment, std). Forming A requires W_l, so this is genuinely weight-derived;
        a sampling-only track never sees the pre-activation view (ReLU has already
        discarded the negative mass in the post-activation clouds).

The alignment <W[:,j], mu_prev> is the basis-aligned statistic that correlated +0.86
with the true per-neuron mean in the probe (README section 8.3). The __main__ sanity
check reproduces that correlation on the public phase-1 set, and confirms the sweep
reproduces the champion's final-layer MSE (~2.08e-6, submission 316178).

Conventions (match whest_ut.py and the challenge "y = ReLU(W^T x)"): z_0 = sigma points
representing N(0, I_width); z_{l+1} = ReLU(z_l @ W_l) (the row-vector form of
ReLU(W_l^T z)); weights[l] is (width, width). The scored quantity is the mean of the
final cloud z_depth.
"""

import math

import numpy as np
from scipy.special import ndtr  # standard normal CDF Phi(z), vectorized


# ----------------------------------------------------------------------------------
# The frozen unscented transform (identical sigma construction to ../estimator.py)
# ----------------------------------------------------------------------------------

def expected_norm(width: int) -> float:
    """E||x|| for x ~ N(0, I_width) = sqrt(2) * Gamma((w+1)/2) / Gamma(w/2).

    This is the radius the sigma points sit at -- NOT sqrt(width). Because ReLU is
    degree-1 homogeneous, the mean is controlled by the first radial moment E||x||;
    using it removes UT's radial bias for free (README section 4.1).
    """
    return math.sqrt(2.0) * math.exp(math.lgamma((width + 1) / 2) - math.lgamma(width / 2))


def sigma_points(width: int, n_rot: int, rng: np.random.Generator, radius: float) -> np.ndarray:
    """Randomized-UT sigma set: the +- radius*e_i stencil under n_rot Haar rotations.

    Shape (2*width*n_rot, width). Sequential draws from `rng` make the set a
    deterministic ("frozen") function of the seed, so training and evaluation see the
    same points -- see PLAN_v1 section 5.
    """
    blocks = []
    for _ in range(n_rot):
        Q, _ = np.linalg.qr(rng.standard_normal((width, width)))  # Haar-random orthogonal
        axes = radius * Q.T                                       # rows: directions at `radius`
        blocks += [axes, -axes]                                   # the +- symmetric set
    return np.concatenate(blocks, axis=0).astype(np.float32)


# ----------------------------------------------------------------------------------
# Elementary statistics
# ----------------------------------------------------------------------------------

def column_moments(cloud: np.ndarray) -> np.ndarray:
    """Per-column (per-neuron) moments over the point axis.

    Input cloud (points, width). Returns (width, 4): mean, std, skewness, excess
    kurtosis. Moments are accumulated in float64 for stable 3rd/4th orders.
    """
    x = cloud.astype(np.float64)             # float64: stable 3rd/4th moments
    mean = x.mean(0)
    c = x - mean
    std = np.sqrt((c * c).mean(0))
    # Deep ReLU layers have DEAD neurons (always 0 -> std = 0), whose skew/kurt are
    # undefined (0/0). Divide with a safe denominator, then zero them: a point mass has
    # no shape. (Only post-activation clouds trigger this; pre-activation std is > 0.)
    safe = np.where(std > 0, std, 1.0)
    skew = np.where(std > 0, (c ** 3).mean(0) / safe ** 3, 0.0)
    kurt = np.where(std > 0, (c ** 4).mean(0) / safe ** 4 - 3.0, 0.0)
    return np.stack([mean, std, skew, kurt], axis=1)  # (width, 4)


def gaussian_relu_mean(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[ReLU(N(m, s^2))] = s*phi(m/s) + m*Phi(m/s), elementwise (phi = std normal pdf).

    The exact per-neuron mean IF the pre-activation were Gaussian -- which it is by CLT
    over `width` terms (README section 8.3, R^2 ~ 1.0 given true moments).
    """
    # s > 0 here (pre-activation variance is healthy at width 256); no guard on m/s.
    z = m / s
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)  # standard normal pdf
    return s * phi + m * ndtr(z)                           # ndtr = Phi, std normal CDF


def participation_ratio(W: np.ndarray) -> np.ndarray:
    """Per-column ||w||_2^4 / ||w||_4^4 (in [1, n_rows]); higher = weights more spread."""
    sq = (W * W).sum(0)      # column ||w||_2^2
    quad = (W ** 4).sum(0)   # column ||w||_4^4  (never 0 for Gaussian columns)
    return sq * sq / quad


# ----------------------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------------------

def extract_features(weights, mlp_seed: int, n_rot: int = 13) -> dict:
    """Run the frozen UT and record per-layer, per-neuron features for both channels.

    Returns a dict of numpy arrays (depth = len(weights), w = width):
        ut_final_mean : (w,)          -- the anchor handed to every track
        post          : (depth, w, 4) -- Channel 2: post-activation z_{l+1} moments
        pre           : (depth, w, 4) -- Channel 1: pre-activation A_{l+1} moments
        actprob       : (depth, w)    -- Channel 1: P(A_{l+1} > 0)
        rect          : (depth, w)    -- Channel 1: gaussian_relu_mean(pre_mean, pre_std)
        pr            : (depth, w)    -- Channel 1 (pure-W): participation ratio of W_l
    Final-layer per-neuron features are the [-1] slices; pool over the neuron axis for
    the recurrence. Splitting into channel ports, pooling, and normalization are the
    model's concern and live in dataset.py (written when model.py defines them).
    """
    weights = [np.asarray(w, dtype=np.float32) for w in weights]
    depth = len(weights)
    width = weights[0].shape[0]
    radius = expected_norm(width)
    rng = np.random.default_rng(int(mlp_seed) & (2 ** 63 - 1))  # frozen rotations from the seed

    Z = sigma_points(width, n_rot, rng, radius)  # z_0

    post, pre, actprob, rect, pr = [], [], [], [], []
    for W in weights:
        A = Z @ W                                   # pre-activation cloud (points, width)
        pm = column_moments(A)                      # (width, 4): mean(=alignment), std, skew, kurt
        pre.append(pm)
        actprob.append((A > 0.0).mean(0))
        rect.append(gaussian_relu_mean(pm[:, 0], pm[:, 1]))
        pr.append(participation_ratio(W))
        Z = np.maximum(A, 0.0)                       # post-activation cloud z_{l+1}
        post.append(column_moments(Z))               # (width, 4)

    post = np.stack(post)          # (depth, width, 4)
    pre = np.stack(pre)            # (depth, width, 4)
    actprob = np.stack(actprob)    # (depth, width)
    rect = np.stack(rect)          # (depth, width)
    pr = np.stack(pr)              # (depth, width)

    return dict(
        width=width, depth=depth, n_rot=n_rot,
        ut_final_mean=post[-1, :, 0],  # (width,)
        post=post, pre=pre, actprob=actprob, rect=rect, pr=pr,
    )


# ----------------------------------------------------------------------------------
# Sanity check: does the alignment reproduce its +0.86 correlation with the truth,
# and does the frozen sweep reproduce the champion's final-layer MSE?
# ----------------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


if __name__ == "__main__":
    from datasets import load_dataset

    N = 12
    ds = load_dataset(
        "aicrowd/arc-whestbench-public-2026", "default", revision="v1-phase1", split="mini"
    )

    ut_mse, mse_rect, c_align, c_rect = [], [], [], []
    for i in range(N):
        ex = ds[i]
        W = np.asarray(ex["weights"], dtype=np.float32)
        truth = np.asarray(ex["final_means"], dtype=np.float64)
        rec = extract_features(W, ex["mlp_seed"])

        ut = rec["ut_final_mean"]           # UT final mean (the anchor)
        align = rec["pre"][-1, :, 0]        # <W_last[:,j], mu_penult>  (the alignment feature)
        rect = rec["rect"][-1]              # analytic Gaussian-ReLU of (alignment, std)

        ut_mse.append(np.mean((ut - truth) ** 2))
        mse_rect.append(np.mean((rect - truth) ** 2))
        c_align.append(_pearson(align, truth))
        c_rect.append(_pearson(rect, truth))

    print(f"MLPs: {N}   n_rot: {rec['n_rot']}   width x depth: {rec['width']} x {rec['depth']}\n")
    print(f"UT final-layer MSE (fixed radius): {np.mean(ut_mse):.3e}"
          f"   (submission 316178 was ~2.08e-6)")
    print(f"corr(alignment, truth):            {np.mean(c_align):.3f}"
          f"   (probe with TRUE moments: +0.86)")
    print(f"corr(analytic-rect, truth):        {np.mean(c_rect):.3f}")
    print(f"analytic-rect final-layer MSE:     {np.mean(mse_rect):.3e}"
          f"   (vs UT {np.mean(ut_mse):.3e})")
