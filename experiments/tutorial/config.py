"""Shared scale presets for the whest tutorial notebooks.

The whole point of these notebooks is to let you turn ONE knob — the scale — and
watch the same code go from "runs in a few seconds so I can play" up to the real
competition geometry. Each preset fixes the network geometry (width d, depth K),
the Monte-Carlo budget M for the ground-truth mean, and the training budget for
our learned method.

Usage in a notebook:

    from config import PRESETS, describe
    cfg = PRESETS["small"]          # or "tiny" / "medium" / "competition"
    cfg["width"], cfg["depth"]      # ... etc

Override freely after picking one, e.g. `cfg = {**PRESETS["small"], "depth": 16}`.
Nothing here is sacred — copy a preset and change a field.
"""

# Geometry + budgets. Fields:
#   width d, depth K            : the ReLU MLP whose mean activation we predict
#   mc_samples M                : random inputs pushed through for the "truth" mean
#   ut_rotations                : randomized-UT rotations (2*d*rotations sigma points)
#   n_train, n_eval             : # random networks for training / testing the method
#   channels c                  : per-coordinate hidden width of the equivariant model
#   epochs, batch, lr           : training the method
PRESETS = {
    # THE DEFAULT — the whole notebook runs in a few seconds on CPU. For playing / first read.
    "tiny":        dict(width=8,   depth=4,  mc_samples=1500,  ut_rotations=3,
                        n_train=1500,   n_eval=500,  channels=16, epochs=25,  batch=512,  lr=1e-3),
    # a few minutes end-to-end — the default for understanding
    "small":       dict(width=8,   depth=8,  mc_samples=4096,  ut_rotations=8,
                        n_train=20000,  n_eval=2000, channels=32, epochs=60,  batch=1024, lr=1e-3),
    # tens of minutes — closer to where concentration kicks in
    "medium":      dict(width=32,  depth=8,  mc_samples=8192,  ut_rotations=8,
                        n_train=50000,  n_eval=5000, channels=32, epochs=80,  batch=2048, lr=1e-3),
    # the real Phase-1 geometry — slow + memory-heavy, for reference runs only
    "competition": dict(width=256, depth=32, mc_samples=8192,  ut_rotations=13,
                        n_train=50000,  n_eval=5000, channels=32, epochs=120, batch=512,  lr=1e-3),
}


def describe(cfg):
    """One-line human summary of a preset, handy to print at the top of a run."""
    d, K = cfg["width"], cfg["depth"]
    return (f"width d={d}, depth K={K}  ->  W in R^{K*d*d}, F: R^{K*d*d} -> R^{d}   "
            f"(MC M={cfg['mc_samples']}, n_train={cfg['n_train']}, channels c={cfg['channels']})")
