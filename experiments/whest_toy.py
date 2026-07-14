"""Deterministic 2x2 toy for the ARC White-Box Estimation Challenge.

The full challenge asks: given the weights W of a deep ReLU MLP, predict the
per-neuron mean post-ReLU activation

    mu = E_{x ~ N(0,I)} [ z_L ]                                      (the target)

WITHOUT running the network on many inputs.  Once W is fixed the input is
integrated out, so mu is a *deterministic* function of the weights:

    F(W) = E_{x ~ N(0,I)} [ f(x; W) ].

This module is a small non-trivial instance of that map.  We use width d = 8
and depth k = 8.

  WHY NOT width 2?  A width-d ReLU layer keeps an input only if some coordinate
  stays positive.  With d = 2 the surviving fraction halves every ~2 layers, so
  a depth-32 net is 100% dead (F identically 0) -- and because ReLU is
  positively homogeneous, rescaling the weights never changes which coordinates
  are positive, so NO initialization fixes it.  Width is the cure: it gives the
  redundancy that keeps deep ReLU nets alive (exactly why He init works at width
  256).  d = 8, k = 8 is the smallest config here that is reliably alive
  (0% dead, ||F|| ~ 0.75) while still deep enough to show sensitivity compounding
  with depth.  All of k, d, var are top-of-file knobs -- change and re-run.

    x = flat(W) in R^{k*d*d} = R^512,      F : R^512 -> R^8.

Everything here is written to be framework-agnostic in spirit (plain matmuls +
ReLU) so the eventual numpy submission is a mechanical port.  We use torch now
for autograd (step 1) and training (step 2).

Notation follows tasks/RCP-ideas-v2.md:
  f  : R^d -> R^d, one forward pass f(x) = ReLU(W_k ... ReLU(W_1 x) ...).
  F  : R^{k*d*d} -> R^d, the mean of f over x ~ N(0, I) -- the deterministic
       target, estimated here by Monte-Carlo (F_mc).
"""

import torch

# --- default problem geometry (change freely) --------------------------------
K = 8           # depth: number of layers
D = 8           # width: d x d weight matrices  (d=2 is degenerate -- see above)


def sample_weights(seed, k=K, d=D, var=None):
    """Draw one random network's weights W, shape (k, d, d).

    He initialization preserves activation variance through ReLU layers:
    entries ~ N(0, 2 / fan_in) with fan_in = d.  This is the edge-of-chaos
    scaling where signal magnitude is roughly preserved across the k layers
    (the regime where sensitivity is interesting).  Pass `var` to override.
    """
    if var is None:
        var = 2.0 / d                                    # He init at width d
    g = torch.Generator().manual_seed(seed)
    # Draw in float64 ALWAYS: torch's normal RNG stream depends on dtype, so
    # without this a given seed would mean different networks under a float32 vs
    # float64 default. Cast downstream (.float()) if you want float32.
    W = torch.randn(k, d, d, generator=g, dtype=torch.float64) * var**0.5   # (k, d, d)
    return W


def f(W, x):
    """One forward pass, batched over inputs.

    f(x) = ReLU(W_k ... ReLU(W_1 x) ...).  ReLU is applied at EVERY layer,
    including the last, matching the challenge's y = ReLU(W^T x) per layer.

    W : (k, d, d)      the weights of a single network
    x : (n, d)         a batch of n inputs
    returns z : (n, d) the post-ReLU activations of the final layer
    """
    z = x                                                # (n, d)
    for W_i in W:                                        # W_i : (d, d)
        # z <- ReLU(z @ W_i^T): row-vector convention, layer i maps R^d -> R^d.
        z = torch.relu(z @ W_i.T)                        # (n, d)
    return z


def F_mc(W, M, seed):
    """Monte-Carlo estimate of the deterministic target F(W) = E_x[f(x; W)].

    Averages f over M inputs x ~ N(0, I).  This is the *label generator* for
    step 2; its noise is ~ sigma_f / sqrt(M) per output coordinate (the label
    convergence study in 01 pins down the actual sigma_f).

    returns a length-d vector (the per-neuron final-layer mean).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, W.shape[-1], generator=g, dtype=W.dtype)   # (M, d), match W
    z = f(W, x)                                          # (M, d)
    return z.mean(dim=0)                                 # (d,)
