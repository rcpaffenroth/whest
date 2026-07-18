"""Recurrent models for learning the deterministic map F(W) - a swap-in harness.

------------------------------------------------------------------------------
WHAT THIS DOES
------------------------------------------------------------------------------
We learn F(W) = E_{x~N(0,I)}[ f(x; W) ], the per-neuron final-layer mean
activation of a depth-K width-D ReLU MLP, as a function of that MLP's weights W.
(See whest_toy.py and notebook 02 for the background: F is a smooth but heavy-
tailed function of W, and a *flat* MLP on flatten(W) learns it poorly, while a
model that respects the ordered-layer structure does much better.)

Two prediction targets, selected by the TARGET knob (only the LABELS change - the
recurrent model is byte-for-byte identical):
  TARGET="mean"     : predict F(W) directly.
  TARGET="residual" : predict the residual F(W) - UT(W), where UT(W) is a CHEAP
                      unscented-transform estimate of the mean from a small, FIXED
                      set of deterministic sigma points. This is the "corrector"
                      idea: stand on UT's shoulders and learn only its (small,
                      deterministic) error. The corrected estimate is UT + (LSTM
                      prediction); we report whether it beats raw UT.

------------------------------------------------------------------------------
THE TRAINING DATA  (one example = one random network)
------------------------------------------------------------------------------
  input   W       : the weights of ONE random ReLU MLP, shape (K, D, D).
                    Drawn He-style, W_i ~ N(0, 2/D). Here K=8 layers, D=8 wide.
  label   F(W)    : shape (D,). Estimated by Monte-Carlo: push M=8192 Gaussian
                    inputs x through f and average the final post-ReLU output.
                    M is large enough that label noise (~1e-4) is negligible.

  We generate N such (W, F(W)) pairs. Networks are indexed by seed; train / val /
  test use DISJOINT seed ranges, so the model is always evaluated on networks it
  has never seen. Labels are cached to disk (data_pool*.pt) so re-runs are fast.

------------------------------------------------------------------------------
HOW THE RECURRENT MODEL IS USED  (this is the whole point)
------------------------------------------------------------------------------
The K=8 layers of W are an ORDERED sequence (layer 0 acts on the input first,
layer K-1 produces the output). We hand the recurrent model that sequence:

      W  (K, D, D)  --reshape-->  a length-K sequence of tokens,
                                  each token = flatten(W_i) in R^{D*D}=R^64.

      token_0 = flatten(W_0)   (input layer)   ->  |
      token_1 = flatten(W_1)                       |  fed in this order
        ...                                        |  into an RNN/LSTM/GRU
      token_{K-1} = flatten(W_{K-1}) (output)  ->  v

  The recurrent core scans the tokens left-to-right, carrying a hidden state.
  We take the hidden state AFTER the last layer-token and pass it through a
  linear head to predict F(W) in R^D. Intuitively the recurrence mirrors the
  layer-by-layer composition that actually defines F.

  This is still pure black-box regression from weights: no sampling at inference,
  no analytic moment propagation. Only the input *ordering* encodes structure.

  Targets are learned in log(1+F) space (F is a heavy-tailed product of per-layer
  gains) and standardized; metrics are reported as held-out R^2 in that space.

------------------------------------------------------------------------------
TO SWAP IN A DIFFERENT RECURRENT MODEL
------------------------------------------------------------------------------
Edit `make_core()` below. Any module that maps a (B, K, in_features) sequence to
an output tensor whose [:, -1] slice is the final (B, hidden) state works - the
built-in nn.RNN / nn.LSTM / nn.GRU all satisfy this (they return (output, ...)
with output of shape (B, K, hidden)). Drop your own nn.Module in the same way.

Run:  uv run python experiments/rnn_harness.py
Set MODELS_TO_RUN to one name to train just that; leave all three to compare.
"""

import os
import sys

import torch
import torch.nn as nn

# make the script runnable from any working directory: anchor imports and the
# data cache to this file's own directory (experiments/).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import whest_toy as toy

# ============================== KNOBS =========================================
MODELS_TO_RUN = ["lstm"]   # any subset of {"rnn","lstm","gru"}; one name trains just that
TARGET   = "residual"   # "mean": predict F(W).   "residual": predict F(W) - UT(W).
HIDDEN   = 256          # recurrent hidden width
LAYERS   = 2            # stacked recurrent layers
N_TRAIN  = 600_000      # number of distinct networks used for training
M        = 8_192        # MC samples per label (label cleanliness)
EPOCHS   = 300
BATCH    = 4096
LR       = 1e-3
WD       = 1e-5         # AdamW weight decay
PATIENCE = 50           # early-stopping patience (epochs w/o val improvement)
PRINT_EVERY = 10        # epochs between progress lines (0 = silent)
SEED     = 0

K, D = toy.K, toy.D                       # depth 8, width 8
IN_FEATURES = D * D                        # 64: one layer-matrix flattened = one token
OUT_DIM = D                                # 8: the final-layer quantity we predict
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== UNSCENTED TRANSFORM (fixed points) =====================
# The FIXED degree-3 cubature set for N(0, I_D): rows +- sqrt(D) e_i, 2D points,
# equal weight. Sample mean 0 and sample covariance I exactly. CRITICAL: this set
# is FIXED for the whole dataset -- NO random rotations. A randomized UT would make
# UT(W) noisy and corrupt the residual label F(W)-UT(W); a fixed set makes UT(W) a
# deterministic function of W, so the residual is a clean deterministic target.
UT_POINTS = torch.cat([torch.eye(D), -torch.eye(D)], dim=0) * (D ** 0.5)   # (2D, D)


def target_transform(y):
    """Map raw labels to the space we learn in. The mean F>=0 is nonnegative and
    heavy-tailed -> log(1+F). The residual F-UT is SIGNED -> identity (no log)."""
    return torch.log1p(y) if TARGET == "mean" else y


# ======================= THE SWAP POINT: recurrent core =======================
def make_core(kind):
    """Return a recurrent module mapping (B, K, IN_FEATURES) -> (output, ...),
    where output has shape (B, K, HIDDEN). Add your own model here."""
    if kind == "rnn":
        return nn.RNN(IN_FEATURES, HIDDEN, LAYERS, batch_first=True, nonlinearity="tanh")
    if kind == "lstm":
        return nn.LSTM(IN_FEATURES, HIDDEN, LAYERS, batch_first=True)
    if kind == "gru":
        return nn.GRU(IN_FEATURES, HIDDEN, LAYERS, batch_first=True)
    raise ValueError(f"unknown recurrent kind: {kind}")


class RecurrentF(nn.Module):
    """Scan the K layer-tokens, then map the final hidden state to F(W)."""
    def __init__(self, kind):
        super().__init__()
        self.core = make_core(kind)
        self.head = nn.Linear(HIDDEN, OUT_DIM)

    def forward(self, x):                  # x: (B, K, IN_FEATURES)
        output = self.core(x)[0]           # (B, K, HIDDEN); [0] drops h/(h,c)
        last = output[:, -1]               # (B, HIDDEN): state after the last layer
        return self.head(last)             # (B, OUT_DIM)


# ============================ DATA GENERATION =================================
def _propagate(points, W):
    """Push a batch of input points through every network's layers.
    points: (B, P, D)  W: (B, K, D, D)  ->  final post-ReLU activations (B, P, D)."""
    z = points
    for i in range(K):
        # z <- ReLU(z @ W_i^T), the i-th layer applied to all B networks at once
        z = torch.relu(torch.einsum("bpd,bed->bpe", z, W[:, i]))
    return z


def make_labels(seeds, chunk=4000, input_seed=1_000_000):
    """Generate (flatten(W), target) for the given network seeds, batched over
    networks. The target is F(W) if TARGET=="mean", else the residual F(W)-UT(W).
      F(W)  : Monte-Carlo mean over M random N(0,I) inputs (the "truth").
      UT(W) : mean over the 2D FIXED sigma points (the cheap estimator).
    Returns X (n, K*D*D) and Y (n, D), float32 on CPU."""
    S = UT_POINTS.to(dev)                                           # (2D, D), fixed
    Xs, Ys = [], []
    for s0 in range(0, len(seeds), chunk):
        sc = seeds[s0:s0 + chunk]
        W = torch.stack([toy.sample_weights(int(s)).float() for s in sc]).to(dev)  # (B,K,D,D)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(input_seed + s0)
        x = torch.randn(M, D, generator=g, device=dev)             # (M, D), shared within chunk
        F = _propagate(x.unsqueeze(0).expand(B, M, D), W).mean(dim=1)          # (B, D) MC truth
        if TARGET == "residual":
            UT = _propagate(S.unsqueeze(0).expand(B, 2 * D, D), W).mean(dim=1) # (B, D) cheap UT
            Y = F - UT                                             # the residual to learn
        else:
            Y = F
        Xs.append(W.reshape(B, -1).cpu())                          # flatten(W): layer-major
        Ys.append(Y.cpu())
    return torch.cat(Xs), torch.cat(Ys)


def load_pool(n):
    """Training pool of n networks (seeds 0..n-1), cached to disk."""
    cache = os.path.join(_HERE, f"data_pool{n // 1000}_M{M}_k{K}_d{D}_{TARGET}.pt")
    if os.path.exists(cache):
        blob = torch.load(cache); print("loaded", cache)
        return blob["X"], blob["Y"]
    print(f"generating {n} networks ...")
    X, Y = make_labels(list(range(n)))
    torch.save({"X": X, "Y": Y}, cache); print("cached", cache)
    return X, Y


# ============================== TRAINING ======================================
def as_sequence(X, xm, xs):
    """Standardize flat weights and reshape to the layer-token sequence."""
    return (((X - xm) / xs).reshape(-1, K, IN_FEATURES)).to(dev)    # (n, K, IN_FEATURES)


def train(kind, Xtr, Ytr, Xva, Yva, Xte, Yte, xm, xs, ym, ys):
    torch.manual_seed(SEED)
    net = RecurrentF(kind).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = nn.MSELoss()

    Xtr_s, Xva_s, Xte_s = as_sequence(Xtr, xm, xs), as_sequence(Xva, xm, xs), as_sequence(Xte, xm, xs)
    Ytr_s = ((target_transform(Ytr) - ym) / ys).to(dev)            # standardized target
    Yva_t, Yte_t = target_transform(Yva), target_transform(Yte)   # transform space, original scale
    unstd = lambda P: P.cpu() * ys + ym                            # standardized -> transform space

    # constant-predictor MSE: the "learned nothing" baseline, so we can show R^2 live.
    const = ((target_transform(Ytr).mean(0) - Yte_t) ** 2).mean().item()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  [{kind}] params={n_params}  training on {Xtr_s.shape[0]} nets "
          f"(const MSE={const:.3e})", flush=True)

    n = Xtr_s.shape[0]
    best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        epoch_loss = 0.0                                           # mean train loss this epoch
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = lossf(net(Xtr_s[idx]), Ytr_s[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        sched.step()
        net.eval()
        with torch.no_grad():
            va = ((unstd(net(Xva_s)) - Yva_t) ** 2).mean().item()
            te = ((unstd(net(Xte_s)) - Yte_t) ** 2).mean().item()
        if va < best_va:                                          # early stopping on val
            best_va, best_te, wait = va, te, 0
        else:
            wait += 1

        if PRINT_EVERY and ep % PRINT_EVERY == 0:
            print(f"    [{kind}] ep {ep:3d}  train {epoch_loss:.4f}  "
                  f"val R2 {1 - va / const:6.3f}  test R2 {1 - te / const:6.3f}  "
                  f"best val R2 {1 - best_va / const:6.3f}  "
                  f"lr {sched.get_last_lr()[0]:.1e}  wait {wait}/{PATIENCE}", flush=True)

        if wait >= PATIENCE:
            print(f"    [{kind}] early stop at epoch {ep} "
                  f"(best val R2 {1 - best_va / const:.3f})", flush=True)
            break

    out = {"kind": kind, "r2": 1 - best_te / const, "mse": best_te, "params": n_params}
    if TARGET == "residual":
        # The bottom line: does UT + LSTM beat raw UT? With residual R = F - UT,
        #   raw UT error       = E[(F - UT)^2]        = E[R^2]         = mse_ut
        #   corrected error    = E[(F - (UT+pred))^2] = E[(R - pred)^2] = best_te
        mse_ut = (Yte ** 2).mean().item()
        out["mse_ut"] = mse_ut
        out["mse_corrected"] = best_te
        out["ut_error_removed"] = 1 - best_te / mse_ut     # >0 means the corrector helps
    return out


def main():
    Xp, Yp = load_pool(N_TRAIN)
    Xtr, Ytr = Xp[:N_TRAIN], Yp[:N_TRAIN]
    # disjoint held-out networks (seeds far from the training range)
    Xva, Yva = make_labels(list(range(905_000, 910_000)))
    Xte, Yte = make_labels(list(range(900_000, 905_000)))

    # standardizers fit on the training weights / (transformed) targets
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    yt = target_transform(Ytr); ym, ys = yt.mean(0), yt.std(0) + 1e-8

    print(f"\ndevice={dev}  K={K} D={D}  tokens/network={K} x {IN_FEATURES}-dim")
    print(f"TARGET={TARGET!r}   train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}   "
          f"hidden={HIDDEN} layers={LAYERS}\n")

    results = []
    for kind in MODELS_TO_RUN:
        r = train(kind, Xtr, Ytr, Xva, Yva, Xte, Yte, xm, xs, ym, ys)
        results.append(r)
        print(f"  {kind:5s}  params={r['params']:>9d}  test R^2={r['r2']:.3f}  "
              f"MSE={r['mse']:.3e}", flush=True)

    print(f"\n{'model':6s} {'params':>10s} {'R2':>9s}")
    for r in sorted(results, key=lambda d: -d["r2"]):
        print(f"{r['kind']:6s} {r['params']:>10d} {r['r2']:>9.3f}")

    if TARGET == "residual":
        print(f"\nDoes the corrector (UT + LSTM) beat raw UT?   "
              f"[R^2 above = variance of the residual explained]")
        print(f"{'model':6s} {'MSE raw UT':>12s} {'MSE corrected':>14s} {'UT error removed':>17s}")
        for r in sorted(results, key=lambda d: -d["ut_error_removed"]):
            print(f"{r['kind']:6s} {r['mse_ut']:>12.3e} {r['mse_corrected']:>14.3e} "
                  f"{r['ut_error_removed']:>16.1%}")


if __name__ == "__main__":
    main()
