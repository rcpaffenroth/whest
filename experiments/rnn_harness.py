"""Recurrent models for learning the deterministic map F(W) - a swap-in harness.

------------------------------------------------------------------------------
WHAT THIS DOES
------------------------------------------------------------------------------
We learn F(W) = E_{x~N(0,I)}[ f(x; W) ], the per-neuron final-layer mean
activation of a depth-K width-D ReLU MLP, as a function of that MLP's weights W.
(See whest_toy.py and notebook 02 for the background: F is a smooth but heavy-
tailed function of W, and a *flat* MLP on flatten(W) learns it poorly, while a
model that respects the ordered-layer structure does much better.)

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
MODELS_TO_RUN = ["rnn", "lstm", "gru"]   # any subset; one name trains just that
HIDDEN   = 256          # recurrent hidden width
LAYERS   = 2            # stacked recurrent layers
N_TRAIN  = 600_000      # number of distinct networks used for training
M        = 8_192        # MC samples per label (label cleanliness)
EPOCHS   = 300
BATCH    = 4096
LR       = 1e-3
WD       = 1e-5         # AdamW weight decay
PATIENCE = 50           # early-stopping patience (epochs w/o val improvement)
SEED     = 0

K, D = toy.K, toy.D                       # depth 8, width 8
IN_FEATURES = D * D                        # 64: one layer-matrix flattened = one token
OUT_DIM = D                                # 8: the final-layer mean we predict
dev = "cuda" if torch.cuda.is_available() else "cpu"


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
def make_labels(seeds, chunk=4000, input_seed=1_000_000):
    """Generate (flatten(W), F(W)) for the given network seeds, batched over
    networks. Returns X (n, K*D*D) and Y (n, D), float32 on CPU."""
    Xs, Ys = [], []
    for s0 in range(0, len(seeds), chunk):
        sc = seeds[s0:s0 + chunk]
        W = torch.stack([toy.sample_weights(int(s)).float() for s in sc]).to(dev)  # (B,K,D,D)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(input_seed + s0)
        x = torch.randn(M, D, generator=g, device=dev)              # (M, D), shared within chunk
        z = x.unsqueeze(0).expand(B, M, D)                          # (B, M, D)
        for i in range(K):
            # z <- ReLU(z @ W_i^T), the i-th layer applied to all B networks at once
            z = torch.relu(torch.einsum("bmd,bed->bme", z, W[:, i]))
        Xs.append(W.reshape(B, -1).cpu())                           # flatten(W): layer-major
        Ys.append(z.mean(dim=1).cpu())                              # (B, D) = MC label F(W)
    return torch.cat(Xs), torch.cat(Ys)


def load_pool(n):
    """Training pool of n networks (seeds 0..n-1), cached to disk."""
    cache = os.path.join(_HERE, f"data_pool{n // 1000}_M{M}_k{K}_d{D}.pt")
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
    Ytr_s = ((torch.log1p(Ytr) - ym) / ys).to(dev)                 # standardized log-target
    Yva_log, Yte_log = torch.log1p(Yva), torch.log1p(Yte)          # log space, original scale
    unstd = lambda P: P.cpu() * ys + ym                            # standardized -> log space

    n = Xtr_s.shape[0]
    best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            lossf(net(Xtr_s[idx]), Ytr_s[idx]).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            va = ((unstd(net(Xva_s)) - Yva_log) ** 2).mean().item()
            te = ((unstd(net(Xte_s)) - Yte_log) ** 2).mean().item()
        if va < best_va:                                            # early stopping on val
            best_va, best_te, wait = va, te, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    [{kind}] early stop at epoch {ep}")
                break

    const = ((torch.log1p(Ytr).mean(0) - Yte_log) ** 2).mean().item()   # constant-predictor MSE
    r2 = 1 - best_te / const
    n_params = sum(p.numel() for p in net.parameters())
    return {"kind": kind, "r2": r2, "mse": best_te, "params": n_params}


def main():
    Xp, Yp = load_pool(N_TRAIN)
    Xtr, Ytr = Xp[:N_TRAIN], Yp[:N_TRAIN]
    # disjoint held-out networks (seeds far from the training range)
    Xva, Yva = make_labels(list(range(905_000, 910_000)))
    Xte, Yte = make_labels(list(range(900_000, 905_000)))

    # standardizers fit on the training weights / log-targets
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    yt = torch.log1p(Ytr); ym, ys = yt.mean(0), yt.std(0) + 1e-8

    print(f"\ndevice={dev}  K={K} D={D}  tokens/network={K} x {IN_FEATURES}-dim")
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}   "
          f"hidden={HIDDEN} layers={LAYERS}\n")

    results = []
    for kind in MODELS_TO_RUN:
        r = train(kind, Xtr, Ytr, Xva, Yva, Xte, Yte, xm, xs, ym, ys)
        results.append(r)
        print(f"  {kind:5s}  params={r['params']:>9d}  test R^2(log)={r['r2']:.3f}  "
              f"MSE={r['mse']:.3e}", flush=True)

    print(f"\n{'model':6s} {'params':>10s} {'R2(log)':>9s}")
    for r in sorted(results, key=lambda d: -d["r2"]):
        print(f"{r['kind']:6s} {r['params']:>10d} {r['r2']:>9.3f}")


if __name__ == "__main__":
    main()
