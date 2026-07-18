"""RCP's idea: feed the RNN the per-layer UT moment trajectory as INPUT.

Hypothesis: the residual F-UT looked unlearnable from W alone because the RNN
would have to internally RECONSTRUCT the UT (sigma-point propagation) before it
could correct it. Hand it the per-layer UT moments as features and that burden
disappears -- and, more generally, W + the cheap dynamical summary should be a
richer input than either alone.

We run the clean 3-way attribution, x 2 targets, same LSTM throughout:
    inputs : W only | UT-features only | W + UT-features
    targets: F (the mean, log1p) | residual F - UT_final (signed, identity)

UT features at layer l = [ mean u_l (R^d), diag Cov_l (R^d) ] from the SINGLE
FIXED 2d-point sigma set (no rotations), computed for free during propagation --
i.e. the cheaply-estimated (mean, spread) state of the activation distribution at
each layer. If W adds nothing beyond UT, "W+UT" ties "UT only"; if the residual is
learnable-given-UT, "W+UT -> residual" beats raw UT.
"""

import os

import numpy as np
import torch
import torch.nn as nn

import whest_toy as toy

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
K, D = toy.K, toy.D                        # depth 8, width 8
M = 8192                                    # MC samples for the F label
N_TRAIN = 300_000
HIDDEN, LAYERS, LR, BATCH, EPOCHS, PATIENCE = 256, 2, 1e-3, 4096, 200, 40

# fixed UT sigma set for N(0,I_D): +-sqrt(D) e_i, no rotations (deterministic)
UT_POINTS = torch.cat([torch.eye(D), -torch.eye(D)], 0) * (D ** 0.5)   # (2D, D)


def gen(seeds, chunk=4000, input_seed=1_000_000):
    """Return per-network: flat(W), UT per-layer mean & cov-diag, MC mean F, UT final."""
    S = UT_POINTS.to(dev)
    Wf, Um, Uc, Ff, Utf = [], [], [], [], []
    for s0 in range(0, len(seeds), chunk):
        W = torch.stack([toy.sample_weights(int(s)).float() for s in seeds[s0:s0 + chunk]]).to(dev)  # (B,K,D,D)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(input_seed + s0)
        x = torch.randn(M, D, generator=g, device=dev)
        z = x.unsqueeze(0).expand(B, M, D)
        for i in range(K):
            z = torch.relu(torch.einsum("bpd,bed->bpe", z, W[:, i]))
        F = z.mean(1)                                              # (B,D) MC truth
        u = S.unsqueeze(0).expand(B, 2 * D, D)
        means, covs = [], []
        for i in range(K):
            u = torch.relu(torch.einsum("bpd,bed->bpe", u, W[:, i]))
            means.append(u.mean(1)); covs.append(u.var(1))        # per-layer (B,D)
        Wf.append(W.reshape(B, -1).cpu())
        Um.append(torch.stack(means, 1).cpu())                    # (B,K,D)
        Uc.append(torch.stack(covs, 1).cpu())
        Ff.append(F.cpu()); Utf.append(means[-1].cpu())           # UT final = last-layer UT mean
    return (torch.cat(Wf), torch.cat(Um), torch.cat(Uc), torch.cat(Ff), torch.cat(Utf))


CACHE = f"data_utfeat_N{N_TRAIN}_M{M}_k{K}_d{D}.pt"
if os.path.exists(CACHE):
    d = torch.load(CACHE); Wf, Um, Uc, Ff, Utf = d["Wf"], d["Um"], d["Uc"], d["Ff"], d["Utf"]; print("loaded", CACHE)
else:
    print("generating train pool ...")
    Wf, Um, Uc, Ff, Utf = gen(list(range(N_TRAIN)))
    torch.save({"Wf": Wf, "Um": Um, "Uc": Uc, "Ff": Ff, "Utf": Utf}, CACHE); print("cached", CACHE)
va = gen(list(range(905_000, 910_000)))        # (Wf,Um,Uc,Ff,Utf)
te = gen(list(range(900_000, 905_000)))
print(f"train {N_TRAIN}  val 5000  test 5000\n")


def tokens(Wf, Um, Uc, kind):
    """Build the (n, K, feat) input sequence for one input variant."""
    Wseq = Wf.reshape(-1, K, D * D)                      # (n,K,D*D) per-layer weights
    UTseq = torch.cat([Um, Uc], dim=-1)                  # (n,K,2D) per-layer UT (mean, cov-diag)
    if kind == "W":      return Wseq
    if kind == "UT":     return UTseq
    if kind == "W+UT":   return torch.cat([Wseq, UTseq], dim=-1)


class LSTMNet(nn.Module):                                 # identical to rnn_harness LSTM, variable input dim
    def __init__(self, in_feat):
        super().__init__()
        self.rnn = nn.LSTM(in_feat, HIDDEN, LAYERS, batch_first=True)
        self.head = nn.Linear(HIDDEN, D)
    def forward(self, x): return self.head(self.rnn(x)[0][:, -1])


def fit(Xtr, Ytr, Xva, Yva, Xte, Yte, transform):
    torch.manual_seed(0)
    xm, xs = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8     # per-feature standardization
    zx = lambda X: ((X - xm) / xs).to(dev)
    yt = transform(Ytr); ym, ys = yt.mean(0), yt.std(0) + 1e-8
    Xtr_d, Ytr_d = zx(Xtr), ((yt - ym) / ys).to(dev)
    Xva_d, Yva_t, Xte_d, Yte_t = zx(Xva), transform(Yva), zx(Xte), transform(Yte)
    net = LSTMNet(Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.MSELoss()
    unz = lambda P: P.cpu() * ys + ym
    n = len(Xtr_d); best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n, device=dev)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]; opt.zero_grad()
            lf(net(Xtr_d[idx]), Ytr_d[idx]).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            v = ((unz(net(Xva_d)) - Yva_t) ** 2).mean().item()
            t = ((unz(net(Xte_d)) - Yte_t) ** 2).mean().item()
        if v < best_va: best_va, best_te, wait = v, t, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    const = ((transform(Ytr).mean(0) - Yte_t) ** 2).mean().item()
    return 1 - best_te / const, best_te


F_resid_te = te[3] - te[4]                                 # test residual F - UT_final
print(f"{'input':7s} {'target':9s} {'R2':>8s} {'note':>28s}")
for kind in ["W", "UT", "W+UT"]:
    Xtr, Xva, Xte = tokens(Wf, Um, Uc, kind), tokens(*va[:3], kind), tokens(*te[:3], kind)
    # target = F (log1p space)
    r2, _ = fit(Xtr, Ff, Xva, va[3], Xte, te[3], torch.log1p)
    print(f"{kind:7s} {'F':9s} {r2:>8.3f} {'log-space R2':>28s}", flush=True)
    # target = residual F - UT_final (identity)
    r2, mse = fit(Xtr, Ff - Utf, Xva, va[3] - va[4], Xte, F_resid_te, lambda y: y)
    mse_ut = (F_resid_te ** 2).mean().item()
    print(f"{kind:7s} {'residual':9s} {r2:>8.3f} {'UT err removed '+f'{1-mse/mse_ut:.1%}':>28s}", flush=True)
