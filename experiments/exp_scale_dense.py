"""Rung 0 - dense recurrent scaling sweep (the bitter-lesson baseline).

How does held-out R^2 scale with the RNN's hidden width, at fixed large data?
Sweep hidden in {256, 512, 1024, 2048} for a dense LSTM over the K ordered layer-
tokens, target log1p(F), on the 600k-network pool. This sets the dense curve the
(later, RCP-supplied) Monarch-sparse cells must beat -- and tells us whether the
recurrent approach keeps climbing with scale or bends.

Data-vs-size is complementary: earlier the GRU scaled 0.48->0.65->0.83 across
100k->300k->600k nets (still rising). This script fixes data and varies capacity.
"""

import torch
import torch.nn as nn

import whest_toy as toy

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
K, D = toy.K, toy.D
M = 8192
LAYERS, LR, BATCH, EPOCHS, PATIENCE = 2, 1e-3, 4096, 250, 40
HIDDENS = [256, 512, 1024, 2048]


def make_labels(seeds, chunk=4000, input_seed=1_000_000):
    Xs, Ys = [], []
    for s0 in range(0, len(seeds), chunk):
        W = torch.stack([toy.sample_weights(int(s)).float() for s in seeds[s0:s0 + chunk]]).to(dev)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(input_seed + s0)
        z = torch.randn(M, D, generator=g, device=dev).unsqueeze(0).expand(B, M, D)
        for i in range(K):
            z = torch.relu(torch.einsum("bpd,bed->bpe", z, W[:, i]))
        Xs.append(W.reshape(B, -1).cpu()); Ys.append(z.mean(1).cpu())
    return torch.cat(Xs), torch.cat(Ys)


# reuse the harness's 600k mean pool if present
import os
PC = f"data_pool600_M{M}_k{K}_d{D}_mean.pt"
if os.path.exists(PC):
    b = torch.load(PC); Xp, Yp = b["X"], b["Y"]; print("loaded", PC)
else:
    Xp, Yp = make_labels(list(range(600_000))); torch.save({"X": Xp, "Y": Yp}, PC); print("cached", PC)
Xva, Yva = make_labels(list(range(905_000, 910_000)))
Xte, Yte = make_labels(list(range(900_000, 905_000)))

tf = torch.log1p
xm, xs = Xp.mean(0), Xp.std(0) + 1e-8
seq = lambda X: (((X - xm) / xs).reshape(-1, K, D * D)).to(dev)
Xtr, Xva_d, Xte_d = seq(Xp), seq(Xva), seq(Xte)
yt = tf(Yp); ym, ys = yt.mean(0), yt.std(0) + 1e-8
Ytr = ((yt - ym) / ys).to(dev)
Yva_t, Yte_t = tf(Yva), tf(Yte)
const = ((yt.mean(0) - Yte_t) ** 2).mean().item()


class LSTMNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.rnn = nn.LSTM(D * D, hidden, LAYERS, batch_first=True); self.head = nn.Linear(hidden, D)
    def forward(self, x): return self.head(self.rnn(x)[0][:, -1])


print(f"\ndense LSTM scaling on {len(Xp)} nets\n{'hidden':>7} {'params':>11} {'R2(log)':>9}")
for h in HIDDENS:
    torch.manual_seed(0)
    net = LSTMNet(h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.MSELoss()
    unz = lambda P: P.cpu() * ys + ym
    n = len(Xtr); best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n, device=dev)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]; opt.zero_grad()
            lf(net(Xtr[idx]), Ytr[idx]).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            v = ((unz(net(Xva_d)) - Yva_t) ** 2).mean().item()
            t = ((unz(net(Xte_d)) - Yte_t) ** 2).mean().item()
        if v < best_va: best_va, best_te, wait = v, t, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    print(f"{h:>7} {sum(p.numel() for p in net.parameters()):>11} {1 - best_te / const:>9.3f}", flush=True)
