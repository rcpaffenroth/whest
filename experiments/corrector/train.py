"""train.py -- retrain the corrector per track and compare to the UT base (PLAN_v1).

The decisive v1 experiment. One architecture (model.Corrector), retrained THREE times,
once per track, with that track's channel mask applied during training (PLAN_v1 sec 1):

    Track 2 (cloud-only)  : w_on=0, c_on=1
    Track 3 (weights-only): w_on=1, c_on=0     (deliberately starved: anchor + PR)
    Track 4 (both)        : w_on=1, c_on=1     (the W x cloud interaction -- the thesis)

Track 1 is the UT base itself (a zero correction), so its val MSE is just mean(target^2).
Every track is initialized identically (same seed) so only the mask differs. Loss is the
scored quantity in raw units: mean( (anchor + correction - truth)^2 ) on the final layer,
evaluated on the 20 held-out MLPs. The corrector's forward pass is ~1e6 FLOPs vs the UT's
~1e10, so it stays far under the 10% budget floor -> multiplier 0.1 -> adjusted = raw*0.1.

Questions this answers: does Track 4 beat the UT base? does the interaction (Track 4)
beat cloud-only (Track 2)? A zero correction is always available, so no track should do
worse than the UT except by overfitting (guarded by zero-init readout + weight decay +
held-out eval).
"""

import torch

from dataset import build_dataset, channel_mask
from model import Corrector

EPOCHS = 4000
LR = 3e-3
WEIGHT_DECAY = 1e-4       # pulls the correction toward 0 (do-no-harm, PLAN_v1 sec 5)
MULT = 0.1               # FLOP multiplier floor; corrector FLOPs are negligible


def run_track(data, w_on, c_on, device):
    """Train one track (mask fixed by w_on,c_on) and return (train_mse, val_mse, best_val)."""
    torch.manual_seed(0)                       # identical init across tracks; only mask differs
    pm, nm = channel_mask(w_on, c_on)
    pm, nm = pm.to(device), nm.to(device)

    # mask the ports for this track (applied to train AND val -- the track IS its mask)
    p_tr, n_tr = data["pooled_train"] * pm, data["neuron_train"] * nm
    p_va, n_va = data["pooled_val"] * pm, data["neuron_val"] * nm
    a_tr, a_va = data["anchor_train"], data["anchor_val"]
    truth_tr = a_tr + data["target_train"]     # truth = anchor + (truth - anchor)
    truth_va = a_va + data["target_val"]

    model = Corrector(out_scale=data["out_scale"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = ((model(p_tr, n_tr, a_tr) - truth_tr) ** 2).mean()   # scored quantity, raw units
        loss.backward()
        opt.step()
        if ep % 250 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val = ((model(p_va, n_va, a_va) - truth_va) ** 2).mean().item()
            best_val = min(best_val, val)

    model.eval()
    with torch.no_grad():
        train_mse = ((model(p_tr, n_tr, a_tr) - truth_tr) ** 2).mean().item()
        val_mse = ((model(p_va, n_va, a_va) - truth_va) ** 2).mean().item()
    return train_mse, val_mse, best_val


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_dataset()
    data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

    base_val = (data["target_val"] ** 2).mean().item()   # Track 1: UT, zero correction
    print(f"device: {device}   n_rot: {data['n_rot']}")
    print(f"\nTrack 1  (UT base):  val MSE {base_val:.3e}   adjusted {base_val * MULT:.3e}\n")

    print(f"{'track':<22}{'train MSE':>12}{'val MSE':>12}{'adjusted':>12}{'vs UT':>9}{'best val':>12}")
    for name, (w, c) in [("2 cloud-only", (0, 1)), ("3 weights-only", (1, 0)),
                         ("4 both (W x cloud)", (1, 1))]:
        tr, va, best = run_track(data, w, c, device)
        print(f"{name:<22}{tr:>12.3e}{va:>12.3e}{va * MULT:>12.3e}"
              f"{base_val / va:>8.2f}x{best:>12.3e}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
