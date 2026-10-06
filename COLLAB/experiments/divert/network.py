"""The trained `net_divert`, and REAL CROWN contracts over it.

WHY THIS FILE EXISTS. Everything upstream of it can be run against the
analytic `return_law`, and that is useful for measuring solver cost, but it is
not a result: a paper claiming to verify neuro-symbolic behavior trees that
never verifies a network has verified nothing. The analytic law is a synthetic
ablation; this is the system under test.

WHAT CHANGES WHEN THE NETWORK ARRIVES. The analytic law happens to be affine
with known gains over most of its domain, so its contracts come out with equal
lower and upper slopes and a near-zero bias gap. A trained ReLU net has no such
structure: `compute_bounds` returns two DIFFERENT hyperplanes per output, with
cross-axis coefficients the analytic law does not have, and a bias gap that
grows with the cell. The contract interface was written to carry lA and uA
separately for exactly this reason.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from affine import coarsen
from plant import A_MAX, return_law

STATE_DIM, HIDDEN = 4, 64
WEIGHTS = "net_divert.pt"
# EXACTLY the region `verify.REGION` tiles with contracts, and no larger. The
# network must be defined wherever a contract is queried, and training outside
# that spends capacity - and CROWN tightness - on states the tree can never
# present to it. Training over the whole 2x2 arena instead left a worst cloning
# error of 1.99 against an actuator limit of 2.0.
DOMAIN = (-0.3, 1.7, -0.6, 1.2, -1.5, 1.1, -1.3, 1.1)


class ReturnNet(nn.Module):
    """4 -> 32 -> 32 -> 2 ReLU policy: state in, acceleration out."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 2))

    def forward(self, state):
        """Accelerations for a batch of states."""
        return self.body(state)


def sample_domain(count, rng):
    """Uniform states over `DOMAIN`, with their analytic-law targets."""
    lows = np.array(DOMAIN[0::2], dtype=np.float32)
    highs = np.array(DOMAIN[1::2], dtype=np.float32)
    states = rng.uniform(lows, highs, size=(count, STATE_DIM)).astype(np.float32)
    targets = np.array([return_law(*row) for row in states], dtype=np.float32)
    return torch.from_numpy(states), torch.from_numpy(targets)


def train(epochs=3000, batch=4096, seed=0, verbose=True):
    """Behavioral cloning of `return_law`. Returns the net and its worst error.

    Worst-case error over a held-out sample is reported rather than mean error
    because the thing the contract has to survive is the worst cell, and a mean
    hides exactly the states where cloning failed.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ReturnNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        states, targets = sample_domain(batch, rng)
        loss = nn.functional.mse_loss(net(states), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and epoch % 500 == 499:
            print(f"  epoch {epoch + 1:4d}  mse {loss.item():.5f}", flush=True)
    states, targets = sample_domain(20000, rng)
    with torch.no_grad():
        worst = float((net(states) - targets).abs().max())
    return net, worst


def load(path=WEIGHTS):
    """Load the trained policy in eval mode."""
    net = ReturnNet()
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def main():
    """Train once and save, reporting the cloning error a reviewer will ask for."""
    net, worst = train()
    torch.save(net.state_dict(), WEIGHTS)
    print(f"saved {WEIGHTS}; worst |net - return_law| over 20k held-out "
          f"states = {worst:.4f}  (actuator limit A_MAX = {A_MAX})")


if __name__ == "__main__":
    main()


class CrownContracts:
    """Affine contracts over the TRAINED network, straight from CROWN.

    `compute_bounds(..., return_A=True)` yields the linear relaxation

        lA . s + lb  <=  net(s)  <=  uA . s + ub      for all s in the cell

    which is the contract shape verbatim. The previous pipeline concretised
    this to an interval at the contract boundary and discarded lA and uA. That
    is the information that makes a regulator provably converge: the Return
    step has to reach a home box AND come to rest, and a bound with no slope
    lets the solver hold the acceleration constant inside a cell, which no
    amount of splitting repairs quickly - zeroth-order cell error falls as
    O(M^(-1/4)) in four dimensions, so the cell count needed to beat the
    damping term near the origin is out of reach of an SMT solver.

    lA and uA are genuinely DIFFERENT hyperplanes here, unlike the analytic
    law's, and they carry cross-axis coefficients the analytic law does not
    have. Keeping them separate is why the contract interface takes four parts.
    """

    def __init__(self, net=None, batch=256):
        from auto_LiRPA import BoundedModule           # noqa: PLC0415

        self.net = net if net is not None else load()
        self.batch = batch
        self.bounded = BoundedModule(self.net, torch.zeros(1, STATE_DIM),
                                     device="cpu")
        self.out_name = self.bounded.output_name[0]
        self.in_name = self.bounded.input_name[0]

    def __call__(self, cells):
        """Contracts for a list of cells, in the shape `affine_contract` returns."""
        out = []
        for start in range(0, len(cells), self.batch):
            out.extend(self._chunk(cells[start:start + self.batch]))
        return out

    def _chunk(self, cells):
        from auto_LiRPA import BoundedTensor            # noqa: PLC0415
        from auto_LiRPA.perturbations import PerturbationLpNorm  # noqa: PLC0415

        lows = torch.tensor([[c[0], c[2], c[4], c[6]] for c in cells],
                            dtype=torch.float32)
        highs = torch.tensor([[c[1], c[3], c[5], c[7]] for c in cells],
                             dtype=torch.float32)
        centre = BoundedTensor(0.5 * (lows + highs),
                               PerturbationLpNorm(x_L=lows, x_U=highs))
        _, _, adict = self.bounded.compute_bounds(
            x=(centre,), method="CROWN", return_A=True,
            needed_A_dict={self.out_name: [self.in_name]})
        entry = adict[self.out_name][self.in_name]
        low_a, up_a = entry["lA"], entry["uA"]
        low_b, up_b = entry["lbias"], entry["ubias"]
        result = []
        for i, cell in enumerate(cells):
            contract = {}
            for axis in (0, 1):
                contract[axis] = (low_a[i, axis].tolist(),
                                  float(low_b[i, axis]),
                                  up_a[i, axis].tolist(),
                                  float(up_b[i, axis]))
            result.append((cell, coarsen(cell, contract)))
        return result
