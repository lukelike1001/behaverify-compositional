"""
Train the smallest honest learned controller for the geofence prototype.

The architecture is the one real systems actually use: a LEARNED inner-loop
controller wrapped in a HAND-WRITTEN safety envelope.  The network cruises
toward a setpoint; the behavior tree preempts it at the geofence.  Neither part
is verifiable alone - the network has no notion of the fence, and the tree has
no notion of what the network will do.

Kept deliberately tiny (2 -> 8 -> 8 -> 1 ReLU) so that CROWN bounds are cheap
and the prototype's difficulty comes from the closed loop rather than from
network size.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch import nn

DT = 0.1
SETPOINT = 0.0          # the controller tries to hold x = 0
V_MAX = 1.0
OUT = pathlib.Path("COLLAB/experiments/geofence")


class Controller(nn.Module):
    """(x, v) -> acceleration, bounded to [-1, 1] by tanh."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Tanh())

    def forward(self, state):
        """Acceleration command for a batch of states."""
        return self.net(state)


def expert(x, v):
    """A PD law the network imitates: pull toward the setpoint, damp velocity."""
    return np.clip(-1.2 * (x - SETPOINT) - 1.6 * v, -1.0, 1.0)


def main():
    """Train by imitation and export ONNX."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-2.0, 2.0, 20000)
    vs = rng.uniform(-V_MAX, V_MAX, 20000)
    inputs = torch.tensor(np.stack([xs, vs], 1), dtype=torch.float32)
    targets = torch.tensor(expert(xs, vs)[:, None], dtype=torch.float32)

    model = Controller()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(600):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        opt.step()
        if epoch % 200 == 0:
            print(f"  epoch {epoch:>4}  mse {loss.item():.5f}")
    print(f"  final mse {loss.item():.5f}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cruise.onnx"
    torch.onnx.export(model, torch.zeros(1, 2), str(path),
                      input_names=["state"], output_names=["accel"],
                      dynamic_axes={"state": {0: "batch"}})
    torch.save(model.state_dict(), OUT / "cruise.pt")
    print(f"  exported {path} and cruise.pt")

    # sanity: does the closed loop without the fence actually settle?
    state = np.array([1.5, 0.0], np.float32)
    for _ in range(60):
        a = float(model(torch.tensor(state[None])).item())
        state = np.array([state[0] + state[1] * DT, state[1] + a * DT], np.float32)
    print(f"  closed loop from x=1.5 settles at x={state[0]:.3f}, v={state[1]:.3f}")


if __name__ == "__main__":
    main()
