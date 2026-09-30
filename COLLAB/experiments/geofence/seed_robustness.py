"""
Is the result an artifact of one trained network?

A benchmark whose verdict depends on a lucky random seed is not a benchmark.
This retrains the controller from several seeds and re-runs the head-to-head on
each, checking that the ordering - compositional proves, reachability does not -
is a property of the SYSTEM rather than of one set of weights.

Each seed gets its own safety check too, because a differently-trained
controller may simply be unsafe, in which case reachability reporting a
violation is correct rather than spurious and the comparison is void.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import z3
from torch import nn

from compare import compositional, crown_cells, per_step_box, X0_HI, X0_LO
from nsbt import DT, HARD_LIMIT, Tree
from train_controller import Controller, expert

STEPS, V0, GRID = 20, 0.8, 12


def train(seed):
    """Train a controller from one seed and return it."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-2.0, 2.0, 20000)
    vs = rng.uniform(-1.0, 1.0, 20000)
    inputs = torch.tensor(np.stack([xs, vs], 1), dtype=torch.float32)
    targets = torch.tensor(expert(xs, vs)[:, None], dtype=torch.float32)
    model = Controller()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad()
        nn.functional.mse_loss(model(inputs), targets).backward()
        opt.step()
    model.eval()
    return model


def true_max(model):
    """Largest x the TRUE closed loop reaches, so we know if safety even holds."""
    tree = Tree()

    def tick(x, v):
        with torch.no_grad():
            a = float(model(torch.tensor([[x, v]], dtype=torch.float32)).item())
        return a
    worst = -np.inf
    for x0 in np.linspace(X0_LO, X0_HI, 30):
        for v0 in np.linspace(-V0, V0, 17):
            x, v = float(x0), float(v0)
            for _ in range(120):
                a = -1.0 if x >= 1.0 else tick(x, v)
                worst = max(worst, x)
                x, v = x + v * DT, v + a * DT
    del tree
    return worst


def main():
    """Retrain from several seeds and check the ordering is stable."""
    print(f"{'seed':>5} {'true max x':>11} {'per-step':>10} {'compositional':>14}")
    for seed in (0, 1, 2, 3):
        model = train(seed)
        path = f"/tmp/geofence_seed{seed}.pt"
        torch.save(model.state_dict(), path)
        cells = crown_cells(GRID, model=model)
        tmax = true_max(model)
        verdict, _ = per_step_box(cells, STEPS, V0)
        res, _ = compositional(cells, STEPS, V0)
        comp = "safe" if res == z3.unsat else "violated" if res == z3.sat else "?"
        flag = "" if tmax <= HARD_LIMIT else "   (TRUE SYSTEM UNSAFE - void)"
        print(f"{seed:>5} {tmax:>11.4f} {verdict:>10} {comp:>14}{flag}")


if __name__ == "__main__":
    main()
