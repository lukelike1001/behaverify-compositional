"""
How close does the TRUE NAV trajectory come to the obstacle [1,2]^2?

This measures a property of the BENCHMARK, not of our abstraction: if the
nominal trajectory grazes the obstacle corner, then no abstraction with
nonzero error can prove obstacle avoidance, however fine the grid.
"""
import itertools
import numpy as np
import onnxruntime as ort

NET = "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx"
DT = 0.2
STEPS = 30


def dist_to_box(x, y, box=((1.0, 2.0), (1.0, 2.0))):
    """Euclidean distance from (x,y) to an axis-aligned box; 0.0 if inside."""
    (xlo, xhi), (ylo, yhi) = box
    dx = max(xlo - x, 0.0, x - xhi)
    dy = max(ylo - y, 0.0, y - yhi)
    return float(np.hypot(dx, dy))


def main():
    """Report the closest approach of any true trajectory to the obstacle."""
    sess = ort.InferenceSession(NET)
    inp = sess.get_inputs()[0].name
    worst = (1e9, None)
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 5), repeat=2):
        state = np.array([x0, y0, 0.0, 0.0], np.float32)
        for k in range(STEPS):
            u = sess.run(None, {inp: state.reshape(1, 4)})[0].ravel()
            d = dist_to_box(state[0], state[1])
            if d < worst[0]:
                worst = (d, (k, float(state[0]), float(state[1]), x0, y0))
            state = state + DT * np.array(
                [state[2] * np.cos(state[3]), state[2] * np.sin(state[3]),
                 u[0], u[1]], np.float32)
    d, (k, x, y, x0, y0) = worst
    print(f"MIN distance from true trajectory to obstacle = {d:.4f}")
    print(f"  at step k={k}, position ({x:.3f},{y:.3f}), start ({x0:.2f},{y0:.2f})")


if __name__ == "__main__":
    main()
