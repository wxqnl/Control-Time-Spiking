"""CPU probe of the spiking interface used throughout spike-driven-vla.

Reproduces the numbers in docs/cts/research_plan.md section 2.2. No weights are loaded.
The repo neuron is BipolarPLIF (turbovla/models/components/wta_operators.py):
    v = u + a * (I - u);  s = +1 if v >= th, -1 if v <= -th, else 0;  hard reset u = v * (1 - |s|)
It is driven by a constant input I for T steps from a zero membrane and read out as mean(s),
which is how every projection / fusion / head module uses it per control query.
Note the charging form: the membrane is an exponential moving average that converges to I,
so it never accumulates evidence; a -> 1 makes it memoryless (v = I).
The integrate-and-fire (IF) reference accumulates instead: v = u + I, with subtractive (soft) reset.
Initial repo values are a = sigmoid(0) = 0.5, th = 0.5; learned values come from the checkpoint (E0a).
"""
import numpy as np


def run(I, T=4, a=0.5, th=0.5, reset="hard", mode="plif"):
    u = np.zeros_like(I)
    out = np.zeros_like(I)
    for _ in range(T):
        v = u + a * (I - u) if mode == "plif" else u + I
        s = (v >= th).astype(float) - (v <= -th).astype(float)
        u = v * (1 - np.abs(s)) if reset == "hard" else v - th * s
        out += s
    return out / T


def affine_rmse(y, x):
    A = np.stack([y, np.ones_like(y)], 1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(np.sqrt(np.mean((A @ c - x) ** 2)))


def best_rmse(z, **kw):
    """Decoding error after choosing the best threshold, i.e. the scale is not the bottleneck."""
    return min(affine_rmse(run(z, th=th, **kw), z) for th in np.geomspace(0.02, 5, 60))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    grid = np.linspace(-4, 4, 400001)
    z = rng.standard_normal(200_000)  # a BN/LN-normalized channel

    print("distinct readout levels vs T")
    for name, kw in [("repo PLIF a=.5, hard", {}), ("IF, soft           ", dict(mode="if", reset="soft"))]:
        print(f"  {name}: " + ", ".join(f"T={T}:{len(np.unique(run(grid, T=T, **kw)))}" for T in (1, 2, 4, 8, 16)))

    y = run(grid)
    dead = grid[y == 0]
    print(f"repo neuron at init, T=4: dead zone [{dead.min():.3f}, {dead.max():.3f}], "
          f"silences {np.mean(run(z) == 0):.1%} of N(0,1) inputs")

    print("best-threshold affine decoding RMSE on N(0,1) (lower is better)")
    for a in (0.1, 0.3, 0.5, 0.7, 0.9):
        print(f"  repo PLIF a={a}, hard: " + "  ".join(
            f"T={T}:{best_rmse(z, T=T, a=a):.3f}" for T in (2, 4, 8, 16)))
    for reset in ("hard", "soft"):
        print(f"  IF, {reset} reset:        " + "  ".join(
            f"T={T}:{best_rmse(z, T=T, mode='if', reset=reset):.3f}" for T in (2, 4, 8, 16)))
