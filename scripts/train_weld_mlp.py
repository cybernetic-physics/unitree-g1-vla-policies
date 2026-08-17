#!/usr/bin/env python3
"""Train the g1_weld_approach learned-mlp policy (repo-side tooling, NOT part of the task pack).

Distills the known-good obstacle-relative controller (apex = obstacle_top_y + 12,
top_halfwidth 30 cm, speed 0.12 m/s) into a tiny [2, 8, 3] tanh MLP with a linear skip
connection — the exact architecture the SDK's ``learned-mlp`` backend decodes and the task
planner executes:

    h   = tanh(W1ᵀ f + b1)
    out = W2ᵀ h + b2 + Wskipᵀ f + bskip        # [apex_cm, top_halfwidth_cm, speed_mps]

Features (feature_spec weld-geometry/v1): f = [obstacle_top_y / 100, seam_x / 300].
Training data samples obstacle_offset in [6, 70] with BOTH +y-face variants the task
geometry produces (offset + 6 normal, offset + 24 collision-stress) and seam_x in
{120, 300}, exactly as tasks/g1_weld_approach/task.py builds observations.

Plain full-batch gradient descent, deterministic seed. Prints final losses and the
base64(little-endian float32) weight strings for two artifacts:

  * trained      -> policies/g1_weld_approach_v24.pt              (must pass 16/16)
  * undertrained -> policies/g1_weld_approach_v24_undertrained.pt (must fail geometrically)

Usage: python scripts/train_weld_mlp.py
"""

from __future__ import annotations

import base64
import json

import numpy as np

# Task geometry constants (mirrors tasks/g1_weld_approach/task.py; kept in sync by eye —
# this script is training tooling, not the judge).
OBS_POS_BASE = 6.0
OBS_POS_COLLISION = 24.0
SEAM_X_NORMAL = 120.0
SEAM_X_TIMEOUT = 300.0

# The distilled controller (the v21 scripted solution).
APEX_MARGIN_CM = 12.0
TOP_HALFWIDTH_CM = 30.0
SPEED_MPS = 0.12

HIDDEN = 8
SEED = 20260816
BAD_SEED = 13
UNDERTRAINED_ITERS = 5
ITERS = 60_000
LR = 0.05

# Per-output normalization so apex (tens of cm) does not drown speed (~0.1 m/s): the net
# is trained against y / OUT_SCALE, then the scales are folded EXACTLY into the linear
# output layer (W2, b2, Wskip, bskip are all linear in the output), so the exported net
# predicts raw [apex_cm, thw_cm, speed_mps] as output_spec trapezoid/v1 requires.
OUT_SCALE = np.array([50.0, 15.0, 0.1])


def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    offsets = np.linspace(6.0, 70.0, 65)
    rows_x, rows_y = [], []
    for off in offsets:
        for obs_pos in (OBS_POS_BASE, OBS_POS_COLLISION):
            for seam_x in (SEAM_X_NORMAL, SEAM_X_TIMEOUT):
                top_y = off + obs_pos
                rows_x.append([top_y / 100.0, seam_x / 300.0])
                rows_y.append([top_y + APEX_MARGIN_CM, TOP_HALFWIDTH_CM, SPEED_MPS])
    return np.array(rows_x), np.array(rows_y)


def init_params(rng: np.random.Generator, scale: float = 0.5) -> dict:
    return {
        "w1": rng.normal(0.0, scale, (2, HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "w2": rng.normal(0.0, scale, (HIDDEN, 3)),
        "b2": np.zeros(3),
        "wskip": rng.normal(0.0, scale, (2, 3)),
        "bskip": np.zeros(3),
    }


def forward(p: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.tanh(x @ p["w1"] + p["b1"])
    out = h @ p["w2"] + p["b2"] + x @ p["wskip"] + p["bskip"]
    return h, out


def train(p: dict, x: np.ndarray, y: np.ndarray, iters: int, lr: float) -> float:
    """Plain full-batch gradient descent on mean squared error against y / OUT_SCALE."""
    n = len(x)
    y_norm = y / OUT_SCALE
    loss = float("inf")
    for _ in range(iters):
        h, out = forward(p, x)
        err = out - y_norm
        loss = float(np.mean(err**2))
        dout = 2.0 * err / (n * 3)
        dh = dout @ p["w2"].T * (1.0 - h**2)
        p["w2"] -= lr * (h.T @ dout)
        p["b2"] -= lr * dout.sum(axis=0)
        p["wskip"] -= lr * (x.T @ dout)
        p["bskip"] -= lr * dout.sum(axis=0)
        p["w1"] -= lr * (x.T @ dh)
        p["b1"] -= lr * dh.sum(axis=0)
    return loss


def fold_out_scale(p: dict) -> dict:
    """Fold OUT_SCALE into the (linear) output layer so the net predicts raw units."""
    return {
        "w1": p["w1"].copy(),
        "b1": p["b1"].copy(),
        "w2": p["w2"] * OUT_SCALE,
        "b2": p["b2"] * OUT_SCALE,
        "wskip": p["wskip"] * OUT_SCALE,
        "bskip": p["bskip"] * OUT_SCALE,
    }


def pack_b64(p: dict) -> str:
    flat = np.concatenate(
        [
            p["w1"].ravel(),  # row-major 2xH
            p["b1"].ravel(),
            p["w2"].ravel(),  # row-major Hx3
            p["b2"].ravel(),
            p["wskip"].ravel(),  # row-major 2x3
            p["bskip"].ravel(),
        ]
    ).astype("<f4")
    return base64.b64encode(flat.tobytes()).decode("ascii")


def checkpoint_json(b64: str) -> str:
    return json.dumps(
        {
            "format": "mlp/f32-le/v1",
            "arch": [2, HIDDEN, 3],
            "weights_b64": b64,
            "feature_spec": "weld-geometry/v1",
            "output_spec": "trapezoid/v1",
        },
        indent=2,
    )


def report_worst(p: dict, x: np.ndarray, y: np.ndarray, label: str) -> None:
    _, out = forward(p, x)
    err = np.abs(out - y)
    print(
        f"  [{label}] max |apex err| = {err[:, 0].max():.4f} cm, "
        f"max |thw err| = {err[:, 1].max():.4f} cm, "
        f"max |speed err| = {err[:, 2].max():.5f} m/s"
    )


def main() -> None:
    x, y = make_dataset()
    print(f"dataset: {len(x)} samples (offset 6..70 cm x {{+6,+24}} face x seam {{120,300}})")

    trained_norm = init_params(np.random.default_rng(SEED))
    loss = train(trained_norm, x, y, ITERS, LR)
    trained = fold_out_scale(trained_norm)
    print(f"trained: {ITERS} iters, final normalized loss = {loss:.3e}")
    report_worst(trained, x, y, "trained")

    under_norm = init_params(np.random.default_rng(BAD_SEED), scale=1.5)
    uloss = train(under_norm, x, y, UNDERTRAINED_ITERS, LR)
    under = fold_out_scale(under_norm)
    print(f"undertrained: {UNDERTRAINED_ITERS} iters, final normalized loss = {uloss:.3e}")
    report_worst(under, x, y, "undertrained")

    print("\n--- checkpoint for policies/g1_weld_approach_v24.pt ---")
    print(checkpoint_json(pack_b64(trained)))
    print("\n--- checkpoint for policies/g1_weld_approach_v24_undertrained.pt ---")
    print(checkpoint_json(pack_b64(under)))


if __name__ == "__main__":
    main()
