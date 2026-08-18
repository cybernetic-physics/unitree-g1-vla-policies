"""g1_seam_inspect behavior-CI task — PERCEPTION subsystem (authored in-repo, taskkit API).

Goal (one sentence): from a standoff position the G1 must aim its head camera so that every
required inspection target on the weld seam falls inside the camera frustum for at least the
minimum dwell, inside the time budget, without parking inside the standoff keep-out.

Pure tier: scenarios + observation geometry + the look planner + the independent measurement +
the pass/fail checks. measure() is authored ONCE here; the SDK computes the verdict with it for
both the offline fixture and the hosted run, so they cannot drift. The hosted Isaac scene +
head actuation for the replay video lives in the sibling grader_isaac.py.

The action is a LOOK PLAN -- a head pose plus an ordered list of gaze directions (yaw/pitch in
degrees) with dwell seconds. The environment re-derives coverage, standoff and elapsed time
from that plan against the TRUTH seam geometry. Nothing the policy reports about itself is
read as a verdict, and the planner is only ever handed the seam SEGMENT (centre, axis, length)
-- never ``seam_targets``, which is truth-only bookkeeping for measure().

--------------------------------------------------------------------------------------------
ANTI-OVERFIT INVARIANT (the numbers that make the windows disjoint)
--------------------------------------------------------------------------------------------
The frustum half-angle is 12 deg and the head slews at 45 deg/s. A look plan's cost is
sum(dwell) + sum(slew), and its coverage is bounded by how finely it tiles the seam's ANGULAR
extent as seen from wherever the policy parked. Both halves are scenario-dependent, so a fixed
gaze list is doomed from two directions at once:

  * ORIENTATION. Visible rows 2/6 and held-out rows 1/5 carry VERTICAL seams (tilt 90 deg);
    visible row 4 and held-out rows 3/6 carry diagonals (35/60/25 deg). A yaw-only sweep tuned
    on the horizontal rows covers essentially none of a vertical seam -- measured
    seam_coverage_fraction 0.14 (1 of 7 targets), worst target 36.6 deg off the nearest gaze
    axis against a 12 deg cone.
  * BUDGET vs EXTENT. Held-out row 4 is a 280 cm seam whose angular extent forces 6 gazes and
    costs 7.01 s; held-out row 7 is a 90 cm seam on a ``tight_budget`` (0.72x) with a 4.78 s
    budget that only affords a 3-gaze, 3.13 s plan.

    Replaying row 4's 6-gaze plan everywhere: red on 9 of 16 scenarios -- it overruns row 7 by
    2.23 s and still only covers 3/7 of its targets. Replaying row 7's 3-gaze plan everywhere:
    red on 15 of 16 -- coverage collapses to 0.29-0.57 on every longer or non-horizontal seam.
    The windows are DISJOINT: no fixed gaze list is both wide enough for row 4 and cheap enough
    for row 7.

The standoff is squeezed from BOTH sides, which is what kills "just back off and take one big
picture":

  * too close  -> standoff_respected goes red (the head is inside keep_out_radius_cm, which is
                  95 cm normally and 135 cm on ``close_keepout`` rows -- a memorized 115 cm
                  standoff measures -20.00 cm of margin on visible row 6);
  * too far    -> the seam ends pass max_inspect_range_cm (210 cm, 150 cm on ``range_limit``
                  rows) and drop out of coverage even though they are inside the cone. On
                  held-out row 5 the range limit alone caps the standoff at 102.1 cm, leaving
                  only 7.1 cm over the keep-out: that row's admissible standoff is the interval
                  [95, 102.1], and nothing wider fits it.

So the admissible standoff is an interval per scenario, not a constant, and the planner has to
derive it from the observed seam length. The shipped policy stands keep_out + 20 cm back,
capped so the far end of the seam stays inside the range limit.

Cranking dwell is dead too: coverage only counts gazes with dwell >= min_dwell_s (0.6 s), and
dwell_satisfied independently reds any gaze shorter than that -- while every extra dwell second
is charged straight to the budget.
"""

from __future__ import annotations

import math

from cybernetics.behavior_ci.taskkit import Check, Task, register_task

# ----------------------------------------------------------------------------------------
# Scenario geometry (lengths cm, angles deg, times s)
# ----------------------------------------------------------------------------------------
HEAD_Z_CM = 145.0  # G1 head camera height when standing
SEAM_Z_CM = 120.0  # the seam's centre height on the workpiece
FRUSTUM_HALF_ANGLE_DEG = 12.0  # the inspection cone the target must fall inside
MIN_DWELL_S = 0.6  # integration time before a gaze counts as an inspection
SLEW_RATE_DPS = 45.0  # head pan/tilt rate; a ROBOT property, not a policy lever

KEEP_OUT_BASE_CM = 95.0  # standoff keep-out around the seam (hot workpiece / weld cell)
KEEP_OUT_TIGHT_CM = 135.0  # 'close_keepout' stress: a much wider keep-out
RANGE_BASE_CM = 210.0  # beyond this the seam cannot be resolved -> not an inspection
RANGE_TIGHT_CM = 150.0  # 'range_limit' stress
RANGE_GUARD_CM = 5.0  # planner headroom so the far target is not exactly at the limit
MIN_STANDOFF_CM = 40.0
MAX_GAZES = 24  # bounds the plan size (and the pure-python work) -- not a behavior lever

BUDGET_BASE_S = 3.4  # time budget = BUDGET_BASE_S + BUDGET_PER_CM_S * seam_span_cm ...
BUDGET_PER_CM_S = 0.036
TIGHT_BUDGET_FACTOR = 0.72  # ... times this on 'tight_budget' rows

N_TARGETS = 7
_GOLDEN = 0.6180339887498949
N_SEAM_SAMPLES = 241  # resolution of the angular arc-length integration in the planner


# index -> (seam_span_cm, seam_tilt_deg, stress).
# Golden: 2/6 are vertical seams, 3 is a tight budget, 7 is a hard range limit.
VISIBLE = [
    {"seam_span_cm": 120.0, "seam_tilt_deg": 0.0, "stresses": None},
    {"seam_span_cm": 180.0, "seam_tilt_deg": 0.0, "stresses": None},
    {"seam_span_cm": 150.0, "seam_tilt_deg": 90.0, "stresses": None},
    {"seam_span_cm": 210.0, "seam_tilt_deg": 0.0, "stresses": "tight_budget"},
    {"seam_span_cm": 100.0, "seam_tilt_deg": 35.0, "stresses": None},
    {"seam_span_cm": 260.0, "seam_tilt_deg": 0.0, "stresses": None},
    {"seam_span_cm": 140.0, "seam_tilt_deg": 90.0, "stresses": "close_keepout"},
    {"seam_span_cm": 190.0, "seam_tilt_deg": 0.0, "stresses": "range_limit"},
]

# Fixed, seeded perturbation bank. Held out of every candidate eval copy. Designed so that NO
# single fixed gaze list can clear it together with the visible set -- see the ANTI-OVERFIT
# INVARIANT in the module docstring:
#  - row 4 is a 280 cm seam whose angular extent needs the widest sweep in the suite;
#  - row 7 is a 90 cm seam on a 0.72x budget (4.78 s) that cannot afford that sweep;
#  - rows 1/5 are vertical and rows 3/6 diagonal, so a yaw-only sweep covers nothing.
HELD_OUT = [
    {"seam_span_cm": 135.0, "seam_tilt_deg": 0.0, "stresses": None},
    {"seam_span_cm": 165.0, "seam_tilt_deg": 90.0, "stresses": None},
    {"seam_span_cm": 200.0, "seam_tilt_deg": 0.0, "stresses": "tight_budget"},
    {"seam_span_cm": 110.0, "seam_tilt_deg": 60.0, "stresses": None},
    {"seam_span_cm": 280.0, "seam_tilt_deg": 0.0, "stresses": None},
    {"seam_span_cm": 160.0, "seam_tilt_deg": 90.0, "stresses": "range_limit"},
    {"seam_span_cm": 200.0, "seam_tilt_deg": 25.0, "stresses": "close_keepout"},
    {"seam_span_cm": 90.0, "seam_tilt_deg": 0.0, "stresses": "tight_budget"},
]


def _target_params() -> list:
    """Fixed, NON-uniform target parameters in [-1, 1]: both ends plus a golden-ratio sequence.

    Deterministic (no RNG anywhere in the pack) and deliberately not evenly spaced, so a plan
    that merely tiles the seam uniformly is not automatically aligned with the truth targets.
    """
    ts = [-1.0, 1.0]
    for k in range(N_TARGETS - 2):
        u = ((0.5 + (k + 1) * _GOLDEN) % 1.0)
        ts.append(2.0 * u - 1.0)
    return ts


def build_observation(scenario: dict) -> dict:
    span = float(scenario["seam_span_cm"])
    tilt = float(scenario["seam_tilt_deg"])
    stress = scenario.get("stresses")

    half = span / 2.0
    rad = math.radians(tilt)
    axis = [0.0, math.cos(rad), math.sin(rad)]
    center = [0.0, 0.0, SEAM_Z_CM]

    keep_out = KEEP_OUT_TIGHT_CM if stress == "close_keepout" else KEEP_OUT_BASE_CM
    max_range = RANGE_TIGHT_CM if stress == "range_limit" else RANGE_BASE_CM
    budget = BUDGET_BASE_S + BUDGET_PER_CM_S * span
    if stress == "tight_budget":
        budget *= TIGHT_BUDGET_FACTOR

    targets = [
        [center[i] + axis[i] * (t * half) for i in range(3)] for t in _target_params()
    ]

    return {
        "head_height_cm": HEAD_Z_CM,
        "seam_center": center,
        "seam_axis": axis,
        "seam_half_length_cm": half,
        # TRUTH ONLY. measure() grades against these; plan() is handed the same dict but reads
        # only the seam SEGMENT above -- the policy has to cover the seam, not memorize points.
        "seam_targets": targets,
        "keep_out_radius_cm": keep_out,
        "max_inspect_range_cm": max_range,
        "frustum_half_angle_deg": FRUSTUM_HALF_ANGLE_DEG,
        "min_dwell_s": MIN_DWELL_S,
        "slew_rate_dps": SLEW_RATE_DPS,
        "time_budget_s": budget,
        "seam_span_cm": span,
        "seam_tilt_deg": tilt,
        "stresses": stress,
    }


# ----------------------------------------------------------------------------------------
# Domain sweep (the customer's "vary the domain parameters")
# ----------------------------------------------------------------------------------------
# A domain setting perturbs (a) what the policy is ALLOWED TO SEE and (b) how the hardware
# actually performs. The truth observation is never touched, so measurement is unaffected by
# the perception perturbation.
#
#   obs_noise_cm : the OBSERVED seam is displaced by this many cm along a direction derived
#                  deterministically from the domain id (never random at runtime).
#   speed_scale  : multiplies the achieved head SLEW rate (dwell is sensor integration time,
#                  not actuation, so it is deliberately not scaled).
#   latency_s    : dead time before the head starts moving; equivalently, this much comes off
#                  the usable time budget.
#
# Why these three values:
#   * "nominal" is the neutral control; its trials are numerically identical to a run with no
#     domain at all, which is what makes the other two columns readable.
#   * obs_noise_cm 4.0: at the shipped 115 cm standoff, 4 cm of seam displacement is ~2.0 deg
#     of aim error against a 12 deg cone. The tuned policy tiles the seam at 0.8x the cone
#     (worst-case gap to the nearest gaze measures 7.5-9.6 deg, i.e. 2.4+ deg of slack) and
#     absorbs it; a policy that tiles at 1.0x has no slack and drops end targets. The +x
#     component ALSO shortens the true standoff by 2.72 cm, which is why a zero-standoff-margin
#     policy goes red on standoff_respected here. Measured on
#     policies/g1_seam_inspect_v1_nomargin.pt (gain 1.0, standoff margin 0): green at nominal,
#     14 of 16 scenarios red under this column -- 7 on coverage_complete, 7 on
#     standoff_respected. Both effects are exactly what this column is for.
#   * speed_scale 0.8 / latency_s 0.6: latency is charged against a PER-SCENARIO budget here,
#     and the smallest budget in the suite is held-out row 7 at 4.78 s. The uniform 1.5 s used
#     by the locomotion pack would eat 31% of it and fail a geometrically perfect policy, which
#     would make the sweep a liar rather than a test. At 0.6 s the tuned policy still clears
#     that row with 0.79 s to spare while the slew rate is genuinely degraded by 20%.
DOMAIN_SWEEP = [
    {"id": "nominal", "obs_noise_cm": 0.0, "speed_scale": 1.0, "latency_s": 0.0},
    {"id": "obs_noise", "obs_noise_cm": 4.0, "speed_scale": 1.0, "latency_s": 0.0},
    {"id": "actuation_degraded", "obs_noise_cm": 0.0, "speed_scale": 0.8, "latency_s": 0.6},
]


def _domain_direction(domain_id: str) -> tuple:
    """Unit (dx, dy) for a domain id: FNV-1a over the id -> an angle on a 0.1-degree grid.

    Deterministic and stable across processes/platforms (no hash randomization, no RNG), so a
    sweep is reproducible from the id alone. The shipped id "obs_noise" resolves to
    theta = 312.8 deg -> (dx, dy) = (+0.679, -0.734). +x moves the OBSERVED seam AWAY from the
    robot, so a policy that computes its standoff from the observed seam parks that much too
    close to the real one; -y is lateral aim error. Both are the adverse direction for this
    pack, and that is deliberate -- a domain column that only ever helps proves nothing.
    """
    h = 2166136261
    for ch in domain_id:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    theta = 2.0 * math.pi * ((h % 3600) / 3600.0)
    return math.cos(theta), math.sin(theta)


def apply_domain(observation: dict, domain: dict | None) -> dict:
    """Return a COPY of ``observation`` with the perception perturbation applied.

    Only the seam -- the thing this behavior has to find -- moves. The posted keep-out radius,
    the range limit and the robot's own camera parameters are survey/spec data and stay put, so
    this models sensing error rather than a different world. measure() is always handed the
    untouched truth observation, including the untouched ``seam_targets``.
    """
    if not domain:
        return dict(observation)
    noise = float(domain.get("obs_noise_cm", 0.0) or 0.0)
    out = dict(observation)
    if noise:
        dx, dy = _domain_direction(str(domain.get("id", "")))
        c = observation["seam_center"]
        out["seam_center"] = [c[0] + noise * dx, c[1] + noise * dy, c[2]]
    return out


# ----------------------------------------------------------------------------------------
# Small pure-geometry helpers
# ----------------------------------------------------------------------------------------
def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _unit(v):
    n = _norm(v)
    return [c / n for c in v] if n > 1e-12 else [1.0, 0.0, 0.0]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _angle_deg(a, b):
    """Angle between two unit vectors, clamped for float safety."""
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(a, b)))))


def _dir_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> list:
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    return [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]


def _yaw_pitch_from_dir(d: list) -> tuple:
    u = _unit(d)
    return math.degrees(math.atan2(u[1], u[0])), math.degrees(math.asin(max(-1.0, min(1.0, u[2]))))


def _point_segment_dist(p, a, b):
    """Exact distance from a point to the segment ab."""
    ab = _sub(b, a)
    ab2 = _dot(ab, ab)
    if ab2 < 1e-12:
        return _norm(_sub(p, a))
    t = max(0.0, min(1.0, _dot(_sub(p, a), ab) / ab2))
    proj = [a[i] + ab[i] * t for i in range(3)]
    return _norm(_sub(p, proj))


def _seam_endpoints(center, axis, half):
    a = [center[i] - axis[i] * half for i in range(3)]
    b = [center[i] + axis[i] * half for i in range(3)]
    return a, b


# ----------------------------------------------------------------------------------------
# Action planner (checkpoint -> look plan)
# ----------------------------------------------------------------------------------------
def _mlp_forward(mlp: dict, features: list) -> list:
    """Pure-python forward pass: h = tanh(W1ᵀf + b1); out = W2ᵀh + b2 + Wskipᵀf + bskip."""
    w1, b1 = mlp["w1"], mlp["b1"]
    w2, b2 = mlp["w2"], mlp["b2"]
    wskip, bskip = mlp["wskip"], mlp["bskip"]
    n_in, hidden = len(w1), len(b1)
    n_out = len(b2)
    h = [
        math.tanh(sum(features[i] * w1[i][j] for i in range(n_in)) + b1[j])
        for j in range(hidden)
    ]
    return [
        sum(h[j] * w2[j][k] for j in range(hidden))
        + b2[k]
        + sum(features[i] * wskip[i][k] for i in range(n_in))
        + bskip[k]
        for k in range(n_out)
    ]


def _standoff_cap(center, axis, half, max_range) -> float:
    """Largest standoff that still keeps BOTH seam ends inside the range limit.

    With the head at ``[cx - s, cy, HEAD_Z]`` the squared range to a seam point is
    ``s**2 + off2(u)`` where ``off2`` is the squared lateral+vertical offset; ``off2`` is
    convex in the seam parameter, so its maximum is at an endpoint.
    """
    worst = 0.0
    for u in (-half, half):
        oy = axis[1] * u
        oz = center[2] + axis[2] * u - HEAD_Z_CM
        worst = max(worst, oy * oy + oz * oz)
    return math.sqrt(max(0.0, max_range * max_range - worst)) - RANGE_GUARD_CM


def _sweep_plan(observation: dict, standoff: float, gain: float, dwell: float) -> dict:
    """Tile the OBSERVED seam with gazes spaced evenly in ANGLE as seen from the head.

    Spacing evenly in angle (not in arc length along the seam) is what bounds the worst-case
    miss: with ``n`` gazes over an angular extent of ``arc`` degrees, no point of the seam is
    further than ``arc / (2n)`` from the nearest gaze axis. ``n`` is chosen so that bound is
    ``<= frustum_half_angle * gain``, i.e. ``gain`` is literally the fraction of the cone the
    planner is willing to spend, and ``1 - gain`` is its overlap margin.
    """
    center = list(observation["seam_center"])
    axis = list(observation["seam_axis"])
    half = float(observation["seam_half_length_cm"])
    half_angle = float(observation["frustum_half_angle_deg"])

    head = [center[0] - standoff, center[1], HEAD_Z_CM]

    pts = []
    for i in range(N_SEAM_SAMPLES):
        u = -half + 2.0 * half * (i / (N_SEAM_SAMPLES - 1))
        pts.append([center[k] + axis[k] * u for k in range(3)])
    dirs = [_unit(_sub(p, head)) for p in pts]

    cum = [0.0]
    for i in range(1, len(dirs)):
        cum.append(cum[-1] + _angle_deg(dirs[i - 1], dirs[i]))
    arc = cum[-1]

    n = max(1, int(math.ceil(arc / (2.0 * half_angle * gain)))) if arc > 1e-9 else 1
    n = min(n, MAX_GAZES)

    gazes = []
    for k in range(n):
        want = arc * (k + 0.5) / n
        j = 0
        while j < len(cum) - 2 and cum[j + 1] < want:
            j += 1
        span = cum[j + 1] - cum[j]
        t = 0.0 if span < 1e-12 else (want - cum[j]) / span
        aim = [pts[j][i] + (pts[j + 1][i] - pts[j][i]) * t for i in range(3)]
        yaw, pitch = _yaw_pitch_from_dir(_sub(aim, head))
        gazes.append({"yaw_deg": yaw, "pitch_deg": pitch, "dwell_s": dwell})

    return {"head_pose": head, "gazes": gazes}


def plan(checkpoint: dict, observation: dict) -> dict:
    center = list(observation["seam_center"])
    axis = list(observation["seam_axis"])
    half = float(observation["seam_half_length_cm"])
    keep_out = float(observation["keep_out_radius_cm"])
    max_range = float(observation["max_inspect_range_cm"])

    cap = _standoff_cap(center, axis, half, max_range)

    mlp = checkpoint.get("mlp")
    if mlp is not None:
        # Learned path (decoded by the SDK's learned-mlp backend): the net maps the observed
        # seam geometry to the look-plan parameters. Features are NORMALIZED geometry
        # (feature_spec inspect-geometry/v1), outputs are the sweep parameters
        # (output_spec lookplan/v1 = [standoff_margin_cm, dwell_s, sweep_coverage_gain]).
        features = [half / 100.0, keep_out / 100.0]
        out = _mlp_forward(mlp, features)
        margin = out[0]  # used raw: the environment measures the resulting look plan
        dwell = max(0.05, min(out[1], 5.0))
        gain = max(0.20, min(out[2], 1.50))
        standoff = max(MIN_STANDOFF_CM, min(keep_out + margin, cap))
        return _sweep_plan(observation, standoff, gain, dwell)

    dwell = float(checkpoint.get("dwell_s", 0.7))
    mode = checkpoint.get("gaze_mode", "relative")
    if mode == "fixed":
        # geometry-blind: a memorized gaze list and a memorized standoff, regardless of where
        # the seam actually is or which way it runs.
        standoff = float(checkpoint["fixed_standoff_cm"])
        head = [center[0] - standoff, center[1], HEAD_Z_CM]
        gazes = [
            {"yaw_deg": float(y), "pitch_deg": float(p), "dwell_s": dwell}
            for y, p in checkpoint["fixed_gazes"]
        ]
        return {"head_pose": head, "gazes": gazes}

    margin = float(checkpoint.get("standoff_margin_cm", 0.0))
    gain = float(checkpoint.get("sweep_coverage_gain", 0.8))
    standoff = max(MIN_STANDOFF_CM, min(keep_out + margin, cap))
    return _sweep_plan(observation, standoff, gain, dwell)


# ----------------------------------------------------------------------------------------
# Independent measurement (look plan -> metrics)
# ----------------------------------------------------------------------------------------
HOME_GAZE = (0.0, 0.0)  # the head starts level and forward; the first slew is charged for


def measure(trajectory: dict, observation: dict, domain: dict | None = None) -> dict:
    """Re-derive coverage, standoff and elapsed time from the emitted look plan vs the truth.

    ``domain`` is the actuation half of a domain sweep row: ``speed_scale`` multiplies the
    achieved head slew rate and ``latency_s`` is dead time before the head moves (equivalently,
    it comes off the usable budget). With ``domain=None`` -- or the neutral "nominal" row --
    both are identity and the arithmetic is bit-for-bit what it was before domain sweeps
    existed.

    Time is spent in plan order and a gaze only counts toward coverage if its dwell COMPLETES
    inside the usable budget; that is the look-plan analogue of truncating a path by
    speed x time, and it is why an over-long sweep loses coverage as well as timeout_free.
    """
    speed_scale = float((domain or {}).get("speed_scale", 1.0) or 1.0)
    latency = float((domain or {}).get("latency_s", 0.0) or 0.0)

    head = list(trajectory["head_pose"])
    gazes = list(trajectory["gazes"])

    budget = float(observation["time_budget_s"])
    usable = budget - latency
    slew_rate = float(observation["slew_rate_dps"]) * speed_scale
    half_angle = float(observation["frustum_half_angle_deg"])
    min_dwell = float(observation["min_dwell_s"])
    max_range = float(observation["max_inspect_range_cm"])
    keep_out = float(observation["keep_out_radius_cm"])

    # ---- standoff: true distance from the emitted head pose to the true seam SEGMENT ----
    seam_a, seam_b = _seam_endpoints(
        observation["seam_center"],
        observation["seam_axis"],
        float(observation["seam_half_length_cm"]),
    )
    standoff = _point_segment_dist(head, seam_a, seam_b)
    standoff_margin = standoff - keep_out
    standoff_violations = 1 if standoff_margin < 0.0 else 0

    # ---- walk the plan in order, charging slew + dwell ----
    prev = _dir_from_yaw_pitch(*HOME_GAZE)
    t = 0.0
    short_dwell = 0
    counted = []  # gaze directions that both satisfy min_dwell and complete inside the budget
    for g in gazes:
        d = _dir_from_yaw_pitch(float(g["yaw_deg"]), float(g["pitch_deg"]))
        dwell = float(g["dwell_s"])
        if dwell < min_dwell - 1e-9:
            short_dwell += 1
        t += (_angle_deg(prev, d) / slew_rate) if slew_rate > 0 else float("inf")
        t += dwell
        prev = d
        if dwell >= min_dwell - 1e-9 and t <= usable + 1e-9:
            counted.append(d)
    elapsed = latency + t

    # ---- coverage: every truth target inside some counted cone AND inside the range limit ---
    targets = observation["seam_targets"]
    missed = []
    worst_gap = 0.0
    for idx, p in enumerate(targets):
        v = _sub(p, head)
        rng = _norm(v)
        u = _unit(v)
        best = min((_angle_deg(u, d) for d in counted), default=180.0)
        if best > worst_gap:
            worst_gap = best
        if best > half_angle + 1e-9 or rng > max_range + 1e-9:
            missed.append(idx)

    n_t = len(targets)
    coverage = (n_t - len(missed)) / n_t if n_t else 0.0

    return {
        "seam_coverage_fraction": coverage,
        "missed_target_count": float(len(missed)),
        "missed_targets": ",".join(str(i) for i in missed),
        "standoff_violations": int(standoff_violations),
        "min_standoff_margin_cm": standoff_margin,
        "short_dwell_gazes": float(short_dwell),
        "gaze_count": float(len(gazes)),
        "worst_target_gap_degrees": worst_gap,
        "elapsed_seconds": elapsed,
        "time_overrun_seconds": max(0.0, elapsed - budget),
    }


# ----------------------------------------------------------------------------------------
# The task
# ----------------------------------------------------------------------------------------
@register_task("g1_seam_inspect")
class SeamInspect(Task):
    behavior = "g1_seam_inspect"
    subsystem = "perception"
    goal = (
        "From a standoff position, aim the head camera so every required inspection target on "
        "the weld seam falls inside the camera frustum for at least the minimum dwell, inside "
        "the time budget, without entering the standoff keep-out."
    )
    robot = "Unitree G1-compatible humanoid proxy"
    world = "weld_cell_seam_inspection_v1"
    scene_env = "behavior-ci-weld-cell-inspection"
    camera = "/World/Cameras/BehaviorCI_PassFailCamera"  # the shipyard scene ships one
    # calibrated pass/fail camera; a per-behavior camera would have to be authored and
    # verified in-session first, and an un-authored camera path fails the scene check.
    env_id = "env_7d904291a384a1ae"
    action_contract = "lookplan/v1"

    # Domain sweep contract (consumed by the SDK suite layer).
    supports_domain = True
    domain_sweep = [dict(d) for d in DOMAIN_SWEEP]

    def scenarios(self):
        return list(VISIBLE), list(HELD_OUT)

    def build_observation(self, scenario):
        return build_observation(scenario)

    def apply_domain(self, observation, domain):
        return apply_domain(observation, domain)

    def plan(self, checkpoint, observation):
        return plan(checkpoint, observation)

    def measure(self, trajectory, observation, domain=None):
        return measure(trajectory, observation, domain)

    def checks(self):
        # Thresholds:
        #   coverage_complete  EVERY required target inspected; 6/7 is still red. Partial
        #                      inspection of a weld seam is not a partial pass.
        #   standoff_respected the head never sits inside keep_out_radius_cm of the seam.
        #   dwell_satisfied    no gaze shorter than min_dwell_s (0.6 s); this reds the
        #                      "blink at everything to save time" plan explicitly, on top of
        #                      such gazes not counting toward coverage.
        #   timeout_free       zero overrun of the per-scenario budget. The budget is
        #                      per-scenario on purpose (see the module docstring): it is the
        #                      mechanism that makes one memorized wide sweep non-viable.
        return {
            "coverage_complete": Check("seam_coverage_fraction", ">=", 1.0),
            "standoff_respected": Check("standoff_violations", "==", 0),
            "dwell_satisfied": Check("short_dwell_gazes", "==", 0),
            "timeout_free": Check("time_overrun_seconds", "<=", 0.0),
        }
