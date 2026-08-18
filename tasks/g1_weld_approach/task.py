"""g1_weld_approach behavior-CI task (authored in-repo against the SDK taskkit API).

Pure tier: scenarios + observation geometry + the action planner + the independent measurement
+ the pass/fail checks. measure() is authored ONCE here; the SDK computes the verdict with it
for both the offline fixture and the hosted run, so they cannot drift. The hosted Isaac scene +
actuation for the replay video lives in the sibling grader_isaac.py.
"""

from __future__ import annotations

import math

from cybernetics.behavior_ci.taskkit import Check, Task, register_task

# ----------------------------------------------------------------------------------------
# Scenario geometry
# ----------------------------------------------------------------------------------------
OBS_NEG = 6.0  # how far the obstacle pokes into -y (straddles the direct line)
OBS_POS_BASE = 6.0  # +y face = offset + this, for ordinary obstacles
OBS_POS_COLLISION = (
    24.0  # 'collision' stress: far +y face so an under-detour hits the box
)
OBS_HALF_X = 12.0
OBS_Z_HI = 25.0
ZONE_W = 8.0  # thickness of the +y restricted-zone band ('safety_zone' stress)
ZONE_HALF_X = 18.0
CEIL_LO = 48.0  # 'ceiling' stress: restricted zone occupies y >= CEIL_LO (overhead)
CEIL_HI = 400.0  # tall enough that any over-detour shelf rises into it
CEIL_HALF_X = 70.0  # wide enough in x to cover the whole detour shelf
TORCH_Z = 5.0
SEAM_X_NORMAL = 120.0
SEAM_X_TIMEOUT = 300.0
TIME_BUDGET_S = 30.0


# index -> (obstacle_offset_cm, stress).  Golden: 3->safety_zone, 5->collision, 7->timeout.
VISIBLE = [
    {"obstacle_offset_cm": 8.0, "stresses": None},
    {"obstacle_offset_cm": 12.0, "stresses": None},
    {"obstacle_offset_cm": 16.0, "stresses": None},
    {"obstacle_offset_cm": 20.0, "stresses": "safety_zone"},
    {"obstacle_offset_cm": 18.0, "stresses": None},
    {"obstacle_offset_cm": 24.0, "stresses": "collision"},
    {"obstacle_offset_cm": 22.0, "stresses": None},
    {"obstacle_offset_cm": 34.0, "stresses": "timeout"},
]

# Fixed, seeded perturbation bank. Held out of every candidate eval copy. Designed so that
# NO single constant (geometry-blind) apex can clear it together with the visible set:
#  - the 'ceiling' row forbids apex >= 48;
#  - the large-offset rows (50,58 -> +y faces 56,64) require apex >= 56 / 64.
# An obstacle-relative policy sizes the detour per observation and clears all of them.
HELD_OUT = [
    {"obstacle_offset_cm": 9.0, "stresses": None},
    {"obstacle_offset_cm": 15.0, "stresses": None},
    {"obstacle_offset_cm": 21.0, "stresses": "safety_zone"},
    {"obstacle_offset_cm": 27.0, "stresses": "collision"},
    {"obstacle_offset_cm": 20.0, "stresses": "timeout"},
    {"obstacle_offset_cm": 24.0, "stresses": "ceiling"},
    {"obstacle_offset_cm": 50.0, "stresses": None},
    {"obstacle_offset_cm": 58.0, "stresses": None},
]


def build_observation(scenario: dict) -> dict:
    off = float(scenario["obstacle_offset_cm"])
    stress = scenario.get("stresses")

    seam_x = SEAM_X_TIMEOUT if stress == "timeout" else SEAM_X_NORMAL
    obs_pos = OBS_POS_COLLISION if stress == "collision" else OBS_POS_BASE

    mx = seam_x / 2.0

    y_lo = -OBS_NEG
    y_hi = off + obs_pos
    obs_center = [mx, 0.5 * (y_lo + y_hi), OBS_Z_HI / 2.0]
    obs_half = [OBS_HALF_X, 0.5 * (y_hi - y_lo), OBS_Z_HI / 2.0]

    if stress == "safety_zone":
        # thin band hugging the obstacle's +y face: y in [y_hi, y_hi + ZONE_W]
        z_lo, z_hi = y_hi, y_hi + ZONE_W
        zone_center = [mx, 0.5 * (z_lo + z_hi), OBS_Z_HI / 2.0]
        zone_half = [ZONE_HALF_X, 0.5 * (z_hi - z_lo), OBS_Z_HI / 2.0]
    elif stress == "ceiling":
        # wide overhead band: y in [CEIL_LO, CEIL_HI]; an over-detour shelf rises into it.
        zone_center = [mx, 0.5 * (CEIL_LO + CEIL_HI), OBS_Z_HI / 2.0]
        zone_half = [CEIL_HALF_X, 0.5 * (CEIL_HI - CEIL_LO), OBS_Z_HI / 2.0]
    else:
        # behind the obstacle on the -y side, out of the +y detour corridor
        zy_hi = y_lo
        zy_lo = y_lo - 16.0
        zone_center = [mx, 0.5 * (zy_lo + zy_hi), OBS_Z_HI / 2.0]
        zone_half = [ZONE_HALF_X, 0.5 * (zy_hi - zy_lo), OBS_Z_HI / 2.0]

    return {
        "start_pose": [0.0, 0.0, TORCH_Z],
        "seam_pose": [seam_x, 0.0, TORCH_Z],
        "obstacle_box": {"center": obs_center, "half_extents": obs_half},
        "restricted_zone": {"center": zone_center, "half_extents": zone_half},
        "time_budget_s": TIME_BUDGET_S,
        "obstacle_offset_cm": off,
        "stresses": stress,
    }


# ----------------------------------------------------------------------------------------
# Domain sweep (the customer's "vary the domain parameters")
# ----------------------------------------------------------------------------------------
# A domain setting perturbs (a) what the policy is ALLOWED TO SEE and (b) how the hardware
# actually performs. The truth observation is never touched, so measurement is unaffected by
# the perception perturbation -- that asymmetry is the point: a policy that only just clears
# the geometry has nothing left when its perception is a couple of cm off.
#
#   obs_noise_cm : the OBSERVED obstacle is displaced by this many cm along a direction derived
#                  deterministically from the domain id (never random at runtime).
#   speed_scale  : multiplies the achieved torch travel speed.
#   latency_s    : dead time before the torch starts moving; equivalently, this much comes off
#                  the usable time budget.
#
# Why these three values -- this pack is the OLDEST behavior here and its shipped policies were
# tuned before domain sweeps existed, so the honest values are tighter than the newer packs':
#   * "nominal" is the neutral control; its trials are numerically identical to a run with no
#     domain at all, which is what makes the other two columns readable.
#   * obs_noise_cm 3.0: the binding geometry is the 'safety_zone' band, ZONE_W = 8 cm wide,
#     against v21/v24's clearance_margin_cm of 12 -- only 4 cm of true slack. The "obs_noise"
#     direction contributes -2.202 cm to the apex, leaving 1.8 cm. A 4 cm setting (what the
#     locomotion and perception packs use) would consume the whole margin and fail a policy
#     that is geometrically correct, which would make the sweep a liar rather than a test.
#   * speed_scale 0.95 / latency_s 0.5: the binding row is visible row 7 (seam at 300 cm),
#     which v21 covers in 26.80 s of a 30 s budget at its shipped 0.12 m/s -- 11% of headroom,
#     and the policy files are frozen red/green history that must not be re-tuned. Degraded,
#     that row measures 0.5 + 3.216/(0.12*0.95) = 28.71 s, still inside the budget. The
#     locomotion and perception packs were designed with headroom and carry 0.8 / 1.5 and
#     0.8 / 0.6 respectively; a uniform setting across packs would be cosmetic, not honest.
DOMAIN_SWEEP = [
    {"id": "nominal", "obs_noise_cm": 0.0, "speed_scale": 1.0, "latency_s": 0.0},
    {"id": "obs_noise", "obs_noise_cm": 3.0, "speed_scale": 1.0, "latency_s": 0.0},
    {"id": "actuation_degraded", "obs_noise_cm": 0.0, "speed_scale": 0.95, "latency_s": 0.5},
]


def _domain_direction(domain_id: str) -> tuple:
    """Unit (dx, dy) for a domain id: FNV-1a over the id -> an angle on a 0.1-degree grid.

    Deterministic and stable across processes/platforms (no hash randomization, no RNG), so a
    sweep is reproducible from the id alone. The shipped id "obs_noise" resolves to
    theta = 312.8 deg -> (dx, dy) = (+0.679, -0.734): the -y component pulls the OBSERVED
    obstacle face back toward the direct line, i.e. the policy under-sizes its detour. That is
    the adverse direction for this pack, and it is chosen on purpose -- a domain column that
    only ever helps the policy proves nothing.
    """
    h = 2166136261
    for ch in domain_id:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    theta = 2.0 * math.pi * ((h % 3600) / 3600.0)
    return math.cos(theta), math.sin(theta)


def apply_domain(observation: dict, domain: dict | None) -> dict:
    """Return a COPY of ``observation`` with the perception perturbation applied.

    Only the obstacle -- the thing this behavior has to route around -- moves. The seam pose,
    the start pose and the posted restricted zone are survey data and stay put, so this models
    sensing error rather than a different world (and in particular the 2 cm target_reach
    threshold is not quietly turned into a coin flip). measure() is always handed the untouched
    truth observation.
    """
    if not domain:
        return dict(observation)
    noise = float(domain.get("obs_noise_cm", 0.0) or 0.0)
    out = dict(observation)
    if noise:
        dx, dy = _domain_direction(str(domain.get("id", "")))
        box = observation["obstacle_box"]
        c = box["center"]
        out["obstacle_box"] = {
            "center": [c[0] + noise * dx, c[1] + noise * dy, c[2]],
            "half_extents": list(box["half_extents"]),
        }
    return out


# ----------------------------------------------------------------------------------------
# Action planner (checkpoint -> trajectory)
# ----------------------------------------------------------------------------------------
# Physical ceiling on torch travel speed (m/s). A real welding end-effector does not move
# at 100 m/s; clamping here stops "crank the speed" from trivially satisfying the timeout
# check while leaving every geometric safety check (collision/intrusion/tilt) unaffected.
MAX_SPEED_MPS = 2.0


def _trapezoid(start: list, seam: list, apex: float, thw: float, speed: float) -> dict:
    """Shared trapezoid construction: start -> P1 -> P2 -> seam with shelf height ``apex``."""
    speed = max(0.0, min(speed, MAX_SPEED_MPS))

    mx = 0.5 * (start[0] + seam[0])
    z = start[2]

    p1 = [mx - thw, apex, z]
    p2 = [mx + thw, apex, z]

    waypoints = [list(start), p1, p2, list(seam)]
    return {"waypoints": waypoints, "speed_mps": speed}


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


def plan(checkpoint: dict, observation: dict) -> dict:
    start = list(observation["start_pose"])
    seam = list(observation["seam_pose"])
    obs = observation["obstacle_box"]
    oc = obs["center"]
    oh = obs["half_extents"]

    # Far (+y) face of the obstacle as seen in the observation geometry.
    obstacle_top_y = oc[1] + oh[1]

    mlp = checkpoint.get("mlp")
    if mlp is not None:
        # Learned path (decoded by the SDK's learned-mlp backend): the net maps the
        # observed geometry to the trapezoid parameters. feature_spec weld-geometry/v1,
        # output_spec trapezoid/v1 = [apex_cm, top_halfwidth_cm, speed_mps].
        features = [obstacle_top_y / 100.0, seam[0] / 300.0]
        out = _mlp_forward(mlp, features)
        apex = out[0]  # used raw: the environment measures the resulting trajectory
        thw = max(15.0, min(out[1], 45.0))
        speed = max(0.05, min(out[2], 2.0))
        return _trapezoid(start, seam, apex, thw, speed)

    mode = checkpoint.get("detour_mode", "relative")
    if mode == "absolute":
        # geometry-blind: a fixed apex regardless of where the obstacle actually is.
        apex = float(checkpoint["absolute_apex_cm"])
    else:
        gain = float(checkpoint.get("detour_gain", 1.0))
        margin = float(checkpoint.get("clearance_margin_cm", 0.0))
        apex = gain * obstacle_top_y + margin

    thw = float(checkpoint.get("top_halfwidth_cm", 30.0))
    speed = float(checkpoint.get("approach_speed_mps", 0.1))
    return _trapezoid(start, seam, apex, thw, speed)


# ----------------------------------------------------------------------------------------
# Independent measurement (trajectory -> metrics)
# ----------------------------------------------------------------------------------------
TILT_K = 0.06  # degrees of base tilt per cm of lateral excursion.
_SAMPLE_STEP_CM = 0.5


def _box_lo_hi(box):
    c = box["center"]
    h = box["half_extents"]
    lo = [c[i] - h[i] for i in range(3)]
    hi = [c[i] + h[i] for i in range(3)]
    return lo, hi


def _seg_aabb_intersect(p0, p1, lo, hi):
    """Exact segment-vs-AABB intersection (slab method). True if they touch/overlap."""
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        d = p1[i] - p0[i]
        if abs(d) < 1e-12:
            if p0[i] < lo[i] - 1e-9 or p0[i] > hi[i] + 1e-9:
                return False
        else:
            t1 = (lo[i] - p0[i]) / d
            t2 = (hi[i] - p0[i]) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def _point_aabb_dist(p, lo, hi):
    s = 0.0
    for i in range(3):
        if p[i] < lo[i]:
            d = lo[i] - p[i]
        elif p[i] > hi[i]:
            d = p[i] - hi[i]
        else:
            d = 0.0
        s += d * d
    return math.sqrt(s)


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def _truncate(waypoints, s_max):
    """Return the polyline vertices from arc-length 0 up to s_max (cm)."""
    out = [list(waypoints[0])]
    acc = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        seg = _dist(a, b)
        if seg < 1e-12:
            continue
        if acc + seg <= s_max + 1e-9:
            out.append(list(b))
            acc += seg
        else:
            t = (s_max - acc) / seg
            out.append(_lerp(a, b, t))
            acc = s_max
            break
    return out


def _perp_dist_to_line(p, a, b):
    """Perpendicular distance from point p to the infinite line through a,b."""
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    ab2 = sum(c * c for c in ab)
    if ab2 < 1e-12:
        return _dist(p, a)
    t = sum(ap[i] * ab[i] for i in range(3)) / ab2
    proj = [a[i] + ab[i] * t for i in range(3)]
    return _dist(p, proj)


def measure(trajectory: dict, observation: dict, domain: dict | None = None) -> dict:
    """Re-derive every metric from the emitted trajectory against the TRUTH geometry.

    ``domain`` is the actuation half of a domain sweep row: ``speed_scale`` multiplies the
    achieved torch speed and ``latency_s`` is dead time before the torch moves (equivalently,
    it comes off the usable time budget). With ``domain=None`` -- or the neutral "nominal" row
    -- both are identity and the arithmetic is bit-for-bit what it was before domain sweeps
    existed, so the frozen v18/v19/v21/v24 verdicts are unchanged.
    """
    speed_scale = float((domain or {}).get("speed_scale", 1.0) or 1.0)
    latency = float((domain or {}).get("latency_s", 0.0) or 0.0)

    wps = [list(w) for w in trajectory["waypoints"]]
    speed = float(trajectory["speed_mps"]) * speed_scale
    budget = float(observation["time_budget_s"]) - latency
    start = list(observation["start_pose"])
    seam = list(observation["seam_pose"])

    # full intended path length (cm)
    length_cm = sum(_dist(wps[i], wps[i + 1]) for i in range(len(wps) - 1))

    # time the full path WOULD take (s), including the domain's dead time
    if speed <= 0:
        elapsed = float("inf")
    else:
        elapsed = latency + (length_cm / 100.0) / speed

    # arc-length the torch can physically cover within the usable time budget (cm)
    reachable_cm = speed * budget * 100.0 if speed > 0 and budget > 0 else 0.0
    s_max = min(length_cm, reachable_cm)

    traversed = _truncate(wps, s_max)
    endpoint = traversed[-1]
    tip = _dist(endpoint, seam)

    obs_lo, obs_hi = _box_lo_hi(observation["obstacle_box"])
    zone_lo, zone_hi = _box_lo_hi(observation["restricted_zone"])

    collision = 0
    intrusion = 0
    for i in range(len(traversed) - 1):
        a, b = traversed[i], traversed[i + 1]
        if _seg_aabb_intersect(a, b, obs_lo, obs_hi):
            collision = 1
        if _seg_aabb_intersect(a, b, zone_lo, zone_hi):
            intrusion = 1

    # min clearance to the obstacle along the traversed path (sampled)
    min_clear = float("inf")
    for i in range(len(traversed) - 1):
        a, b = traversed[i], traversed[i + 1]
        seg = _dist(a, b)
        n = max(1, int(seg / _SAMPLE_STEP_CM))
        for k in range(n + 1):
            p = _lerp(a, b, k / n)
            d = _point_aabb_dist(p, obs_lo, obs_hi)
            if d < min_clear:
                min_clear = d
    if min_clear == float("inf"):
        min_clear = _point_aabb_dist(endpoint, obs_lo, obs_hi)

    # base tilt from maximum lateral excursion off the start->seam line
    max_lat = 0.0
    for p in traversed:
        lat = _perp_dist_to_line(p, start, seam)
        if lat > max_lat:
            max_lat = lat
    tilt = TILT_K * max_lat

    return {
        "torch_tip_distance_to_target_cm": tip,
        "collision_count": int(collision),
        "restricted_zone_intrusions": int(intrusion),
        "max_base_tilt_degrees": tilt,
        "elapsed_seconds": elapsed,
        "min_clearance_to_obstacle_cm": min_clear,
    }


# ----------------------------------------------------------------------------------------
# The task
# ----------------------------------------------------------------------------------------
@register_task("g1_weld_approach")
class WeldApproach(Task):
    behavior = "g1_weld_approach"
    subsystem = "manipulation"
    goal = (
        "Route the torch from its start pose to the weld seam around the shifted obstacle, "
        "inside the time budget, without colliding, entering the restricted zone, or "
        "exceeding the base-tilt limit."
    )
    robot = "Unitree G1-compatible humanoid proxy"
    world = "tabletop_welding_obstacle_shift_v1"
    scene_env = "behavior-ci-tabletop-welding"
    camera = "/World/Cameras/BehaviorCI_PassFailCamera"
    env_id = "env_7d904291a384a1ae"

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
        return {
            "target_reach": Check("torch_tip_distance_to_target_cm", "<=", 2.0),
            "collision_free": Check("collision_count", "==", 0),
            "safety_zone_clear": Check("restricted_zone_intrusions", "==", 0),
            "base_stable": Check("max_base_tilt_degrees", "<=", 5.0),
            "timeout_free": Check("elapsed_seconds", "<", 30),
        }
