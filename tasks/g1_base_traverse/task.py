"""g1_base_traverse behavior-CI task — LOCOMOTION subsystem (authored in-repo, taskkit API).

Goal (one sentence): the G1 must walk from its start pose to the work position at the far end
of a cluttered aisle inside the time budget, keeping its base clear of a fixed pallet, staying
out of the restricted zone, and never exceeding the base-tilt limit.

Pure tier: scenarios + observation geometry + the path planner + the independent measurement +
the pass/fail checks. measure() is authored ONCE here; the SDK computes the verdict with it for
both the offline fixture and the hosted run, so they cannot drift. The hosted Isaac scene +
actuation for the replay video lives in the sibling grader_isaac.py.

Everything below is stdlib-only, deterministic, and geometry-first: the policy emits a BASE
PATH and the environment re-derives every metric from that path against the TRUTH geometry.
No number the policy reports about itself is read.

--------------------------------------------------------------------------------------------
ANTI-OVERFIT INVARIANT (the numbers that make the windows disjoint)
--------------------------------------------------------------------------------------------
The policy's only real lever is the lateral offset ``L`` of its detour shelf (cm, +y off the
aisle centre line). The base is a disc of radius BASE_RADIUS_CM = 25, so every solid is
inflated by 25 cm before the segment test. For a scenario whose pallet's far (+y) face is at
``y_hi``:

  * plain rows require        L >  y_hi + 25           (clear the inflated pallet)
  * ``racking`` rows forbid   L >= RACK_LO_CM - 25 = 145
        (the racking strip occupies y >= 170 over the WHOLE aisle run, so the diagonal legs
         cannot sneak under it either -- any path that reaches y = 145 anywhere is red)
  * base_stable forbids       L >  6.0 / 0.022 = 272.7      (TILT_K_DEG_PER_CM, 6 deg cap)

HELD_OUT is built so NO CONSTANT (geometry-blind) L can clear it:

  * held-out row 4 is a ``racking`` row     ->  forbids L >= 145
  * held-out row 6, clutter_offset 105 -> y_hi = 125  ->  requires L > 150
  * held-out row 7, clutter_offset 130 -> y_hi = 150  ->  requires L > 175

  L >= 145 is forbidden by row 4 and L <= 150 is forbidden by row 6: the windows are DISJOINT.
  Swept empirically: over every constant L in [60, 400] cm at 1 cm steps, the best a constant
  ever does is 15/16 (L = 176, which loses row 4 on zone_clear). Cranking L is equally dead:
  L > 272.7 busts base_stable (gain 2.0 measures 6.09/6.93/8.03 deg on held-out rows 2/6/7).

A pallet-relative policy (L = observed far face + margin M) sizes the detour per observation
and clears all sixteen. The admissible M window is NARROW and two-sided:

  * M <= 52  -> the diagonal leg into the shelf clips the inflated ``keep_out`` band on
                visible row 3 (zone_clear red). The floor is M = 53.
  * M >= 100 -> the shelf itself enters the racking strip on held-out row 4 (zone_clear red).
  * so the window is M in [53, 99]; the shipped policy uses M = 65, i.e. 12 cm above the floor
    and 34 cm below the ceiling. Under the obs_noise domain column the floor rises to M = 56.

The ``timeout`` rows close the last hole: at walk_speed 0.75 m/s the 17.5 m traverse measures
23.72 s against a 22 s budget and the base stops 128.7 cm short, so reached_goal and
timeout_free both go red. Speed is capped at MAX_WALK_SPEED_MPS, so "walk faster" is bounded.
"""

from __future__ import annotations

import math

from cybernetics.behavior_ci.taskkit import Check, Task, register_task

# ----------------------------------------------------------------------------------------
# Scenario geometry (all lengths cm, all times s)
# ----------------------------------------------------------------------------------------
BASE_RADIUS_CM = 25.0  # G1 base modelled as a disc; every solid is inflated by this
PELVIS_Z_CM = 70.0  # the height the base path is measured at
SOLID_Z_HI_CM = 140.0  # pallet/zone boxes span z in [0, 140] so z never gates a test

CLUTTER_NEG_CM = 60.0  # how far the pallet pokes into -y (it straddles the centre line)
CLUTTER_FACE_BASE_CM = 20.0  # +y face = clutter_offset + this, for ordinary rows
CLUTTER_FACE_PINCH_CM = 70.0  # 'pinch' stress: far +y face, so an under-detour hits the pallet
CLUTTER_HALF_X_CM = 120.0

KEEP_OUT_W_CM = 25.0  # thickness of the +y restricted band ('keep_out' stress)
KEEP_OUT_HALF_X_CM = 180.0
BACK_ZONE_W_CM = 80.0  # ordinary rows park the restricted zone behind the pallet, in -y

RACK_LO_CM = 170.0  # 'racking' stress: restricted strip occupies y >= 170 ...
RACK_HI_CM = 1400.0  # ... and is tall/deep enough that any over-detour shelf enters it
RACK_PAD_X_CM = 50.0  # the strip runs the WHOLE aisle, so diagonal legs cannot sneak past it

AISLE_X_CM = 900.0
AISLE_X_LONG_CM = 1750.0  # 'timeout' stress: a 17.5 m bay-to-bay traverse
TIME_BUDGET_S = 22.0

# index -> (clutter_offset_cm, stress).  Golden: 3->keep_out, 5->pinch, 7->timeout.
VISIBLE = [
    {"clutter_offset_cm": 10.0, "stresses": None},
    {"clutter_offset_cm": 18.0, "stresses": None},
    {"clutter_offset_cm": 26.0, "stresses": None},
    {"clutter_offset_cm": 34.0, "stresses": "keep_out"},
    {"clutter_offset_cm": 22.0, "stresses": None},
    {"clutter_offset_cm": 30.0, "stresses": "pinch"},
    {"clutter_offset_cm": 40.0, "stresses": None},
    {"clutter_offset_cm": 55.0, "stresses": "timeout"},
]

# Fixed, seeded perturbation bank. Held out of every candidate eval copy. Designed so that
# NO single constant (geometry-blind) lateral offset can clear it together with the visible
# set -- see the ANTI-OVERFIT INVARIANT in the module docstring:
#  - the 'racking' row forbids L >= 145;
#  - the large-offset rows (105, 130 -> +y faces 125, 150) require L > 150 / 175.
# A pallet-relative policy sizes the detour per observation and clears all of them.
HELD_OUT = [
    {"clutter_offset_cm": 14.0, "stresses": None},
    {"clutter_offset_cm": 24.0, "stresses": "keep_out"},
    {"clutter_offset_cm": 36.0, "stresses": "pinch"},
    {"clutter_offset_cm": 28.0, "stresses": "timeout"},
    {"clutter_offset_cm": 25.0, "stresses": "racking"},
    {"clutter_offset_cm": 48.0, "stresses": None},
    {"clutter_offset_cm": 105.0, "stresses": None},
    {"clutter_offset_cm": 130.0, "stresses": None},
]


def build_observation(scenario: dict) -> dict:
    off = float(scenario["clutter_offset_cm"])
    stress = scenario.get("stresses")

    aisle_x = AISLE_X_LONG_CM if stress == "timeout" else AISLE_X_CM
    face = CLUTTER_FACE_PINCH_CM if stress == "pinch" else CLUTTER_FACE_BASE_CM

    mx = aisle_x / 2.0
    zc = SOLID_Z_HI_CM / 2.0

    y_lo = -CLUTTER_NEG_CM
    y_hi = off + face
    clutter_center = [mx, 0.5 * (y_lo + y_hi), zc]
    clutter_half = [CLUTTER_HALF_X_CM, 0.5 * (y_hi - y_lo), zc]

    if stress == "keep_out":
        # thin band hugging the pallet's +y face: y in [y_hi, y_hi + KEEP_OUT_W_CM]
        z_lo, z_hi = y_hi, y_hi + KEEP_OUT_W_CM
        zone_center = [mx, 0.5 * (z_lo + z_hi), zc]
        zone_half = [KEEP_OUT_HALF_X_CM, 0.5 * (z_hi - z_lo), zc]
    elif stress == "racking":
        # racking strip along the far aisle wall: y in [170, 1400], spanning the whole run in
        # x so an over-detour cannot dodge it by starting the shelf early or late.
        zone_center = [mx, 0.5 * (RACK_LO_CM + RACK_HI_CM), zc]
        zone_half = [mx + RACK_PAD_X_CM, 0.5 * (RACK_HI_CM - RACK_LO_CM), zc]
    else:
        # behind the pallet on the -y side, out of the +y detour corridor
        zy_hi = y_lo
        zy_lo = y_lo - BACK_ZONE_W_CM
        zone_center = [mx, 0.5 * (zy_lo + zy_hi), zc]
        zone_half = [KEEP_OUT_HALF_X_CM, 0.5 * (zy_hi - zy_lo), zc]

    return {
        "start_pose": [0.0, 0.0, PELVIS_Z_CM],
        "work_pose": [aisle_x, 0.0, PELVIS_Z_CM],
        "obstacle_box": {"center": clutter_center, "half_extents": clutter_half},
        "restricted_zone": {"center": zone_center, "half_extents": zone_half},
        "base_radius_cm": BASE_RADIUS_CM,
        "aisle_length_cm": aisle_x,
        "time_budget_s": TIME_BUDGET_S,
        "clutter_offset_cm": off,
        "stresses": stress,
    }


# ----------------------------------------------------------------------------------------
# Domain sweep (the customer's "vary the domain parameters")
# ----------------------------------------------------------------------------------------
# A domain setting perturbs (a) what the policy is ALLOWED TO SEE and (b) how the hardware
# actually performs. The truth observation is never touched, so measurement is unaffected by
# the perception perturbation -- that asymmetry is the whole point: a policy that only just
# clears the geometry has nothing left when its perception is a few cm off.
#
#   obs_noise_cm : the OBSERVED pallet is displaced by this many cm along a direction derived
#                  deterministically from the domain id (never random at runtime).
#   speed_scale  : multiplies the achieved walking speed.
#   latency_s    : dead time before the base starts moving; equivalently, this much is taken
#                  off the usable time budget.
#
# Why these three values:
#   * "nominal" is the neutral control. Its trials must be numerically identical to a run with
#     no domain at all -- that is what makes the sweep's other two columns readable.
#   * obs_noise_cm 4.0: the admissible pallet-relative margin window is [53, 99] cm and the
#     tuned policy sits at 65, i.e. 12 cm above the floor. A 4 cm perception error moves the
#     effective margin by -2.936 cm (the -y component of the "obs_noise" direction), which
#     eats ~24% of that slack: enough to turn a zero-margin policy red (see
#     policies/g1_base_traverse_v1_nomargin.pt, margin 53 -- green at nominal, red here on both
#     keep_out rows) without punishing an honestly-tuned one.
#   * speed_scale 0.8 / latency_s 1.5: the binding row is the 17.5 m 'timeout' traverse, which
#     the tuned policy walks in 14.8 s of a 22 s budget. Degraded, that becomes
#     1.5 + 17.79/(1.2*0.8) = 20.0 s -- still inside the budget with ~2 s to spare, so the
#     column tests actuation degradation rather than simply deleting the behavior.
DOMAIN_SWEEP = [
    {"id": "nominal", "obs_noise_cm": 0.0, "speed_scale": 1.0, "latency_s": 0.0},
    {"id": "obs_noise", "obs_noise_cm": 4.0, "speed_scale": 1.0, "latency_s": 0.0},
    {"id": "actuation_degraded", "obs_noise_cm": 0.0, "speed_scale": 0.8, "latency_s": 1.5},
]


def _domain_direction(domain_id: str) -> tuple:
    """Unit (dx, dy) for a domain id: FNV-1a over the id -> an angle on a 0.1-degree grid.

    Deterministic and stable across processes/platforms (no hash randomization, no RNG), so a
    sweep is reproducible from the id alone. The shipped id "obs_noise" resolves to
    theta = 312.8 deg -> (dx, dy) = (+0.679, -0.734): the -y component pushes the OBSERVED
    pallet face TOWARD the aisle centre line, i.e. the policy under-estimates how far it must
    step out. That is the adverse direction for this pack, and it is chosen on purpose -- a
    domain column that only ever helps the policy proves nothing.
    """
    h = 2166136261
    for ch in domain_id:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    theta = 2.0 * math.pi * ((h % 3600) / 3600.0)
    return math.cos(theta), math.sin(theta)


def apply_domain(observation: dict, domain: dict | None) -> dict:
    """Return a COPY of ``observation`` with the perception perturbation applied.

    Only the pallet -- the thing this behavior has to reason about -- moves. The robot's own
    start pose, the work position and the posted restricted zone are survey data and stay put,
    so this models sensing error, not a different world. measure() is always handed the
    untouched truth observation.
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
# Action planner (checkpoint -> base path)
# ----------------------------------------------------------------------------------------
# Physical ceiling on G1 walking speed (m/s). A humanoid does not traverse a shop floor at
# 100 m/s; clamping here stops "crank the speed" from trivially satisfying timeout_free while
# leaving every geometric check (collision/intrusion/tilt) untouched.
MAX_WALK_SPEED_MPS = 1.4


def _detour_path(start: list, work: list, lateral: float, hspan: float, speed: float) -> dict:
    """Shared construction: start -> P1 -> P2 -> work, with a shelf at +y ``lateral``."""
    speed = max(0.0, min(speed, MAX_WALK_SPEED_MPS))

    mx = 0.5 * (start[0] + work[0])
    # The shelf must fit inside the aisle run; a wider-than-aisle shelf is meaningless.
    hspan = max(40.0, min(hspan, 0.45 * abs(work[0] - start[0])))
    z = start[2]

    p1 = [mx - hspan, lateral, z]
    p2 = [mx + hspan, lateral, z]

    waypoints = [list(start), p1, p2, list(work)]
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
    work = list(observation["work_pose"])
    box = observation["obstacle_box"]
    bc = box["center"]
    bh = box["half_extents"]

    # Far (+y) face of the pallet as seen in the observation geometry the policy is given.
    clutter_far_y = bc[1] + bh[1]
    aisle_len = abs(work[0] - start[0])

    mlp = checkpoint.get("mlp")
    if mlp is not None:
        # Learned path (decoded by the SDK's learned-mlp backend): the net maps the observed
        # aisle geometry to the detour parameters. Features are NORMALIZED geometry
        # (feature_spec traverse-geometry/v1), outputs are the shelf parameters
        # (output_spec detour/v1 = [lateral_offset_cm, detour_half_span_cm, walk_speed_mps]).
        features = [clutter_far_y / 100.0, aisle_len / 1000.0]
        out = _mlp_forward(mlp, features)
        lateral = out[0]  # used raw: the environment measures the resulting path
        hspan = max(40.0, min(out[1], 450.0))
        speed = max(0.05, min(out[2], MAX_WALK_SPEED_MPS))
        return _detour_path(start, work, lateral, hspan, speed)

    mode = checkpoint.get("offset_mode", "relative")
    if mode == "absolute":
        # geometry-blind: a fixed sidestep regardless of where the pallet actually is.
        lateral = float(checkpoint["absolute_offset_cm"])
    else:
        gain = float(checkpoint.get("offset_gain", 1.0))
        margin = float(checkpoint.get("clearance_margin_cm", 0.0))
        lateral = gain * clutter_far_y + margin

    hspan = float(checkpoint.get("detour_half_span_cm", 200.0))
    speed = float(checkpoint.get("walk_speed_mps", 0.6))
    return _detour_path(start, work, lateral, hspan, speed)


# ----------------------------------------------------------------------------------------
# Independent measurement (base path -> metrics)
# ----------------------------------------------------------------------------------------
# HEURISTIC, and labelled as one: this task has no dynamics model, so base tilt is taken to be
# proportional to the maximum lateral excursion off the straight start->work line. It stands in
# for the roll a humanoid picks up when it side-steps hard around clutter. It is monotone in
# excursion, which is the property the anti-gaming design needs (cranking the detour must cost
# something); it is NOT a measurement of real G1 roll, and nothing here claims otherwise.
TILT_K_DEG_PER_CM = 0.022
_SAMPLE_STEP_CM = 1.0


def _box_lo_hi(box, pad: float = 0.0):
    """AABB bounds, optionally inflated by ``pad`` in x/y (the base disc's footprint).

    Inflating the AABB is the CONSERVATIVE outer bound of the true Minkowski sum of a disc and
    a box (which has rounded corners); it can only ever call a near-corner grazing pass a hit,
    never the reverse. z is left alone -- the solids already span the robot's full height.
    """
    c = box["center"]
    h = box["half_extents"]
    pads = (pad, pad, 0.0)
    lo = [c[i] - h[i] - pads[i] for i in range(3)]
    hi = [c[i] + h[i] + pads[i] for i in range(3)]
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
    """Re-derive every metric from the emitted path against the TRUTH geometry.

    ``domain`` is the actuation half of a domain sweep row: ``speed_scale`` multiplies the
    achieved walking speed and ``latency_s`` is dead time before the base moves (equivalently,
    it comes off the usable time budget). With ``domain=None`` -- or the neutral "nominal" row
    -- both are identity and the arithmetic is bit-for-bit what it was before domain sweeps
    existed.
    """
    speed_scale = float((domain or {}).get("speed_scale", 1.0) or 1.0)
    latency = float((domain or {}).get("latency_s", 0.0) or 0.0)

    wps = [list(w) for w in trajectory["waypoints"]]
    speed = float(trajectory["speed_mps"]) * speed_scale
    budget = float(observation["time_budget_s"]) - latency
    start = list(observation["start_pose"])
    work = list(observation["work_pose"])
    radius = float(observation.get("base_radius_cm", BASE_RADIUS_CM))

    # full intended path length (cm)
    length_cm = sum(_dist(wps[i], wps[i + 1]) for i in range(len(wps) - 1))

    # time the full traverse WOULD take (s), including the domain's dead time
    if speed <= 0:
        elapsed = float("inf")
    else:
        elapsed = latency + (length_cm / 100.0) / speed

    # arc-length the base can physically cover within the usable time budget (cm)
    reachable_cm = speed * budget * 100.0 if speed > 0 and budget > 0 else 0.0
    s_max = min(length_cm, reachable_cm)

    traversed = _truncate(wps, s_max)
    endpoint = traversed[-1]
    goal_gap = _dist(endpoint, work)

    # Solids are inflated by the base radius: the G1 is a disc, not a point.
    obs_lo, obs_hi = _box_lo_hi(observation["obstacle_box"], radius)
    zone_lo, zone_hi = _box_lo_hi(observation["restricted_zone"], radius)

    collision = 0
    intrusion = 0
    for i in range(len(traversed) - 1):
        a, b = traversed[i], traversed[i + 1]
        if _seg_aabb_intersect(a, b, obs_lo, obs_hi):
            collision = 1
        if _seg_aabb_intersect(a, b, zone_lo, zone_hi):
            intrusion = 1

    # min clearance of the base SHELL to the pallet along the traversed path (sampled);
    # negative means the disc overlapped the pallet.
    raw_lo, raw_hi = _box_lo_hi(observation["obstacle_box"])
    min_clear = float("inf")
    for i in range(len(traversed) - 1):
        a, b = traversed[i], traversed[i + 1]
        seg = _dist(a, b)
        n = max(1, int(seg / _SAMPLE_STEP_CM))
        for k in range(n + 1):
            p = _lerp(a, b, k / n)
            d = _point_aabb_dist(p, raw_lo, raw_hi) - radius
            if d < min_clear:
                min_clear = d
    if min_clear == float("inf"):
        min_clear = _point_aabb_dist(endpoint, raw_lo, raw_hi) - radius

    # base tilt from maximum lateral excursion off the start->work line (documented heuristic)
    max_lat = 0.0
    for p in traversed:
        lat = _perp_dist_to_line(p, start, work)
        if lat > max_lat:
            max_lat = lat
    tilt = TILT_K_DEG_PER_CM * max_lat

    return {
        "base_distance_to_work_pose_cm": goal_gap,
        "collision_count": int(collision),
        "restricted_zone_intrusions": int(intrusion),
        "max_base_tilt_degrees": tilt,
        "elapsed_seconds": elapsed,
        "min_clearance_to_obstacle_cm": min_clear,
        "path_length_cm": length_cm,
    }


# ----------------------------------------------------------------------------------------
# The task
# ----------------------------------------------------------------------------------------
@register_task("g1_base_traverse")
class BaseTraverse(Task):
    behavior = "g1_base_traverse"
    subsystem = "locomotion"
    goal = (
        "Walk from the start pose to the work position at the far end of a cluttered aisle "
        "inside the time budget, without touching the pallet, entering the restricted zone, "
        "or exceeding the base-tilt limit."
    )
    robot = "Unitree G1-compatible humanoid proxy"
    world = "shop_floor_aisle_clutter_shift_v1"
    scene_env = "behavior-ci-shop-floor-aisle"
    camera = "/World/Cameras/BehaviorCI_TraverseCamera"
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
        # Thresholds:
        #   reached_goal   the base stops within one footprint diameter / 2 of the work pose.
        #   collision_free zero segment-vs-inflated-pallet intersections along the walked path.
        #   zone_clear     zero segment-vs-inflated-restricted-zone intersections.
        #   base_stable    6 deg of roll; with TILT_K 0.022 deg/cm that caps lateral excursion
        #                  at 272.7 cm, above the 215 cm the hardest held-out row legitimately
        #                  needs -- so honest detours pass and cranked ones do not.
        #   timeout_free   the 22 s aisle budget (the 17.5 m row costs the tuned policy 14.8 s).
        return {
            "reached_goal": Check("base_distance_to_work_pose_cm", "<=", 15.0),
            "collision_free": Check("collision_count", "==", 0),
            "zone_clear": Check("restricted_zone_intrusions", "==", 0),
            "base_stable": Check("max_base_tilt_degrees", "<=", 6.0),
            "timeout_free": Check("elapsed_seconds", "<", 22.0),
        }
