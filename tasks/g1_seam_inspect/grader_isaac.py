"""g1_seam_inspect hosted Isaac grader (replay-only).

Uploaded into the hosted Isaac session for the REPLAY VIDEO only — the pass/fail verdict comes
from the task's pure measure() in the SDK, so this file does not (and must not) re-implement
measurement. It places the per-scenario seam, its inspection targets and the standoff keep-out,
parks the G1 at the emitted head pose and plays the gaze list out on the waist/head joints so
the replay clip shows the inspection sweep. Text-only; never imported by the SDK (it imports
omni.*).
"""


import math

# ----------------------------------------------------------------------------------------
# NOTE: there is deliberately NO measurement code in this file. Coverage, standoff and elapsed
# time are a pure function of (look plan, observation) authored once in task.py and evaluated
# by the SDK, so the hosted and offline gates cannot disagree on pass/fail. Everything below is
# scene dressing + actuation for the camera.
# ----------------------------------------------------------------------------------------

ROOT = "/G1"
JOINT = "/G1/joints/{}_joint"
SEAM = "/World/WeldSeam"
TARGET = "/World/SeamTarget_{:02d}"
KEEP_OUT = "/World/StandoffKeepOut"
CM_TO_M = 0.01

# The G1 has no dedicated pan/tilt neck in the stock articulation, so the gaze is played out on
# the waist chain with the head joints preferred when the loaded USD provides them. Cosmetic
# only: the camera prim the SDK captures from is the one named in the task's `camera` field.
YAW_JOINTS = ("head_yaw", "waist_yaw")
PITCH_JOINTS = ("head_pitch", "waist_pitch")


def _omni():
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    return omni, omni.usd, omni.timeline, omni.kit.app, Gf, Usd, UsdGeom, UsdPhysics


def _failure_result(run, code, message):
    """Shape the SDK expects if a hosted trial cannot be staged at all (infra, not behavior)."""
    metrics = {
        "seam_coverage_fraction": 0.0,
        "missed_target_count": 99.0,
        "missed_targets": "",
        "standoff_violations": 1,
        "min_standoff_margin_cm": -999.0,
        "short_dwell_gazes": 99.0,
        "gaze_count": 0.0,
        "worst_target_gap_degrees": 180.0,
        "elapsed_seconds": 999.0,
        "time_overrun_seconds": 999.0,
    }
    return {
        "metrics": metrics,
        "events": [{"run": run, "time_seconds": 0.0, "code": code, "message": message}],
        "trajectory_id": f"g1-run{run:02d}",
    }


def _place_point(stage, UsdGeom, Gf, path, pose_cm, radius, color):
    c = [pose_cm[i] * CM_TO_M for i in range(3)]
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(radius)
    x = UsdGeom.Xformable(s.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*c))
    s.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _place_seam(stage, UsdGeom, Gf, center, axis, half_cm):
    """Draw the seam as a thin capsule along its axis (a 3D segment, matching the geometry)."""
    cap = UsdGeom.Capsule.Define(stage, SEAM)
    cap.CreateRadiusAttr(0.012)
    cap.CreateHeightAttr(2.0 * half_cm * CM_TO_M)
    cap.CreateAxisAttr("Z")
    # rotate +Z onto the seam axis: the axis lives in the y-z plane, so a single X rotation does
    # it (tilt 0 -> along +y is -90 deg about X; tilt 90 -> along +z is 0 deg).
    tilt = math.degrees(math.atan2(axis[2], axis[1])) if (axis[1] or axis[2]) else 90.0
    x = UsdGeom.Xformable(cap.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[center[i] * CM_TO_M for i in range(3)]))
    x.AddRotateXOp().Set(float(tilt - 90.0))
    cap.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.75, 0.15)])


def _place_keep_out(stage, UsdGeom, Gf, center, half_cm, radius_cm):
    """The standoff keep-out is a capsule of ``radius_cm`` around the seam segment."""
    cap = UsdGeom.Capsule.Define(stage, KEEP_OUT)
    cap.CreateRadiusAttr(radius_cm * CM_TO_M)
    cap.CreateHeightAttr(2.0 * half_cm * CM_TO_M)
    cap.CreateAxisAttr("Z")
    x = UsdGeom.Xformable(cap.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[center[i] * CM_TO_M for i in range(3)]))
    cap.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.1, 0.1)])
    cap.CreateDisplayOpacityAttr([0.18])


def _set_root(stage, UsdGeom, Gf, head_pose_cm, yaw_deg):
    """Park the G1 under the emitted head pose (the head sits above the root, not at it)."""
    prim = stage.GetPrimAtPath(ROOT)
    if not prim.IsValid():
        return
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(
        Gf.Vec3d(head_pose_cm[0] * CM_TO_M, head_pose_cm[1] * CM_TO_M, 0.0)
    )
    x.AddRotateZOp().Set(float(yaw_deg))


def _drive(stage, UsdPhysics, names, deg):
    """Set the first joint from ``names`` that the loaded articulation actually provides."""
    for name in names:
        prim = stage.GetPrimAtPath(JOINT.format(name))
        if not prim.IsValid():
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(
            prim, "angular"
        )
        drive.GetTargetPositionAttr().Set(float(deg))
        return True
    return False


def behavior_ci_arm_replay(args):
    """Stage the weld cell and play the emitted gaze list out during the capture window.

    Dresses the scene from the observation, parks the robot at the emitted head pose, snaps the
    head level (not filmed), then steps through the gaze list -- slewing between gazes and
    holding each for its dwell -- so the NEXT isaac.capture_video films the sweep in motion.
    Purely visual; the verdict is the geometric measure() in task.py.
    """
    action = args.get("action") or {}
    observation = args.get("observation") or {}
    head = action.get("head_pose") or [0.0, 0.0, 0.0]
    gazes = action.get("gazes") or []
    try:
        _, ousd, otimeline, okitapp, Gf, Usd, UsdGeom, UsdPhysics = _omni()
        stage = ousd.get_context().get_stage()
    except Exception:  # pragma: no cover - hosted only
        return {"staged": False}

    center = observation.get("seam_center")
    axis = observation.get("seam_axis")
    half = float(observation.get("seam_half_length_cm", 0.0))
    if center and axis:
        _place_seam(stage, UsdGeom, Gf, center, axis, half)
        _place_keep_out(
            stage, UsdGeom, Gf, center, half, float(observation.get("keep_out_radius_cm", 0.0))
        )
    for i, t in enumerate(observation.get("seam_targets") or []):
        _place_point(stage, UsdGeom, Gf, TARGET.format(i), t, 0.03, (0.15, 0.85, 0.3))

    otimeline.get_timeline_interface().play()
    app = okitapp.get_app()

    # The robot faces +x (toward the seam) from wherever the policy parked it.
    _set_root(stage, UsdGeom, Gf, head, 0.0)
    # 1) snap the head level and settle (not filmed).
    _drive(stage, UsdPhysics, YAW_JOINTS, 0.0)
    _drive(stage, UsdPhysics, PITCH_JOINTS, 0.0)
    for _ in range(30):
        app.update()

    # 2) slew + dwell through the plan; the capture window films this.
    prev_yaw, prev_pitch = 0.0, 0.0
    for g in gazes:
        yaw = float(g.get("yaw_deg", 0.0))
        pitch = float(g.get("pitch_deg", 0.0))
        steps = max(4, int(max(abs(yaw - prev_yaw), abs(pitch - prev_pitch)) / 2.0))
        for k in range(1, steps + 1):
            f = k / steps
            _drive(stage, UsdPhysics, YAW_JOINTS, prev_yaw + (yaw - prev_yaw) * f)
            _drive(stage, UsdPhysics, PITCH_JOINTS, prev_pitch + (pitch - prev_pitch) * f)
            app.update()
        for _ in range(max(1, int(float(g.get("dwell_s", 0.0)) * 20))):
            app.update()
        prev_yaw, prev_pitch = yaw, pitch
    return {"staged": True}


def _world_pos(stage, UsdGeom, Usd, Gf, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return Gf.Vec3d(0, 0, 0)
    t = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(prim)
        .ExtractTranslation()
    )
    return Gf.Vec3d(t[0], t[1], t[2])
