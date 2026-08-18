"""g1_base_traverse hosted Isaac grader (replay-only).

Uploaded into the hosted Isaac session for the REPLAY VIDEO only — the pass/fail verdict comes
from the task's pure measure() in the SDK, so this file does not (and must not) re-implement
measurement. It places the per-scenario pallet / restricted zone / work position and drives the
G1 base along the emitted path so the replay clip shows the traverse. Text-only; never imported
by the SDK (it imports omni.*).
"""


import math

# ----------------------------------------------------------------------------------------
# NOTE: there is deliberately NO measurement code in this file. The geometric verdict is a
# pure function of (trajectory, observation) authored once in task.py and evaluated by the
# SDK, so the hosted and offline gates cannot disagree on pass/fail. Everything below is
# scene dressing + actuation for the camera.
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Isaac actuation (real G1 in the saved scene) -- evidence + executability check.
# ----------------------------------------------------------------------------------------
ROOT = "/G1"
PELVIS = "/G1/pelvis"
JOINT = "/G1/joints/{}_joint"
WORK_POSE = "/World/WorkPose"
PALLET = "/World/AisleClutter"
ZONE = "/World/RestrictedZone"
CM_TO_M = 0.01


def _omni():
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    return omni, omni.usd, omni.timeline, omni.kit.app, Gf, Usd, UsdGeom, UsdPhysics


def _failure_result(run, code, message):
    """Shape the SDK expects if a hosted trial cannot be staged at all (infra, not behavior)."""
    metrics = {
        "base_distance_to_work_pose_cm": 999.0,
        "collision_count": 1,
        "restricted_zone_intrusions": 1,
        "max_base_tilt_degrees": 90.0,
        "elapsed_seconds": 999.0,
        "min_clearance_to_obstacle_cm": -999.0,
        "path_length_cm": 0.0,
    }
    return {
        "metrics": metrics,
        "events": [{"run": run, "time_seconds": 0.0, "code": code, "message": message}],
        "trajectory_id": f"g1-run{run:02d}",
    }


def _place_box(stage, UsdGeom, Gf, path, box, color, opacity=1.0):
    c = [box["center"][i] * CM_TO_M for i in range(3)]
    s = [2.0 * box["half_extents"][i] * CM_TO_M for i in range(3)]
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*c))
    x.AddScaleOp().Set(Gf.Vec3f(*s))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if opacity < 1.0:
        cube.CreateDisplayOpacityAttr([opacity])


def _place_point(stage, UsdGeom, Gf, path, pose_cm, radius=0.06):
    c = [pose_cm[i] * CM_TO_M for i in range(3)]
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(radius)
    x = UsdGeom.Xformable(s.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*c))


def _resample(waypoints, n):
    """Even arc-length resampling of the emitted polyline into ``n`` base poses (cm)."""
    if len(waypoints) < 2:
        return [list(waypoints[0])] * n if waypoints else []
    segs = []
    total = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        d = math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(3)))
        segs.append((a, b, d))
        total += d
    out = []
    for j in range(n):
        want = total * (j / max(1, n - 1))
        acc = 0.0
        pose = list(waypoints[-1])
        for a, b, d in segs:
            if d < 1e-9:
                continue
            if acc + d >= want:
                t = (want - acc) / d
                pose = [a[k] + (b[k] - a[k]) * t for k in range(3)]
                break
            acc += d
        out.append(pose)
    return out


def _heading_deg(p0, p1):
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def _set_base(stage, UsdGeom, Gf, pose_cm, heading_deg):
    """Place the G1 root at a base pose. Drives the ROOT transform, not a gait controller --
    the replay is a visualization of the planned path, not a dynamics claim."""
    prim = stage.GetPrimAtPath(ROOT)
    if not prim.IsValid():
        return
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(pose_cm[0] * CM_TO_M, pose_cm[1] * CM_TO_M, 0.0))
    x.AddRotateZOp().Set(float(heading_deg))


def _swing_legs(stage, UsdPhysics, phase):
    """Cosmetic leg swing so the clip reads as walking rather than sliding."""
    amp = 22.0 * math.sin(phase)
    for joint, deg in (
        ("left_hip_pitch", -amp),
        ("right_hip_pitch", amp),
        ("left_knee", abs(amp) * 0.9),
        ("right_knee", abs(amp) * 0.9),
    ):
        prim = stage.GetPrimAtPath(JOINT.format(joint))
        if not prim.IsValid():
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(
            prim, "angular"
        )
        drive.GetTargetPositionAttr().Set(float(deg))


def behavior_ci_arm_replay(args):
    """Stage the aisle and walk the G1 along the emitted path during the capture window.

    Dresses the scene from the observation, snaps the robot back to the start pose (not
    filmed), then steps it along the resampled path so the NEXT isaac.capture_video films the
    traverse in motion. Purely visual; the verdict is the geometric measure() in task.py.
    """
    action = args.get("action") or {}
    observation = args.get("observation") or {}
    waypoints = action.get("waypoints") or [[0.0, 0.0, 0.0]]
    try:
        _, ousd, otimeline, okitapp, Gf, Usd, UsdGeom, UsdPhysics = _omni()
        stage = ousd.get_context().get_stage()
    except Exception:  # pragma: no cover - hosted only
        return {"staged": False}

    if observation.get("obstacle_box"):
        _place_box(stage, UsdGeom, Gf, PALLET, observation["obstacle_box"], color=(0.55, 0.35, 0.1))
    if observation.get("restricted_zone"):
        _place_box(
            stage,
            UsdGeom,
            Gf,
            ZONE,
            observation["restricted_zone"],
            color=(0.9, 0.1, 0.1),
            opacity=0.30,
        )
    if observation.get("work_pose"):
        _place_point(stage, UsdGeom, Gf, WORK_POSE, observation["work_pose"], radius=0.10)

    otimeline.get_timeline_interface().play()
    app = okitapp.get_app()

    poses = _resample(waypoints, 96)
    # 1) snap HOME and settle (not filmed).
    _set_base(stage, UsdGeom, Gf, poses[0], _heading_deg(poses[0], poses[min(1, len(poses) - 1)]))
    for _ in range(30):
        app.update()
    # 2) walk it out; the capture window films this.
    for i in range(1, len(poses)):
        _set_base(stage, UsdGeom, Gf, poses[i], _heading_deg(poses[i - 1], poses[i]))
        _swing_legs(stage, UsdPhysics, i * 0.45)
        app.update()
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
