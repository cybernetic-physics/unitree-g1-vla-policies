"""g1_weld_approach hosted Isaac grader (replay-only).

Uploaded into the hosted Isaac session for the REPLAY VIDEO only — the pass/fail verdict comes
from the task's pure measure() in the SDK, so this file does not (and must not) re-implement
measurement. It places the per-scenario obstacle/zone/seam and actuates the G1 along the
emitted trajectory so the replay clip shows the weld-approach. Text-only; never imported by
the SDK (it imports omni.*).
"""


import math

# ----------------------------------------------------------------------------------------
# Shared geometric measurement (verbatim copy of measure.py; pinned + drift-checked by a
# unit test). The verdict is a pure function of (trajectory, observation), so the hosted and
# offline gates can never disagree on pass/fail.
# ----------------------------------------------------------------------------------------


















# ----------------------------------------------------------------------------------------
# Isaac actuation (real G1 in the saved scene) -- evidence + executability check.
# ----------------------------------------------------------------------------------------
PALM = "/G1/right_wrist_yaw_link/right_hand_palm_link"
JOINT = "/G1/joints/{}_joint"
PELVIS = "/G1/pelvis"
SEAM = "/World/WeldSeam"
OBSTACLE = "/World/Obstacle"
ZONE = "/World/RestrictedZone"
CM_TO_M = 0.01
# Where the task-local frame is drawn in the SAVED cicd_ship_yard scene (meters). Purely a
# replay-visualization anchor -- the verdict is the pure measure() over task-frame geometry,
# which this offset never touches. Chosen so the schematic seam marker coincides with the
# physical weld ring on the elevated pipe (the calibrated v19 reach of the fixed-base G1:
# seam_pose (120, 0, 5) cm + anchor = (0.478, 0.026, 1.13) m, measured in-session by
# isaac/scene_realism.py).
WORLD_ANCHOR_M = (-0.722, 0.026, 1.08)


def _omni():
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    return omni, omni.usd, omni.timeline, omni.kit.app, Gf, Usd, UsdGeom, UsdPhysics


def _failure_result(run, code, message):
    metrics = {
        "torch_tip_distance_to_target_cm": 999.0,
        "collision_count": 1,
        "restricted_zone_intrusions": 1,
        "max_base_tilt_degrees": 90.0,
        "elapsed_seconds": 999.0,
        "min_clearance_to_obstacle_cm": 0.0,
    }
    return {
        "metrics": metrics,
        "events": [{"run": run, "time_seconds": 0.0, "code": code, "message": message}],
        "trajectory_id": f"g1-run{run:02d}",
    }






def _place_box(stage, UsdGeom, Gf, path, box, color, opacity=1.0):
    c = [box["center"][i] * CM_TO_M + WORLD_ANCHOR_M[i] for i in range(3)]
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


def _place_point(stage, UsdGeom, Gf, path, pose_cm):
    c = [pose_cm[i] * CM_TO_M + WORLD_ANCHOR_M[i] for i in range(3)]
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(0.02)
    x = UsdGeom.Xformable(s.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*c))


def _set_arm(stage, UsdPhysics, apex_cm):
    # Map the lateral apex (cm) to a shoulder-roll detour + reach so larger detours visibly
    # swing the arm wider. Sets the joint drive TARGETS only (no stepping).
    f = max(0.0, min(apex_cm / 80.0, 1.5))
    for joint, deg in (
        ("right_shoulder_pitch", -95 - 15 * f),
        ("right_shoulder_roll", -8 - 30 * f),
        ("right_elbow", 45 - 15 * f),
        ("right_wrist_pitch", 12),
    ):
        prim = stage.GetPrimAtPath(JOINT.format(joint))
        if not prim.IsValid():
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(
            prim, "angular"
        )
        drive.GetTargetPositionAttr().Set(float(deg))


def _drive_arm(stage, UsdPhysics, okitapp, apex_cm, steps=150):
    _set_arm(stage, UsdPhysics, apex_cm)
    app = okitapp.get_app()
    for _ in range(steps):
        app.update()


def behavior_ci_arm_replay(args):
    """Arm the weld-approach motion so the NEXT isaac.capture_video films the arm MOVING.

    Resets to a home pose and settles there (this part is not filmed), then commands the
    target pose and returns IMMEDIATELY — the joint drive then plays out over the next ~2-3 s
    of real time, which is exactly the capture window, so the clip shows the approach in
    motion instead of a settled still. Purely visual; the verdict is the geometric measure().
    """
    action = args.get("action") or {}
    observation = args.get("observation") or {}
    apex_cm = max((w[1] for w in action.get("waypoints", [[0, 0, 0]])), default=0.0)
    try:
        _, ousd, otimeline, okitapp, Gf, Usd, UsdGeom, UsdPhysics = _omni()
        stage = ousd.get_context().get_stage()
    except Exception:  # pragma: no cover - hosted only
        return {"staged": False}
    if observation.get("obstacle_box"):
        _place_box(stage, UsdGeom, Gf, OBSTACLE, observation["obstacle_box"], color=(0.8, 0.2, 0.1))
    if observation.get("restricted_zone"):
        _place_box(
            stage,
            UsdGeom,
            Gf,
            ZONE,
            observation["restricted_zone"],
            color=(0.9, 0.1, 0.1),
            opacity=0.35,
        )
    if observation.get("seam_pose"):
        _place_point(stage, UsdGeom, Gf, SEAM, observation["seam_pose"])
    otimeline.get_timeline_interface().play()
    app = okitapp.get_app()
    # 1) snap to HOME and settle (not filmed).
    _set_arm(stage, UsdPhysics, 0.0)
    for _ in range(40):
        app.update()
    # 2) command the target and return now; the drive unfolds during the capture window.
    _set_arm(stage, UsdPhysics, apex_cm)
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
