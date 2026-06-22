"""Cybernetic Physics Behavior CI — in-session module (uploaded by the SDK adapter).

The ``cybernetics.behavior_ci`` ``IsaacSessionAdapter`` uploads this file into a
fresh Isaac session under ``/data/workspace`` and calls:

- ``setup_scene(args)`` once after spawning the Unitree G1 — builds the welding
  scene (table, weld seam, shifted obstacle, red restricted zone, pass/fail
  camera, fixed base) and CALIBRATES it by measuring the real right-hand reach
  for the regressed (v18) and fixed (v19) controller poses, then places the weld
  seam at the v19 reach and the obstacle at the v18 reach;
- ``behavior_ci_run_trial(args)`` per policy — drives the scripted weld-approach
  on the real G1 and measures pass/fail from the settled physical state.

This is authored at runtime on a blank session, so it does not depend on a
pre-published environment snapshot. It is NOT a learned policy: the controller
parameter ``clearance_margin_cm`` drives a real reach whose measured outcome
(distance to seam, clearance to obstacle, zone intrusion, base tilt) is computed
from physics, not hardcoded. A real VLA/GR00T policy would replace the scripted
joint targets while keeping this measurement contract.
"""

import math

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

PALM = "/G1/right_wrist_yaw_link/right_hand_palm_link"
JOINT = "/G1/joints/{}_joint"
SEAM = "/World/WeldSeam"
OBSTACLE = "/World/Obstacle"
ZONE = "/World/RestrictedZone"
PELVIS = "/G1/pelvis"
COLLISION_CLEARANCE_M = 0.09


def _stage():
    return omni.usd.get_context().get_stage()


def _set_drive(stage, joint, degrees):
    prim = stage.GetPrimAtPath(JOINT.format(joint))
    drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(
        prim, "angular"
    )
    drive.GetTargetPositionAttr().Set(float(degrees))


def _world_pos(stage, path):
    t = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        .ExtractTranslation()
    )
    return Gf.Vec3d(t[0], t[1], t[2])


def _world_aabb(stage, path):
    rng = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(stage.GetPrimAtPath(path))
        .ComputeAlignedRange()
    )
    return rng.GetMin(), rng.GetMax()


def _base_tilt_deg(stage):
    up = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(stage.GetPrimAtPath(PELVIS))
        .TransformDir(Gf.Vec3d(0, 0, 1))
        .GetNormalized()
    )
    return round(math.degrees(math.acos(max(-1.0, min(1.0, up[2])))), 2)


def _pose(stage, f):
    # Readable weld-approach reach; f in [0,1] interpolates v18 (small clearance)
    # to v19 (corrected clearance). Same mapping used for calibration and trials.
    _set_drive(stage, "right_shoulder_pitch", -95 + f * (-15))
    _set_drive(stage, "right_shoulder_roll", -8)
    _set_drive(stage, "right_elbow", 45 + f * (-15))
    _set_drive(stage, "right_wrist_pitch", 12)


def _reach(stage, f, steps=150):
    _pose(stage, f)
    app = omni.kit.app.get_app()
    for _ in range(steps):
        app.update()
    return _world_pos(stage, PALM)


def _box(stage, path, scale, center, color, opacity=1.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(*scale))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if opacity < 1.0:
        c.CreateDisplayOpacityAttr([opacity])


def _place(stage, path, center, scale):
    x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(*scale))


def setup_scene(args):
    """Build + calibrate the welding scene around an already-spawned G1."""
    stage = _stage()
    camera = args.get("camera", "/World/Cameras/BehaviorCI_PassFailCamera")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(120)
    stage.SetTimeCodesPerSecond(24)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Cameras")

    _box(stage, "/World/Ground", (6, 6, 0.1), (0, 0, -0.05), (0.18, 0.18, 0.2))
    _box(stage, "/World/Table", (0.6, 0.7, 0.06), (0.3, -0.58, 0.42), (0.55, 0.57, 0.6))
    _box(stage, SEAM, (0.16, 0.03, 0.012), (0, -0.5, 0.5), (1.0, 0.45, 0.05))
    _box(stage, OBSTACLE, (0.10, 0.10, 0.10), (0, -0.5, 0.5), (0.15, 0.35, 0.85))
    _box(
        stage, ZONE, (0.20, 0.22, 0.26), (0, -0.5, 0.5), (0.9, 0.07, 0.07), opacity=0.28
    )
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(900)
    kl = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    kl.CreateIntensityAttr(2200)
    klx = UsdGeom.Xformable(kl.GetPrim())
    klx.ClearXformOpOrder()
    klx.AddRotateXYZOp().Set(Gf.Vec3f(-45, 15, 0))
    cam = UsdGeom.Camera.Define(stage, camera)
    cam.CreateFocalLengthAttr(24.0)
    if not stage.GetPrimAtPath("/World/G1_BaseFix").IsValid():
        fj = UsdPhysics.FixedJoint.Define(stage, "/World/G1_BaseFix")
        fj.CreateBody1Rel().SetTargets([PELVIS])

    # Calibrate: where does the real arm settle for the regressed vs fixed pose?
    omni.timeline.get_timeline_interface().play()
    p18 = _reach(stage, 0.0)
    p19 = _reach(stage, 1.0)

    # Seam = good (v19) reach; obstacle + restricted zone = bad (v18) reach.
    _place(stage, SEAM, (p19[0], p19[1], p19[2]), (0.16, 0.03, 0.012))
    _place(stage, OBSTACLE, (p18[0], p18[1], p18[2]), (0.10, 0.10, 0.10))
    _place(stage, ZONE, (p18[0] + 0.02, p18[1], p18[2] + 0.02), (0.20, 0.22, 0.26))
    g1_min_z = _world_aabb(stage, "/G1")[0][2]
    _place(stage, "/World/Ground", (0.0, 0.0, g1_min_z - 0.05), (6, 6, 0.1))
    midx, midy = (p18[0] + p19[0]) / 2, (p18[1] + p19[1]) / 2
    topz = min(p18[2], p19[2]) - 0.03
    _place(stage, "/World/Table", (midx, midy, topz - 0.03), (0.6, 0.7, 0.06))

    M = Gf.Vec3d(midx, midy, (p18[2] + p19[2]) / 2)
    look = Gf.Matrix4d()
    look.SetLookAt(Gf.Vec3d(M[0] - 1.0, M[1] - 1.3, M[2] + 0.55), M, Gf.Vec3d(0, 0, 1))
    cx = UsdGeom.Xformable(stage.GetPrimAtPath(camera))
    cx.ClearXformOpOrder()
    cx.AddTransformOp().Set(look.GetInverse())

    omni.usd.get_context().save_stage()
    return {
        "p18": [round(p18[i], 3) for i in range(3)],
        "p19": [round(p19[i], 3) for i in range(3)],
        "camera": camera,
    }


def behavior_ci_run_trial(args):
    """Run one weld-approach trial on the real G1 and return measured metrics."""
    stage = _stage()
    controller = args.get("controller", {})
    run = int(args.get("run", 0))
    steps = int(args.get("steps", 150))
    margin = float(controller.get("clearance_margin_cm", 6.0))
    f = max(0.0, min(1.0, (margin - 6.0) / 8.0))

    palm = _reach(stage, f, steps)
    seam = _world_pos(stage, SEAM)
    obstacle = _world_pos(stage, OBSTACLE)
    zmin, zmax = _world_aabb(stage, ZONE)
    clearance = (palm - obstacle).GetLength()
    intruded = all(zmin[i] <= palm[i] <= zmax[i] for i in range(3))
    elapsed = round(steps / 60.0, 2)

    metrics = {
        "torch_tip_distance_to_target_cm": round((palm - seam).GetLength() * 100, 2),
        "collision_count": 1 if clearance < COLLISION_CLEARANCE_M else 0,
        "restricted_zone_intrusions": 1 if intruded else 0,
        "max_base_tilt_degrees": _base_tilt_deg(stage),
        "elapsed_seconds": elapsed,
        "min_clearance_to_obstacle_cm": round(clearance * 100, 2),
        "weld_point_xyz": [round(palm[0], 3), round(palm[1], 3), round(palm[2], 3)],
    }
    events = []
    if metrics["restricted_zone_intrusions"]:
        events.append(
            {
                "run": run,
                "time_seconds": elapsed,
                "code": "SAFETY_ZONE_INTRUSION",
                "message": "G1 weld point settled inside the red restricted zone",
            }
        )
    if metrics["collision_count"]:
        events.append(
            {
                "run": run,
                "time_seconds": elapsed,
                "code": "OBSTACLE_COLLISION",
                "message": f"G1 weld point {metrics['min_clearance_to_obstacle_cm']}cm from the obstacle",
            }
        )
    if metrics["torch_tip_distance_to_target_cm"] > 2.0 and not events:
        events.append(
            {
                "run": run,
                "time_seconds": elapsed,
                "code": "TARGET_MISS",
                "message": f"weld tip {metrics['torch_tip_distance_to_target_cm']}cm from seam",
            }
        )
    return {"metrics": metrics, "events": events, "trajectory_id": f"g1-run{run:02d}"}
