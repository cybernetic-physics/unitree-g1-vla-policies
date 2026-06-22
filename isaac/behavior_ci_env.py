"""In-session Behavior CI entrypoint for the hosted Isaac backend.

This module is uploaded into the Cybernetic Physics Isaac session (under
``/data/workspace``) and is what ``cybernetics.behavior_ci``'s
``IsaacSessionAdapter`` invokes per trial via ``isaac.execute_script``:

    import behavior_ci_env
    result = behavior_ci_env.behavior_ci_run_trial(args)  # {"metrics": {...}, "events": [...]}

It drives a scripted weld-approach on the **real Unitree G1** articulation and
measures every metric from the robot's settled physical state — forward
kinematics of the right-hand link plus world-space geometry of the weld seam,
obstacle, and restricted zone. It is NOT a learned policy: the controller
parameter ``clearance_margin_cm`` drives a real reach whose measured outcome
(distance to seam, clearance to obstacle, zone intrusion, base tilt) is computed
from physics, not hardcoded. A real VLA/GR00T policy would replace the scripted
joint targets while keeping this measurement contract.

Validated live (rtx4090 session): clearance_margin_cm=6 → weld point settles on
the obstacle, inside the red zone, ~16cm from the seam (FAIL); =14 → reaches the
seam within ~0.6cm, ~16cm clear of the obstacle (PASS).
"""

import math

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

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


def behavior_ci_run_trial(args):
    """Run one weld-approach trial on the real G1 and return measured metrics."""
    stage = _stage()
    controller = args.get("controller", {})
    run = int(args.get("run", 0))
    steps = int(args.get("steps", 150))
    margin = float(controller.get("clearance_margin_cm", 6.0))
    # readable controller param -> reach extension; 0 at v18 (margin 6), 1 at v19 (margin 14)
    f = max(0.0, min(1.0, (margin - 6.0) / 8.0))

    _set_drive(stage, "right_shoulder_pitch", -95 + f * (-15))
    _set_drive(stage, "right_shoulder_roll", -8)
    _set_drive(stage, "right_elbow", 45 + f * (-15))
    _set_drive(stage, "right_wrist_pitch", 12)

    omni.timeline.get_timeline_interface().play()
    app = omni.kit.app.get_app()
    for _ in range(steps):
        app.update()

    palm = _world_pos(stage, PALM)
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
