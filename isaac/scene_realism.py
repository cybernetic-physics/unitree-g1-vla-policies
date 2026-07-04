"""cicd_ship_yard realism pass — reproducible in-session dressing + CI-prim authoring.

Provenance for the saved environment pinned by ``tasks/g1_weld_approach/task.py``
(``env_id = "env_f4b83937bc980161"``, environment name ``cicd_ship_yard``). The
scene was converged by a capture -> critique -> refine loop over a live hosted
Isaac session (7 iterations, before/after in ``assets/scene-previews/``), then
published from that session; this module re-authors the same result on a fresh
session that already contains the Unitree G1 (``/G1``) and the pipe-yard props
(``/World/Pipe``, ``/World/Props``, ``/World/SiteGround``).

Run order on a live session::

    dress_scene({})       # materials + layout + calibrated CI prims + camera
    verify_contract({})   # v18 must fail, v19 must pass, measured from physics

What it fixes over the raw authored yard (the loop's findings, in order):

1.  The G1 rendered as chrome — its OmniPBR materials ship ``metallic_constant=1``
    with mid-grey diffuse. Real G1s are matte dark polymer; set diffuse/metallic/
    roughness on the robot's own ``Looks`` materials (they are bound inside
    instanced link scopes, so binding a new material at ``/G1`` does NOT work).
2.  The pipe was unlit white plastic — bind carbon-steel PBR to the sections and
    a scorched heat-affected band to the weld ring; dim the arc light so the ring
    doesn't read as a glowing strap.
3.  The ground tiled visibly — bind the asset-server ``Base/Natural/Dirt.mdl``
    with world-space projection instead of the small-scale placeholder texture.
4.  The pipe floated mid-air — put it on steel jack stands at the CALIBRATED
    working height, its robot-facing surface through the measured v19 reach
    (a 2G-position side-wall weld), and plant the G1's legs in a braced stance.
5.  The CI contract prims did not exist in the yard scene — author
    ``/World/WeldSeam`` (bead marker at the v19 reach), ``/World/Obstacle`` (+ a
    clamp-jaw visual) at the v18 reach, ``/World/RestrictedZone`` sized so it
    covers the v18 approach but EXCLUDES the seam (the stance change shifts the
    settled reach ~5 cm — always recalibrate after posture edits, or a perfect
    weld "intrudes"), and the fixed pass/fail camera.

Everything scene-metric is MEASURED from the settled physics state (the same
``_pose`` mapping the grader drives), never hardcoded, so the dressing survives
robot/asset updates: recalibrate, re-place, re-verify.
"""

import json
import math

import omni.client
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

PALM = "/G1/right_wrist_yaw_link/right_hand_palm_link"
JOINT = "/G1/joints/{}_joint"
SEAM = "/World/WeldSeam"
OBSTACLE = "/World/Obstacle"
ZONE = "/World/RestrictedZone"
CAMERA = "/World/Cameras/BehaviorCI_PassFailCamera"
COLLISION_CLEARANCE_M = 0.09
RING_R = 0.142  # weld-ring outer radius; the pipe axis sits one ring radius behind the seam

ASSETS_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
)


def _stage():
    return omni.usd.get_context().get_stage()


def _set_drive(stage, joint, degrees):
    prim = stage.GetPrimAtPath(JOINT.format(joint))
    drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(
        prim, "angular"
    )
    drive.GetTargetPositionAttr().Set(float(degrees))


def _pose(stage, f):
    # Same v18->v19 interpolation the grader drives; calibration MUST match it.
    _set_drive(stage, "right_shoulder_pitch", -95 + f * (-15))
    _set_drive(stage, "right_shoulder_roll", -8)
    _set_drive(stage, "right_elbow", 45 + f * (-15))
    _set_drive(stage, "right_wrist_pitch", 12)


def _settle(steps=150):
    app = omni.kit.app.get_app()
    for _ in range(steps):
        app.update()
    t = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(_stage().GetPrimAtPath(PALM))
        .ExtractTranslation()
    )
    return Gf.Vec3d(t[0], t[1], t[2])


def _world_pos(stage, path):
    t = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        .ExtractTranslation()
    )
    return Gf.Vec3d(t[0], t[1], t[2])


def _make_pbr(stage, path, diffuse, rough, metal, emissive=None, opacity=None):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*emissive)
        )
    if opacity is not None:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _set_omnipbr(stage, shader_path, **inputs):
    sh = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
    if not sh.GetPrim().IsValid():
        return False
    for name, value in inputs.items():
        inp = sh.GetInput(name)
        if not inp:
            tn = (
                Sdf.ValueTypeNames.Color3f
                if isinstance(value, tuple)
                else Sdf.ValueTypeNames.Float
            )
            inp = sh.CreateInput(name, tn)
        inp.Set(Gf.Vec3f(*value) if isinstance(value, tuple) else value)
    return True


def _box(stage, path, scale, center, mat=None, opacity=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(*scale))
    if opacity is not None:
        c.CreateDisplayOpacityAttr([opacity])
    if mat:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def _cyl(stage, path, radius, height, center, mat):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr("Z")
    x = UsdGeom.Xformable(c.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


def _move(stage, path, pos, scale=None):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(*pos))
    if scale is not None and len(ops) > 1:
        ops[1].Set(Gf.Vec3f(*scale))


def _materials_pass(stage):
    # 1. G1 finish: edit the robot's OWN Looks materials (bound inside instances).
    for shader in (
        "/G1/Looks/DefaultMaterial/DefaultMaterial",
        "/G1/left_hand/Looks/DefaultMaterial/DefaultMaterial",
        "/G1/right_hand/Looks/DefaultMaterial/DefaultMaterial",
    ):
        _set_omnipbr(
            stage,
            shader,
            diffuse_color_constant=(0.115, 0.12, 0.13),
            metallic_constant=0.12,
            reflection_roughness_constant=0.55,
        )
    _set_omnipbr(
        stage,
        "/G1/Looks/DefaultMaterial_0/DefaultMaterial",
        diffuse_color_constant=(0.04, 0.04, 0.045),
        metallic_constant=0.1,
        reflection_roughness_constant=0.6,
    )

    # 2. Pipe steel + scorched weld ring; keep the arc light subtle.
    steel = _make_pbr(stage, "/World/Looks/PipeSteel", (0.38, 0.39, 0.41), 0.38, 0.9)
    haz = _make_pbr(
        stage,
        "/World/Looks/WeldHAZ",
        (0.11, 0.10, 0.105),
        0.5,
        0.85,
        emissive=(0.02, 0.004, 0.001),
    )
    for path, mat in (
        ("/World/Pipe/SectionA", steel),
        ("/World/Pipe/SectionB", steel),
        ("/World/Pipe/WeldSeam", haz),
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    ring = stage.GetPrimAtPath("/World/Pipe/WeldSeam")
    if ring.IsValid():
        UsdGeom.Cylinder(ring).GetHeightAttr().Set(0.035)
    arc = stage.GetPrimAtPath("/World/Pipe/WeldArc")
    if arc.IsValid():
        UsdLux.SphereLight(arc).GetIntensityAttr().Set(600)
    for spark in ("/World/WeldSpark", "/World/WeldSparkLight"):
        prim = stage.GetPrimAtPath(spark)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()

    # 3. Ground: real dirt MDL, world-projected so nothing visibly tiles.
    if omni.client.stat(ASSETS_ROOT + "/NVIDIA/Materials/Base/Natural/Dirt.mdl")[0] == (
        omni.client.Result.OK
    ):
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=ASSETS_ROOT + "/NVIDIA/Materials/Base/Natural/Dirt.mdl",
            mtl_name="Dirt",
            mtl_path="/World/Looks/SiteDirtReal",
        )
        _set_omnipbr(stage, "/World/Looks/SiteDirtReal/Shader", project_uvw=True)
        sh = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/SiteDirtReal/Shader"))
        (
            sh.GetInput("texture_scale")
            or sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)
        ).Set(Gf.Vec2f(0.35, 0.35))
        ground = stage.GetPrimAtPath("/World/SiteGround")
        if ground.IsValid():
            UsdShade.MaterialBindingAPI.Apply(ground).Bind(
                UsdShade.Material(stage.GetPrimAtPath("/World/Looks/SiteDirtReal"))
            )


def _stance_pass(stage):
    # Braced standing legs; the fixed-base default leaves them dangling backwards.
    for side in ("left", "right"):
        for joint, deg in (
            ("hip_pitch", -18),
            ("hip_roll", 0),
            ("hip_yaw", 0),
            ("knee", 32),
            ("ankle_pitch", -14),
            ("ankle_roll", 0),
        ):
            _set_drive(stage, f"{side}_{joint}", deg)


def _layout_pass(stage):
    # Keep the camera sightline (from -X/-Y toward the work zone) clear of props.
    _move(stage, "/World/Props/GasCart", (-0.35, 1.15, 0))
    _move(stage, "/World/Props/DrumA", (3.4, 3.2, 0))
    _move(stage, "/World/Props/DrumB", (3.8, 2.4, 0))


def calibrate(stage):
    """Measure where the real arm settles for the v18 and v19 poses (twice each)."""
    omni.timeline.get_timeline_interface().play()
    _pose(stage, 0.0)
    p18a = _settle()
    p18 = _settle(60)
    _pose(stage, 1.0)
    p19a = _settle()
    p19 = _settle(60)
    if (p18 - p18a).GetLength() > 0.005 or (p19 - p19a).GetLength() > 0.005:
        raise RuntimeError("calibration not repeatable; check physics stepping")
    return p18, p19


def _ci_prims_pass(stage, p18, p19):
    steel_dk = _make_pbr(stage, "/World/Looks/StandSteel", (0.16, 0.17, 0.19), 0.5, 0.85)
    clamp = _make_pbr(stage, "/World/Looks/ClampSteel", (0.55, 0.35, 0.08), 0.55, 0.75)
    bead = _make_pbr(stage, "/World/Looks/WeldBead", (0.32, 0.28, 0.26), 0.45, 0.9)
    zone_red = _make_pbr(
        stage, "/World/Looks/ZoneRed", (0.85, 0.06, 0.06), 0.9, 0.0, opacity=0.13
    )

    # Pipe: robot-facing surface through the v19 reach (side-wall / 2G weld).
    axis = Gf.Vec3d(p19[0] + RING_R, p19[1], p19[2])
    ring = _world_pos(stage, "/World/Pipe/WeldSeam")
    delta = axis - ring
    pipe = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Pipe"))
    op = pipe.GetOrderedXformOps()[0]
    old = op.Get()
    op.Set(Gf.Vec3d(old[0] + delta[0], old[1] + delta[1], old[2] + delta[2]))

    # Jack stands under both sections, ground to pipe bottom.
    pipe_bot = p19[2] - 0.145
    UsdGeom.Xform.Define(stage, "/World/PipeStands")
    for tag, y in (("A", -0.95), ("B", 0.95)):
        root = f"/World/PipeStands/Stand{tag}"
        UsdGeom.Xform.Define(stage, root)
        _box(stage, root + "/Base", (0.34, 0.34, 0.025), (axis[0], y, 0.0125), steel_dk)
        _cyl(
            stage,
            root + "/Post",
            0.032,
            pipe_bot - 0.06,
            (axis[0], y, (pipe_bot - 0.06) / 2 + 0.025),
            steel_dk,
        )
        _box(stage, root + "/Head", (0.18, 0.24, 0.035), (axis[0], y, pipe_bot - 0.018), steel_dk)

    # CI contract prims, placed from the measured reaches.
    _box(stage, SEAM, (0.16, 0.03, 0.012), tuple(p19), bead)
    _box(stage, OBSTACLE, (0.10, 0.10, 0.10), tuple(p18), clamp)
    _box(
        stage,
        "/World/Obstacle_ClampJaw",
        (0.12, 0.04, 0.03),
        (p18[0], p18[1], p18[2] + 0.06),
        steel_dk,
    )
    # Zone covers the v18 approach but its top stays BELOW the seam: 0.18 z-extent
    # centred 3 cm under p18 keeps p19 (14.7 cm above p18) outside.
    _box(
        stage,
        ZONE,
        (0.20, 0.22, 0.18),
        (p18[0] + 0.02, p18[1], p18[2] - 0.03),
        zone_red,
        opacity=0.13,
    )

    # Fixed pass/fail camera: 3/4 front vantage, seam + obstacle + zone in frame.
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    cam = UsdGeom.Camera.Define(stage, CAMERA)
    cam.CreateFocalLengthAttr(21.0)
    mid = (p18 + p19) / 2.0
    look = Gf.Matrix4d()
    look.SetLookAt(
        Gf.Vec3d(mid[0] - 1.7, mid[1] - 2.1, mid[2] + 0.75),
        Gf.Vec3d(mid[0], mid[1], mid[2] - 0.15),
        Gf.Vec3d(0, 0, 1),
    )
    x = UsdGeom.Xformable(cam.GetPrim())
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(look.GetInverse())


def dress_scene(args):
    """Full realism pass: materials, stance, layout, calibrated CI prims, camera."""
    stage = _stage()
    _materials_pass(stage)
    _stance_pass(stage)
    _layout_pass(stage)
    p18, p19 = calibrate(stage)
    _ci_prims_pass(stage, p18, p19)
    omni.usd.get_context().save_stage()
    return {
        "p18": [round(v, 3) for v in p18],
        "p19": [round(v, 3) for v in p19],
        "camera": CAMERA,
    }


def verify_contract(args):
    """Drive v18/v19 and confirm the demo story holds in THIS scene, from physics."""
    stage = _stage()
    omni.timeline.get_timeline_interface().play()
    results = {}
    for name, f in (("v18", 0.0), ("v19", 1.0)):
        _pose(stage, f)
        palm = _settle()
        seam = _world_pos(stage, SEAM)
        obstacle = _world_pos(stage, OBSTACLE)
        rng = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeWorldBound(stage.GetPrimAtPath(ZONE))
            .ComputeAlignedRange()
        )
        zmin, zmax = rng.GetMin(), rng.GetMax()
        clearance = (palm - obstacle).GetLength()
        results[name] = {
            "tip_cm": round((palm - seam).GetLength() * 100, 2),
            "clearance_cm": round(clearance * 100, 2),
            "collision": clearance < COLLISION_CLEARANCE_M,
            "zone_intrusion": all(zmin[i] <= palm[i] <= zmax[i] for i in range(3)),
        }
    v18, v19 = results["v18"], results["v19"]
    results["verdict"] = {
        "v18_fails": v18["collision"] or v18["zone_intrusion"] or v18["tip_cm"] > 2.0,
        "v19_passes": (
            not v19["collision"] and not v19["zone_intrusion"] and v19["tip_cm"] <= 2.0
        ),
    }
    if not (results["verdict"]["v18_fails"] and results["verdict"]["v19_passes"]):
        raise RuntimeError("contract broken: " + json.dumps(results))
    return results
