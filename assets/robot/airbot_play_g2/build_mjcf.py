"""Build the AIRBOT Play + G2 combined MJCF from play_g2.urdf.

MuJoCo's URDF importer gives us correct kinematics / inertia / mesh transforms
but drops everything the sim needs on top: it ignores URDF <mimic>, adds no
actuators, and puts every visual geom in group 1 / collision geom in group 0.
This script compiles the URDF via MjSpec and augments it to match the contract
RobotSceneSim relies on (mirroring franka_emika_panda/panda.xml + robotiq 2f85):

  * visual geoms  -> group 2   (GEOM_GROUP_ROBOT_VISUAL)
  * collision geoms -> group 3 (GEOM_GROUP_ROBOT_COLLISION)
  * position actuators on the 6 arm joints + the gripper master g2_joint
  * the URDF mimic (g2_left/right_joint = 0.5 * g2_joint) realised as two
    <equality><joint> couplers (MuJoCo has no mimic)

Run:  python assets/robot/airbot_play_g2/build_mjcf.py
Out:  assets/robot/airbot_play_g2/airbot_play_g2.xml
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
URDF = HERE / "play_g2.urdf"
OUT = HERE / "airbot_play_g2.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]     # joint1..joint6 (revolute)
GRIPPER_MASTER = "g2_joint"                          # prismatic [0, 0.072]
FINGER_JOINTS = ["g2_left_joint", "g2_right_joint"]  # mimic master * 0.5

GEOM_GROUP_VISUAL = 2
GEOM_GROUP_COLLISION = 3


def main() -> None:
    os.chdir(HERE)                       # so meshdir="meshes" resolves
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.meshdir = "meshes"
    spec.modelname = "airbot_play_g2"

    # CRITICAL: keep fixed-joint bodies separate. With the default fusestatic,
    # compile() fuses the gripper's fixed chain (link6→eef_connect_base_link→
    # g2_base_link) into link6, and to_xml() then serialises the fused result
    # WRONG — it drops the accumulated arm_connect_joint offset (0.0955 m), so
    # the whole gripper collapses 9.55 cm back onto the wrist. Disabling
    # fusestatic preserves the true kinematics through the to_xml round-trip.
    spec.compiler.fusestatic = False

    # Offscreen framebuffer big enough for the pipeline's render passes.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # --- joint dynamics: armature + damping (stabilises the servo loop) ----
    # Mirrors panda.xml (armature 0.1 / damping 1 on the arm). The gripper
    # slides need armature too or the near-massless master DOF blows up under
    # the position servo + mimic equality constraints.
    for jn in ARM_JOINTS:
        j = spec.joint(jn)
        j.armature = 0.1
        j.damping = 1.0
    for jn in [GRIPPER_MASTER, *FINGER_JOINTS]:
        j = spec.joint(jn)
        j.armature = 0.01
        j.damping = 1.0

    # The URDF's g2_virtual_link (mimic master) is massless; give it a small
    # but non-degenerate mass/inertia so its DOF is well-conditioned.
    vlink = spec.body("g2_virtual_link")
    vlink.mass = 0.02
    vlink.inertia = [1e-5, 1e-5, 1e-5]
    vlink.explicitinertial = True

    # --- geom groups: visual (contype==0) -> 2, collision -> 3 -------------
    n_vis = n_col = 0
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            g.group = GEOM_GROUP_VISUAL
            n_vis += 1
        else:
            g.group = GEOM_GROUP_COLLISION
            n_col += 1

    # NOTE: finger contact params (friction / solimp / solref / priority) are
    # NOT set here — MjSpec.to_xml() truncates solimp and silently drops the
    # spin/roll friction on the round-trip. RobotSceneSim applies them post
    # compile from the RobotProfile (gripper_contact_*) instead.

    # --- actuators: position servos on arm joints + gripper master --------
    # Arm: stiff position servo so the arm holds the commanded joint angles.
    for jn in ARM_JOINTS:
        a = spec.add_actuator()
        a.name = f"act_{jn}"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = jn
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = 300.0
        a.biasprm[1] = -300.0             # biasprm = [0, -kp, -kv]
        a.biasprm[2] = -20.0
        jnt = spec.joint(jn)
        a.ctrlrange = jnt.range
        a.ctrllimited = 1

    # Gripper: position servo on the prismatic master g2_joint [0, 0.072].
    ag = spec.add_actuator()
    ag.name = f"act_{GRIPPER_MASTER}"
    ag.trntype = mujoco.mjtTrn.mjTRN_JOINT
    ag.target = GRIPPER_MASTER
    ag.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    ag.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    ag.gainprm[0] = 200.0
    ag.biasprm[1] = -200.0
    ag.biasprm[2] = -5.0
    ag.ctrlrange = spec.joint(GRIPPER_MASTER).range
    ag.ctrllimited = 1

    # --- realise the URDF mimic: finger = 0.5 * master --------------------
    for fj in FINGER_JOINTS:
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = fj
        eq.name2 = GRIPPER_MASTER
        eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
        # polycoef: fj = c0 + c1*master + ... ; mimic multiplier 0.5, offset 0
        eq.data[:5] = [0.0, 0.5, 0.0, 0.0, 0.0]

    m = spec.compile()   # validate
    print(f"[build_mjcf] compiled OK: nq={m.nq} nu={m.nu} "
          f"neq={m.neq} nbody={m.nbody}")
    print(f"[build_mjcf] geoms: visual(g2)={n_vis} collision(g3)={n_col}")

    OUT.write_text(spec.to_xml())
    print(f"[build_mjcf] wrote {OUT}")


if __name__ == "__main__":
    main()
