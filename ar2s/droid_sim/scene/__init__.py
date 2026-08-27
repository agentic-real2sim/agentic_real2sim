"""SceneConfig schema + generic RobotSceneSim (see docs/agent_pipeline/architecture.md §3).

Import the submodules directly (``from ar2s.droid_sim.scene.config import
SceneObject``). The package used to re-export SceneConfig/RobotSceneSim/... here,
which meant every import under ``scene/`` pulled mujoco and h5py through
robot_scene_sim -- the standalone scene-data path reads ``shift`` and
``config`` using numpy alone, so the re-exports are gone rather than lazy. No
caller used them.
"""
