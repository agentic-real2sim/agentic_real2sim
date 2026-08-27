"""ar2s.agents_geometry_prior — geometry-prior inference between visual and sysid.

Sits between the visual agent (raw geometry: mesh + scale + FP pose) and
the scene-prep stages (collision, camera select, calibration). Job:
fix per-object body-frame orientation, snap each object's most-vertical
axis to world +Z, and decide per-edge support anchors from the support
tree so the emitted ``manifest.yaml`` carries the right
``objects[].pos_offset`` + ``ground.offset`` before any physics happens.

Mirrors ``ar2s.agents_physical_prior``'s layout (peer stage, NOT a scene-prep
subskill). Output is a *prior* — downstream scene-prep skills consume the
patched manifest and never re-derive ground geometry.

Public surface (planned):
    ``apply.apply_geometry_priors(episode_dir, run_root) -> dict``
    ``cli.run_pipeline`` — ``python -m ar2s.agents_geometry_prior.cli.run_pipeline``

Skills:
    ``mesh_orient``    — VLM 4-候选 body-frame flip (NONE / flipX / flipY / flipZ)
    ``mesh_axis_align`` — geometric SO(3) snap of most-vertical body axis to +Z
    ``z_anchor``        — VLM ANCHOR_TO_UPPER/LOWER + deterministic tree walk
"""
