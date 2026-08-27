"""Stage 5: RGB + masks -> .glb mesh per object (qianjun_sam3d env), batched.

Env: ``qianjun_sam3d``. Entrypoint is
``ar2s/agents_visual/_toolkit/scripts/run_sam3d_batch.py`` — a single-pass
wrapper around sam-3d-objects' ``notebook.inference`` that loads the pipeline
ONCE and reconstructs every job in the list (one ~13 GB model load instead of
one per object). Each job carries its own RGB frame + mask, so different
objects can be meshed from different per-object keyframes. Produced mesh scale
is arbitrary — overridden in stage 6 (mesh_scale).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from ar2s.agents_visual._toolkit._runners import MODELS_ROOT, env_for_stage, run_in_env, script_path


@dataclass
class MeshRecoverJob:
    object_name: str
    rgb_frame_path: str                     # frame to feed sam3d-objects
    mask_path: str                          # binary mask for the object on that frame
    output_glb_path: str


@dataclass
class MeshRecoverInput:
    jobs: list[MeshRecoverJob]


@dataclass
class ObjectMeshInfo:
    object_name: str
    success: bool
    glb_path: str = ""
    error: str = ""


@dataclass
class MeshRecoverReport:
    success: bool                           # True iff at least one job succeeded
    error: str = ""
    objects: list[ObjectMeshInfo] = field(default_factory=list)


def run(inp: MeshRecoverInput, *, log_dir: str) -> MeshRecoverReport:
    if not inp.jobs:
        return MeshRecoverReport(success=False, error="jobs is empty")

    for job in inp.jobs:
        Path(job.output_glb_path).parent.mkdir(parents=True, exist_ok=True)

    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    jobs_json = log_root / "mesh_recover_jobs.json"
    jobs_json.write_text(json.dumps([
        {
            "object_name":     j.object_name,
            "image_path":      j.rgb_frame_path,
            "mask_path":       j.mask_path,
            "output_glb_path": j.output_glb_path,
        }
        for j in inp.jobs
    ], indent=2))

    # pipeline.yaml uses bare-filename references for sibling ckpts, resolved
    # relative to the yaml's dir. The hardened model layout keeps the HF
    # snapshot's checkpoint files under MODELS_ROOT/sam_3d_objects/hf/checkpoints.
    pipeline_yaml = MODELS_ROOT / "sam_3d_objects" / "hf" / "checkpoints" / "pipeline.yaml"
    result = run_in_env(
        env=env_for_stage("mesh_recover"),
        script_path=script_path("mesh_recover"),
        args=[
            "--jobs_json", str(jobs_json),
            "--config",    str(pipeline_yaml),
        ],
        log_dir=log_dir,
        stage="mesh_recover",
    )

    objects: list[ObjectMeshInfo] = []
    for job in inp.jobs:
        sidecar = Path(job.output_glb_path + ".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            if meta.get("success"):
                objects.append(ObjectMeshInfo(
                    object_name=job.object_name, success=True, glb_path=job.output_glb_path,
                ))
            else:
                objects.append(ObjectMeshInfo(
                    object_name=job.object_name, success=False,
                    error=f"sam3d inference failed: {meta.get('error', '<no error in sidecar>')}",
                ))
        else:
            objects.append(ObjectMeshInfo(
                object_name=job.object_name, success=False,
                error=(
                    f"no sidecar for {job.object_name} (rc={result.returncode}); "
                    f"see {result.stderr_path}\n{result.stderr_tail}"
                ),
            ))

    n_ok = sum(1 for o in objects if o.success)
    return MeshRecoverReport(
        success=(n_ok > 0),
        error="" if n_ok > 0 else "all mesh_recover jobs failed",
        objects=objects,
    )
