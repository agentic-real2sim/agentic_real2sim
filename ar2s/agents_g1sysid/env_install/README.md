# ar2s.agents_g1sysid — Environment Setup

`ar2s.agents_g1sysid` depends on **torch + mujoco + onnxruntime** which conflict with
the base `droid_sim` development environment. Keep them in a **separate
conda environment**.

---

## Why separate?

| | base `droid_sim` dev env | `g1sysid` (conda `g1sysid`) |
|---|---|---|
| Python | project default | 3.12 |
| Core packages | h5py, imageio, usd-core | torch, mujoco, onnxruntime, pydantic, scipy |
| Agent/tool stack | — | langchain-core, langchain-anthropic, langgraph |
| Install method | project default | `pip install -r requirements.txt` |

---

## Setup

### Option A — Conda (recommended)

```bash
# 1. Use a local copy with LFS assets (ONNX + STL meshes)
cd /path/to/agentic_real2sim
git lfs pull

# 2. Create the g1sysid env under .conda/g1sysid.
#    This installs ar2s editable plus requirements.txt and requirements_llm.txt.
bash ar2s/agents_g1sysid/env_install/install_g1sysid.sh

# 3. Activate by prefix
conda activate "$PWD/.conda/g1sysid"
```

`.conda` is a symlink to this machine's conda env root. The installer reuses an
existing `.conda` if the checkout already has one; otherwise it creates the
symlink pointing at `AR2S_CONDA_PATH`, falling back to
`/media/eric/data/conda_envs/droid_sim_envs` (the original developer's path)
when neither is present. On a fresh checkout, set the env root explicitly:

```bash
AR2S_CONDA_PATH=/path/to/conda_envs \
    bash ar2s/agents_g1sysid/env_install/install_g1sysid.sh
```

### Option B — pip into an existing conda env

If you already have torch + mujoco installed (e.g. in the `bfm0` conda env):

```bash
conda activate bfm0               # or any env with torch + mujoco
pip install -r ar2s/agents_g1sysid/env_install/requirements.txt
pip install -r ar2s/agents_g1sysid/env_install/requirements_llm.txt
pip install --no-deps -e .
```

### CUDA torch (optional)

The default `requirements.txt` installs CPU-only torch (sufficient because
the policy runs through ONNX, not PyTorch). For GPU torch:

```bash
# Replace the torch line with the CUDA build matching your driver:
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# or cu128 for newer drivers
```

---

## Running

After activating the `g1sysid` env, run from the editable repo install:

```bash
cd /path/to/droid_sim
conda activate "$PWD/.conda/g1sysid"

python -m ar2s.agents_g1sysid.cli.run_pipeline . \
    --skip-stage1 \
    --motion ar2s/agents_g1sysid/assets/motions/pose_variation_0.npz \
    --pkl    ar2s/agents_g1sysid/assets/motions/pose_variation_0.pkl \
    --task-type goal \
    --n-iters 200 \
    --wandb-project g1sysid-experiments
```

---

## Package summary

| Package | Why needed |
|---|---|
| `torch` | BFM-Zero `env.py` and model code import it at the module level |
| `mujoco` | Physics simulation (`MuJoCoBFMZeroEnv`) |
| `onnxruntime` | ONNX policy inference (`OnnxPolicy`) |
| `pydantic` | `FBcprAuxModel` config parsing (transitively imported by env.py) |
| `scipy` | `Rotation` used in `env.py` for quaternion math |
| `joblib` | Loading `.pkl` latent context files |
| `safetensors` | BFM-Zero model weight loading |
| `numpy` | Array math throughout |
| `imageio[ffmpeg]` | Rollout video saving |
| `pyyaml` | `summary.yaml` serialisation |
| `wandb` | Experiment tracking (optional but installed by default) |
| `langchain-core` | Stage-1 Motion Reader Agent tools (`@tool` decorator) |
| `langchain-anthropic` | Anthropic LLM backend for Stage-1 + convergence agents |
| `langgraph` | ReAct agent loop (`create_react_agent`) |
