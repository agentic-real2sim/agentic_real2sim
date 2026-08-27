#!/usr/bin/env bash
#
# Shared conda-prefix helpers for repository env installers.
#
# Env root policy:
#   AR2S_CONDA_PATH=/some/path bash scripts/env_install_*/install_*.sh
# defaults to:
#   /media/eric/data/conda_envs/droid_sim_envs
# Package downloads/cache default to:
#   $AR2S_CONDA_PATH/pkgs
# Pip cache defaults to:
#   $AR2S_CONDA_PATH/pip-cache
# Warp kernel cache defaults to:
#   $AR2S_CONDA_PATH/warp-cache
# Temporary build/unpack files default to:
#   $AR2S_CONDA_PATH/tmp
# Python user-site packages are disabled so envs do not silently depend on
# packages under ~/.local.
#
# Install scripts create envs under $REPO_ROOT/.conda/<env-name>. By default,
# .conda is a symlink to AR2S_CONDA_PATH, so conda prefixes stay off the default
# base envs directory while commands still have a stable repo-local path. On
# cluster/cloud filesystems AR2S_CONDA_PATH may point directly at
# $REPO_ROOT/.conda; in that case .conda remains a real directory. Conda's
# package/cache directories are redirected under AR2S_CONDA_PATH unless the
# corresponding AR2S_* override is set explicitly.

AR2S_DEFAULT_CONDA_PATH="/media/eric/data/conda_envs/droid_sim_envs"

ar2s_conda_env_root() {
    local root="${AR2S_CONDA_PATH:-$AR2S_DEFAULT_CONDA_PATH}"
    root="${root%/}"
    if [ -z "$root" ]; then
        echo "ERROR: AR2S_CONDA_PATH resolved to an empty path." >&2
        exit 1
    fi
    printf '%s\n' "$root"
}

ar2s_source_conda() {
    local conda_base
    if [ -n "${CONDA_EXE:-}" ]; then
        conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
    else
        echo "ERROR: conda not on PATH; install miniforge/miniconda first." >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
}

ar2s_conda_pkgs_root() {
    local env_root
    local root

    env_root="$(ar2s_conda_env_root)"
    root="${AR2S_CONDA_PKGS_DIRS:-$env_root/pkgs}"
    root="${root%/}"
    printf '%s\n' "$root"
}

ar2s_pip_cache_root() {
    local env_root
    local root

    env_root="$(ar2s_conda_env_root)"
    root="${AR2S_PIP_CACHE_DIR:-$env_root/pip-cache}"
    root="${root%/}"
    printf '%s\n' "$root"
}

ar2s_warp_cache_root() {
    local env_root
    local root

    env_root="$(ar2s_conda_env_root)"
    root="${AR2S_WARP_CACHE_PATH:-$env_root/warp-cache}"
    root="${root%/}"
    printf '%s\n' "$root"
}

ar2s_tmp_root() {
    local env_root
    local root

    env_root="$(ar2s_conda_env_root)"
    root="${AR2S_TMPDIR:-$env_root/tmp}"
    root="${root%/}"
    printf '%s\n' "$root"
}

ar2s_xdg_cache_root() {
    local env_root
    local root

    env_root="$(ar2s_conda_env_root)"
    root="${AR2S_XDG_CACHE_HOME:-${XDG_CACHE_HOME:-$env_root/xdg-cache}}"
    root="${root%/}"
    printf '%s\n' "$root"
}

ar2s_torch_cache_root() {
    local env_root
    local root

    env_root="$(ar2s_conda_env_root)"
    root="${AR2S_TORCH_HOME:-${TORCH_HOME:-$env_root/torch-cache}}"
    root="${root%/}"
    printf '%s\n' "$root"
}

ar2s_hf_cache_root() {
    local env_root
    local root

    env_root="$(ar2s_conda_env_root)"
    root="${AR2S_HF_HOME:-${HF_HOME:-$env_root/huggingface}}"
    root="${root%/}"
    printf '%s\n' "$root"
}

ar2s_prepare_conda_env_root() {
    local repo_root="$1"
    local env_root
    local pkgs_root
    local pip_cache_root
    local warp_cache_root
    local tmp_root
    local xdg_cache_root
    local torch_cache_root
    local hf_cache_root
    local link_path
    local env_root_abs
    local link_path_abs

    env_root="$(ar2s_conda_env_root)"
    pkgs_root="$(ar2s_conda_pkgs_root)"
    pip_cache_root="$(ar2s_pip_cache_root)"
    warp_cache_root="$(ar2s_warp_cache_root)"
    tmp_root="$(ar2s_tmp_root)"
    xdg_cache_root="$(ar2s_xdg_cache_root)"
    torch_cache_root="$(ar2s_torch_cache_root)"
    hf_cache_root="$(ar2s_hf_cache_root)"
    link_path="$repo_root/.conda"
    env_root_abs="$(realpath -m "$env_root")"
    link_path_abs="$(realpath -m "$link_path")"

    mkdir -p "$env_root"
    mkdir -p "$pkgs_root"
    mkdir -p "$pip_cache_root"
    mkdir -p "$warp_cache_root"
    mkdir -p "$tmp_root"
    mkdir -p "$xdg_cache_root"
    mkdir -p "$torch_cache_root"
    mkdir -p "$hf_cache_root"
    export CONDA_PKGS_DIRS="$pkgs_root"
    export PIP_CACHE_DIR="$pip_cache_root"
    export WARP_CACHE_PATH="$warp_cache_root"
    export TMPDIR="$tmp_root"
    export TEMP="$tmp_root"
    export TMP="$tmp_root"
    export XDG_CACHE_HOME="$xdg_cache_root"
    export TORCH_HOME="$torch_cache_root"
    export HF_HOME="$hf_cache_root"
    export PYTHONNOUSERSITE=1

    if [ "$env_root_abs" = "$link_path_abs" ]; then
        return
    fi

    if [ -e "$link_path" ] && [ ! -L "$link_path" ]; then
        echo "ERROR: $link_path exists but is not a symlink." >&2
        echo "       Move it aside or set AR2S_CONDA_PATH explicitly." >&2
        exit 1
    fi

    if [ -L "$link_path" ]; then
        rm "$link_path"
    fi
    ln -s "$env_root" "$link_path"
}

ar2s_write_conda_activation_hooks() {
    local env_prefix="$1"
    local activate_dir="$env_prefix/etc/conda/activate.d"
    local hook_path="$activate_dir/ar2s_cache_paths.sh"

    mkdir -p "$activate_dir"
    {
        printf 'export CONDA_PKGS_DIRS=%q\n' "$(ar2s_conda_pkgs_root)"
        printf 'export PIP_CACHE_DIR=%q\n' "$(ar2s_pip_cache_root)"
        printf 'export WARP_CACHE_PATH=%q\n' "$(ar2s_warp_cache_root)"
        printf 'export TMPDIR=%q\n' "$(ar2s_tmp_root)"
        printf 'export TEMP=%q\n' "$(ar2s_tmp_root)"
        printf 'export TMP=%q\n' "$(ar2s_tmp_root)"
        printf 'export XDG_CACHE_HOME=%q\n' "$(ar2s_xdg_cache_root)"
        printf 'export TORCH_HOME=%q\n' "$(ar2s_torch_cache_root)"
        printf 'export HF_HOME=%q\n' "$(ar2s_hf_cache_root)"
        printf 'export PYTHONNOUSERSITE=1\n'
        printf 'export LD_LIBRARY_PATH=%q:%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' \
            "$env_prefix/targets/x86_64-linux/lib" "$env_prefix/lib"
        # Override CUDA_HOME / PATH to point at the env's own nvcc.
        # Some host images (vast.ai PyTorch base) set CUDA_HOME=/usr/local/cuda
        # at Docker layer to a different CUDA version than the env was built
        # against. torch.utils.cpp_extension's CUDA-version check then refuses
        # the source build with "detected CUDA X.Y mismatches PyTorch cuZ.W".
        # Forcing CUDA_HOME to $CONDA_PREFIX uses the env's nvcc and matches
        # the bundled torch wheel.
        printf 'export CUDA_HOME=%q\n' "$env_prefix"
        printf 'export PATH=%q${PATH:+:$PATH}\n' "$env_prefix/bin:"
    } > "$hook_path"
}

ar2s_conda_env_prefix() {
    local repo_root="$1"
    local env_name="$2"
    printf '%s/.conda/%s\n' "$repo_root" "$env_name"
}

ar2s_conda_env_exists() {
    local env_prefix="$1"
    [ -d "$env_prefix/conda-meta" ]
}

ar2s_print_env_location() {
    local env_name="$1"
    local env_prefix="$2"
    echo "env name   : $env_name"
    echo "env prefix : $env_prefix"
    echo "env root   : $(ar2s_conda_env_root)"
    echo "pkg cache  : ${CONDA_PKGS_DIRS:-<conda default>}"
    echo "pip cache  : ${PIP_CACHE_DIR:-<pip default>}"
    echo "warp cache : ${WARP_CACHE_PATH:-<warp default>}"
    echo "tmp dir    : ${TMPDIR:-<system default>}"
    echo "xdg cache  : ${XDG_CACHE_HOME:-<system default>}"
    echo "torch cache: ${TORCH_HOME:-<torch default>}"
    echo "hf cache   : ${HF_HOME:-<huggingface default>}"
    echo "no user site: ${PYTHONNOUSERSITE:-<python default>}"
}

ar2s_create_prefix_env_or_skip() {
    local env_name="$1"
    local env_prefix="$2"
    local python_version="$3"

    ar2s_print_env_location "$env_name" "$env_prefix"
    if ar2s_conda_env_exists "$env_prefix"; then
        echo "    [skip] env '$env_name' already exists at $env_prefix"
    else
        conda create -y -p "$env_prefix" "python=$python_version" pip
    fi
    ar2s_write_conda_activation_hooks "$env_prefix"
}

ar2s_create_prefix_env_or_fail() {
    local env_name="$1"
    local env_prefix="$2"
    local python_version="$3"

    ar2s_print_env_location "$env_name" "$env_prefix"
    if ar2s_conda_env_exists "$env_prefix"; then
        echo "WARNING: $env_name already exists at $env_prefix"
        echo "Remove with: conda env remove -p \"$env_prefix\""
        exit 1
    fi

    conda create -y -p "$env_prefix" "python=$python_version" pip
    ar2s_write_conda_activation_hooks "$env_prefix"
}
