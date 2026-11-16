# Containers

Two images are built from this directory:

| Image | Dockerfile | Contents | Default command |
|---|---|---|---|
| `distillgapbench:cpu` | `Dockerfile` | Debian, CPython 3.12.13, the controller and tests | test suite and sandbox check |
| `distillgapbench:gpu` | `Dockerfile.gpu` | `vllm/vllm-openai:v0.24.0` plus the same controller | test suite and sandbox check |

Both pin the versions used for v1: CPython 3.12.13 (python-build-standalone, installed by uv 0.13.6), vLLM 0.24.0,
torch 2.11.0, `anthropic` 0.75.0, `openai` 2.41.0. One difference: the official vLLM image ships the CUDA 13.0 build
of torch; the study ran the CUDA 12.9 build. Both base images are published for linux/arm64 and linux/amd64.

Tested on Isambard-AI (NVIDIA GH200, linux/arm64): both images built with Podman; the test suite and the sandbox
check passed under Podman (with `--security-opt unmask=ALL`) and under Apptainer; the GPU is visible under Podman
(CDI) and Apptainer (`--nv`). The amd64 images build from the same recipe but have not been run on GPUs.

## Where things live inside the image

| Path | What | Visible to sandboxed code |
|---|---|---|
| `/opt/distillgapbench` | the harness | no |
| `/opt/python` | the controller and sandbox interpreter | only a private copy of the interpreter |
| `/home/dg/runs` | results (mount a host directory here) | no |
| `/home/dg/.cache/huggingface` | student weights (mount your cache here) | no |

Every task the harness executes, whether a researcher's reference solution, a student's completion or a hidden-test
check, runs in the namespace sandbox (`sandbox/`). The sandbox has no network and its own process and mount
namespaces. `/home`, `/opt`, `/tmp`, `/var`, `/run`, `/mnt`, `/srv`, `/media` and the cluster data mounts are
replaced by empty filesystems, and the environment is reduced to `PATH`, `LANG` and `PYTHONHASHSEED`. Keep results
and keys under those paths, and pass keys as environment variables or in a file under `/home`.

## Docker

The sandbox creates Linux user namespaces. Docker's default seccomp and AppArmor profiles block that, and Docker's
masked `/proc` paths stop the sandbox from mounting its own `/proc`, so add:

```sh
--security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined
```

These relax the container's own confinement; the sandbox inside still isolates the code it runs. GPU runs need the
NVIDIA Container Toolkit (`--gpus all`). On Ubuntu 24.04 hosts, unprivileged user namespaces may also be restricted
by `kernel.apparmor_restrict_unprivileged_userns`; `scripts/check_sandbox.py` reports what is missing.

## Podman

Rootless Podman keeps its default seccomp profile; the sandbox only needs the masked `/proc` paths lifted:

```sh
podman build -t distillgapbench:cpu .
podman run --rm --network none --security-opt unmask=ALL localhost/distillgapbench:cpu
```

For GPUs, use the CDI device of the NVIDIA Container Toolkit (`--device nvidia.com/gpu=all`).

## Apptainer and Slurm

On clusters, build the image once with Podman or Docker, convert it, and run it with `--nv`:

```sh
podman build -f Dockerfile.gpu -t distillgapbench:gpu .
podman save --format docker-archive -o distillgapbench-gpu.tar localhost/distillgapbench:gpu
apptainer build distillgapbench-gpu.sif docker-archive://distillgapbench-gpu.tar
```

Apptainer runs as your own user and mounts your home directory. Point the caches and results at a scratch directory
the sandbox hides (anything under `/home`, `/scratch`, `/projects`, `/local`, `/tmp` or `/mnt`):

```sh
#!/bin/bash
#SBATCH --gpus=1 --cpus-per-task=16 --mem=128G --time=02:00:00
W=/scratch/$USER/distillgapbench
mkdir -p "$W"/{hf,cache,runs}
apptainer exec --nv --bind "$W" \
  --env HF_HOME="$W/hf" --env XDG_CACHE_HOME="$W/cache" --env VLLM_CACHE_ROOT="$W/cache/vllm" \
  --pwd /opt/distillgapbench distillgapbench-gpu.sif \
  bash scripts/run_smoke.sh "$W/runs/smoke"
```

For a researcher run, replace the last line with
`bash scripts/run_cells.sh cells/v1_qwen4b.json "$W/runs/qwen4b" --only ...` and provide keys in
`~/.distillgap/provider_keys.env` (mode 600) or via `DISTILLGAP_ENV_FILE`. `run_cells.sh` serves one student per
GPU from the GPUs Slurm assigned (`CUDA_VISIBLE_DEVICES`); `cells/v1_leaderboard.json` needs four.

## Without containers

On a Linux machine with `unshare` (util-linux) and user namespaces enabled:

```sh
uv python install 3.12.13
uv venv --python 3.12.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests && .venv/bin/python scripts/check_sandbox.py
```

and install vLLM 0.24.0 in a separate environment for the students (`VLLM_PYTHON=/path/to/vllm-env/bin/python`). Do
not install bubblewrap: the sandbox pool needs the `unshare` mechanism, which the study used.
