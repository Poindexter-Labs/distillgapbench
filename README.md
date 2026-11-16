# DistillGapBench v1 harness

DistillGapBench measures how well a frontier model, the *researcher*, finds the edge of a smaller model's ability.
For a frozen base model, the *student*, the researcher writes minimal pairs of tasks: a *control* the student solves
and a small edit of it, the *challenge*, that the student fails. A pair scores as a *frontier gap* only when both
tasks pass checks the researcher did not write and fresh student samples confirm the split: the control passes on at
least 2 of 4 samples and the challenge on none.

This package is the harness that produced the v1 leaderboard. It contains the grid runner, the three environments
(coding, mathematics, code output prediction), the namespace sandbox that runs researcher- and student-written code,
the twelve researcher routes, and the cell lists for all 156 leaderboard cells.

Leaderboard and paper: https://app.poindexterlabs.net/benches/distillgapbench. Source archives: the
[releases page](https://github.com/Poindexter-Labs/distillgapbench/releases).

## What you need

| To | You need |
|---|---|
| Check the package | Docker or Podman. No GPU, network or API keys. |
| Run the GPU smoke test | One NVIDIA GPU and internet access for the student weights (about 4 GB). No API keys. |
| Score a researcher | One GPU per student served at a time, and API keys for the researcher and the validators. |

Built and checked on NVIDIA GH200 (linux/arm64) with Podman 4.9 and Apptainer 1.5 on Isambard-AI. The same Dockerfiles build
for linux/amd64, but that image has not yet run on GPUs. Model weights, API keys and run outputs are not included.

## Quick start

```sh
git clone https://github.com/Poindexter-Labs/distillgapbench.git && cd distillgapbench

# 1. CPU check: the test suite and the sandbox check (no GPU, network or keys)
docker build -t distillgapbench:cpu .
docker run --rm --network none \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined \
  distillgapbench:cpu

# 2. GPU smoke test: the operator template on Qwen3-1.7B-Base, output prediction, 4 pairs (no keys)
docker build -f Dockerfile.gpu -t distillgapbench:gpu .
mkdir -p runs
docker run --rm --gpus all \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined \
  -v "$HOME/.cache/huggingface:/home/dg/.cache/huggingface" -v "$PWD/runs:/home/dg/runs" \
  distillgapbench:gpu bash scripts/run_smoke.sh

# 3. Score one researcher on one student: Claude Fable 5.2 on Qwen3-4B-Base, all three environments
cp .env.example .env    # fill in the keys you need (see below)
docker run --rm --gpus all --env-file .env \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined \
  -v "$HOME/.cache/huggingface:/home/dg/.cache/huggingface" -v "$PWD/runs:/home/dg/runs" \
  distillgapbench:gpu bash scripts/run_cells.sh cells/v1_qwen4b.json /home/dg/runs/qwen4b \
  --only fable52__qwen4b__coding fable52__qwen4b__math fable52__qwen4b__output
```

The security options let the sandbox create Linux user namespaces and mount its own `/proc`, which Docker's default
profile blocks. Rootless Podman needs only `--security-opt unmask=ALL`; Apptainer needs nothing extra.
[docs/CONTAINERS.md](docs/CONTAINERS.md) covers Podman, Apptainer and Slurm.

Mount results under `/home/dg` and pass keys with `--env-file`. The sandbox hides `/home` and `/opt` (the code) from
every program it runs and gives them a clean environment, so researcher- and student-written code cannot read keys,
hidden tests or other results.

## Keys

Copy `.env.example` to `.env` and fill in only what you call. Each researcher's provider is listed in
`researchers_v6.json`.

- **Researchers:** Azure OpenAI (GPT-6 Astra, GPT-6.1 Sol, GPT-5.6), Amazon Bedrock (Claude models, Kimi K3.1),
  Gemini API (Gemini 4 models) and Meta (Muse Wonder 1.0).
- **Validators:** in coding and mathematics another model family re-solves every task blind. GPT-6 Astra (Azure)
  validates the Anthropic researchers; Claude Opus 5.5 (Bedrock) validates the others and the template. So coding and
  mathematics need both an Azure and a Bedrock key. Output prediction has no validator: the code is executed.
- **Judge:** optional; see [docs/JUDGE.md](docs/JUDGE.md).
- **Students:** `google/gemma-3-4b-pt` is gated on Hugging Face. Accept its licence and set `HF_TOKEN`; the
  other three students download without a token.

## Running the grid

`scripts/run_cells.sh CELLS OUT [run_grid.py options]` serves every student the cell list needs (one vLLM server per
student, one GPU each, bound to localhost, as in the study) and then runs `scripts/run_grid.py`. Per cell it runs the
discovery arms, the independent final evaluation on fresh student samples, the judge, and writes
`OUT/<cell>/report.json`. Runs resume: a frozen arm or a finished stage is never redone.

| Cell file | Contents |
|---|---|
| `cells/v1_leaderboard.json` | All 156 v1 cells: 12 researchers and the template, 4 students, 3 environments, 3 repetitions, adaptive and batch arms. Needs 4 GPUs. |
| `cells/v1_<student>.json` | The 39 cells of one student. Needs 1 GPU. |
| `cells/smoke_template.json` | The no-key smoke test. |
| `cells/smoke_researcher.json` | Claude Sonnet 5.5, 2 pairs on output prediction (Bedrock key; a few dollars). |

Select cells with `--only CELL_ID ...`. Build the leaderboard from finished cells with
`python scripts/collect_grid.py --runs runs/qwen4b runs/gemma4b ... --out leaderboard.json --md leaderboard.md`.

Budgets per arm are those of v1: 16 pairs, at most 20 researcher calls, an 8-hour wall clock. At list prices a
researcher's full row (72 arms) cost $106 to $1,043 in researcher calls plus $22 to $183 in validator calls in v1.
Students sample at temperature 0.6, top-p 0.95, with per-item seeds. GPU sampling is not bitwise reproducible: in
our qualification about 1 to 2% of samples differed between runs, with no pass/fail flips.

## Layout

```
distillgap/        protocol: arms, pairs, prompts, verifier, final evaluation, judge, reports
envs/              mathematics and output prediction (coding lives in distillgap/), plus appendix environments
sandbox/           Linux namespace sandbox and the persistent sandbox pool
scripts/           run_cells.sh, run_grid.py, run_smoke.sh, collect_grid.py, selfcheck.sh, check_sandbox.py
routes.py          the twelve researcher routes (no provider fallback; refusals are outcomes)
researchers_v6.json  routes, efforts, prices, validators
cells/             cell lists
tests/             test suite (fixture student, fake routes; no network)
```

## Changes from the study code

The protocol code is the code that ran v1. For release we changed only how it is configured:

- Credentials come from environment variables or a `.env` file (`routes.load_credentials`).
- Students load from Hugging Face at the pinned revisions instead of cluster paths.
- The sandbox ships in `sandbox/`, and the container images pin the study's interpreter (CPython 3.12.13), vLLM
  0.24.0 and client libraries.
- The light-path judge is optional, because its prompts are internal to Poindexter (`--judge`, docs/JUDGE.md).
- The chess environment, which needs python-chess and is not part of v1, is left out.

## License and citation

GPL-3.0 (see LICENSE).

```bibtex
@misc{pricope2026distillgapbench,
  title  = {DistillGapBench: Measuring How Well Frontier Models Locate the Edge of a Smaller Model's Ability},
  author = {Pricope, Tidor-Vlad},
  year   = {2026},
  url    = {https://app.poindexterlabs.net/benches/distillgapbench}
}
```
