# iHEALTH cluster — operating rules (apply to ALL work under /home/lmurrayh)

These rules are mandatory for every session in this tree. They come from the
iHEALTH cluster guidelines (i-health.cl/ih-condor docs) and from hard experience.

## Execution: SLURM ONLY — never run compute on the login node
- The login node (`ih-condor`) must NOT run heavy work. Running Python / JAX /
  plotting / any real compute there spikes RAM and **gets the user kicked off
  the server**. Even a one-off `python -c "import jax"` counts as compute.
- Route EVERY non-trivial command through SLURM: `srun` for short interactive
  checks, `sbatch` for jobs. Only trivial shell ops may run on the login node
  (ls, grep, cat, file edits, git, and `sbatch`/`squeue` submission itself).
- **Lightweight interactive (verified working).** The `interactive` partition
  REQUIRES a matching qos (omitting it → "Invalid qos specification"):
  ```
  srun --partition=interactive --qos=interactive --mem=2G --cpus-per-task=1 --time=0:10:00 bash -c '<cmd>'
  ```
- **GPU work:** `--partition=batch --qos=batch --gpus=1` via `srun`, or `sbatch`
  a wrapper script (e.g. `scripts/run_*_condor.sh`).
- Activate the env *inside* the job:
  ```
  module load conda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>
  ```
- For CPU-only checks, `export JAX_PLATFORMS=cpu` silences the harmless
  login-node GPU probe ("CUDA_ERROR_NO_DEVICE").
- When unsure whether something is "heavy enough" to need SLURM: route it
  through SLURM anyway.

## Storage conventions
- `/home/lmurrayh` — source code & important files (this tree). NOT for large outputs.
- `/mnt/workspace/lmurrayh` — conda envs, model weights/checkpoints, logs, and
  all experiment outputs. **Write results here, not in the repo or home.**
- `/mnt/researchers/<researcher>` — datasets shared across the research sub-group.
- `/mnt/nas1/<researcher>` — cold storage for datasets.
- Check quotas: `quota-pretty` and `quota-pretty -g $RESEARCHER`.

## Software / environments
- conda and micromamba are pre-installed — `module load conda` (or `micromamba`).
- **Do NOT mix `conda` and `pip` installs in one env** — create Python via conda,
  install everything else via pip.
- No `sudo` / admin installs.
- Conda envs live under `/mnt/workspace/$USER/.conda/envs` (the default), which
  satisfies the "envs in workspace" rule automatically.

## Projects in this tree
- **st-DIP** — `~/st-dip` (standalone st-DIP cardiac-CINE reconstruction). Conda
  env: `stdip`. Dataset (read-only, shared): `/mnt/researchers/claudia-prieto/datasets/pulseqCINE/DATA_0.55T`
  (the `.mat` for `--regime-spokes` lives in each subject's `datasets/` subfolder).
  Outputs: `/mnt/workspace/lmurrayh/stdip-results`. See `~/st-dip/README.md`.