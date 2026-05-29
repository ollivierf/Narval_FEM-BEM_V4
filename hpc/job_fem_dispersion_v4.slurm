#!/bin/bash
# =============================================================================
#  hpc/job_fem_dispersion_v4.slurm  --  v4 numerical dispersion on SACADO MeSU
#
#  Runs the Fourier-mode FEM for the THREE v4 families on the same meridional
#  mesh:  long -> L(0,n),  torsion -> T(0,n),  flexion -> F(1,n).
#  Counterpart of run_analytic_v4.py for cross-validation of the f-k diagrams.
#
#  Fixes vs the v3 job:
#    * NO hardcoded path: work from $SLURM_SUBMIT_DIR, compute in $SCRATCH
#      (MeSU charter: jobs must run in scratch, not HOME).
#    * runs the three modes sequentially (each writes its own npz/png).
#    * env: reuses the existing 'narwhal' conda env by default; override with
#      'export TUSK_ENV=<name_or_path>' before sbatch.
# =============================================================================
#SBATCH --job-name=tusk_fem_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # single rank: see "MPI launcher note" below
#SBATCH --cpus-per-task=8           # used by PETSc/MUMPS threads + numba JIT
#SBATCH --time=10:00:00
#SBATCH --output=fem_v4_%j.out
#SBATCH --error=fem_v4_%j.err
# #SBATCH --partition=<your_partition>   # set per `sinfo`
# #SBATCH --account=<your_alloc>
#SBATCH --export=ALL                # propagate TUSK_ENV from the submit shell

# -----------------------------------------------------------------------------
# MPI launcher note (READ ME if your first FEM run aborted with
#   "Runtime environment uses unsupported PMI version PMIx"):
#
# SACADO MeSU's `srun` launches tasks via PMIx, but the MPICH inside the conda
# env 'narwhal' (shipped with dolfinx-from-conda-forge) was built against the
# legacy PMI-2 interface, so `srun python ...` aborts at MPI_Init.  Three fixes:
#
#   * Default below: run a SINGLE rank with plain `python ...`  (works always;
#     MPI inside dolfinx degrades cleanly to a single-process communicator).
#     For the axisymmetric meridian mesh used here this is plenty fast.
#
#   * For real MPI parallelism, replace the `python ...` line below with:
#         srun --mpi=pmi2 -n "$SLURM_NTASKS" python fem_dispersion_v4_hpc.py ...
#     and bump `--ntasks` and drop `--cpus-per-task`.  Check `srun --mpi=list`
#     to confirm which PMI variants this MeSU SLURM accepts.
#
#   * Alternatively, use the conda env's own mpiexec which knows MPICH's PMI:
#         mpiexec -n 4 python fem_dispersion_v4_hpc.py ...
#     (this bypasses SLURM's PMIx, single-node only).
# -----------------------------------------------------------------------------

set -euo pipefail

# --- environment ---
module purge
# NOTE: we deliberately do NOT load the cluster's openmpi module here.
# The dolfinx in 'narwhal' was built against the MPICH that lives INSIDE the
# conda env; loading a system MPI on top of it just creates ABI/PMI conflicts
# (the symptom is "Runtime environment uses unsupported PMI version PMIx"
# when srun hands its PMIx to MPICH).  Pure conda env is enough for single-rank.
source /softs/tools/conda/2025.06/etc/profile.d/conda.sh

# Reuse the existing v3 env 'narwhal' by default; allow override via TUSK_ENV.
ENV_TARGET="${TUSK_ENV:-narwhal}"
conda activate "$ENV_TARGET"
echo "Activated conda env: $ENV_TARGET"

# threading hints for the single-rank build (PETSc/MUMPS + Numba JIT)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

which python
python -c "import dolfinx, sys; print('dolfinx', dolfinx.__version__)"

# --- run in scratch, copy inputs in, copy outputs back (MeSU policy) ---------
RUNDIR="${SCRATCH:-/tmp}/tusk_fem_${SLURM_JOB_ID}"
mkdir -p "$RUNDIR"
cp -t "$RUNDIR" \
   config.txt config_refined.yaml \
   cylindrical_dispersion.py postprocess_dispersion.py \
   generate_axisymmetric_mesh.py fem_dispersion_v4_hpc.py
# bring the analytic overlay if it was produced on the login node
[ -d data ] && cp -r data "$RUNDIR/" || true
cd "$RUNDIR"

# 1) fine meridional mesh resolved for 200 kHz
python generate_axisymmetric_mesh.py

# 2) the three Fourier-mode families (each: 2-D FFT -> numerical f-k)
#    Single-rank by default; see MPI launcher note above for the parallel form.
for MODE in long torsion flexion; do
    echo "=== FEM v4 mode: $MODE ==="
    python fem_dispersion_v4_hpc.py --mode "$MODE"
done

# --- copy results back to the submission directory ---------------------------
cp -t "$SLURM_SUBMIT_DIR" fem_dispersion_v4_*.npz fem_dispersion_v4_*.png \
                          fem_sensors_v4_*.npz 2>/dev/null || true

echo "FEM v4 dispersion done. Outputs: fem_dispersion_v4_{long,torsion,flexion}.{npz,png}"
# Acknowledgement (required):
# "This work was granted access to the HPC resources of the SACADO MeSU
#  platform at Sorbonne Universite."
