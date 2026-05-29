#!/bin/bash
# =============================================================================
#  hpc/job_bem_radiation_v4.slurm  --  v4 numerical radiation on SACADO MeSU
#
#  Runs bem_radiation_v4_hpc.py (FMM assembly, per-band adaptive mesh, memory
#  guard).  This REPLACES the v3 BEM job that OOM-crashed at 415 GiB:
#    * FMM assembler  -> memory is O(N log N), a few GB, not O(N^2) ~150 GB.
#    * --mem here is therefore modest (32 GB is ample for <=120k nodes via FMM).
#    * per-frequency meshes are built/destroyed in scratch, capping node count.
#
#  Fixes vs the v3 job: no hardcoded path; runs in $SCRATCH; sane memory;
#                       reuses the existing 'narwhal' conda env by default
#                       (override with 'export TUSK_ENV=<name_or_path>').
# =============================================================================
#SBATCH --job-name=tusk_bem_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # FMM/bempp assembly is threaded (OpenMP/OpenCL)
#SBATCH --mem=32G                   # FMM is light; dense would have needed >400G
#SBATCH --time=08:00:00
#SBATCH --output=bem_v4_%j.out
#SBATCH --error=bem_v4_%j.err
# #SBATCH --partition=<your_partition>
# #SBATCH --account=<your_alloc>
#SBATCH --export=ALL                # propagate TUSK_ENV from the submit shell

set -euo pipefail

module purge
# We don't need a system MPI here -- bempp-cl is shared-memory only and the
# conda env brings its own dolfinx/MPICH for compatibility (unused here).
# Loading the cluster openmpi on top of conda-MPICH causes PMI conflicts;
# the BEM job is single-process, so no MPI module is needed.
source /softs/tools/conda/2025.06/etc/profile.d/conda.sh

# Reuse the existing v3 env 'narwhal' by default; override with TUSK_ENV.
ENV_TARGET="${TUSK_ENV:-narwhal}"
conda activate "$ENV_TARGET"
echo "Activated conda env: $ENV_TARGET"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BEMPP_FMM_THREADS=$SLURM_CPUS_PER_TASK

python -c "
try:
    import bempp.api as b
except ImportError:
    import bempp_cl.api as b
print('bempp', b.__version__)
try:
    import exafmm
    print('FMM (exafmm): available')
except ImportError:
    print('FMM (exafmm): NOT available -- BEM will use hmat fallback')
"

# --- run in scratch ----------------------------------------------------------
RUNDIR="${SCRATCH:-/tmp}/tusk_bem_${SLURM_JOB_ID}"
mkdir -p "$RUNDIR"
cp -t "$RUNDIR" config.txt config_refined.yaml \
   cylindrical_dispersion.py bem_radiation_v4_hpc.py
[ -d data ] && cp -r data "$RUNDIR/" || true
cd "$RUNDIR"

# numerical far-field radiation 20..200 kHz (step 5 kHz), FMM, adaptive meshing
python bem_radiation_v4_hpc.py

cp -t "$SLURM_SUBMIT_DIR" bem_radiation_v4.npz 2>/dev/null || true

echo "BEM v4 radiation done. Output: bem_radiation_v4.npz (compare data/radiation_v4.npz)"
# Acknowledgement (required):
# "The simulations were performed on the SACADO MeSU platform at Sorbonne Universite."
