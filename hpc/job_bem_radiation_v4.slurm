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
#  Fixes vs the v3 job: no hardcoded path; runs in $SCRATCH; sane memory.
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

set -euo pipefail

module purge
module load mpi/openmpi/4.1.8/gcc || module load openmpi || true
source /softs/tools/conda/2025.06/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/tusk"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BEMPP_FMM_THREADS=$SLURM_CPUS_PER_TASK

python -c "import bempp.api as b; print('bempp', b.__version__, '| FMM:', getattr(b,'fmm_available',lambda:'?')())"

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
