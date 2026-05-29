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
# =============================================================================
#SBATCH --job-name=tusk_fem_v4
#SBATCH --nodes=1
#SBATCH --ntasks=24                 # MPI ranks (axisymmetric meridian: 1 node ok)
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=fem_v4_%j.out
#SBATCH --error=fem_v4_%j.err
# #SBATCH --partition=<your_partition>   # set per `sinfo`
# #SBATCH --account=<your_alloc>

set -euo pipefail

# --- environment ---
module purge
module load mpi/openmpi/4.1.8/gcc || module load openmpi || true
source /softs/tools/conda/2025.06/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/tusk"

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
for MODE in long torsion flexion; do
    echo "=== FEM v4 mode: $MODE ==="
    srun -n "$SLURM_NTASKS" python fem_dispersion_v4_hpc.py --mode "$MODE"
done

# --- copy results back to the submission directory ---------------------------
cp -t "$SLURM_SUBMIT_DIR" fem_dispersion_v4_*.npz fem_dispersion_v4_*.png \
                          fem_sensors_v4_*.npz 2>/dev/null || true

echo "FEM v4 dispersion done. Outputs: fem_dispersion_v4_{long,torsion,flexion}.{npz,png}"
# Acknowledgement (required):
# "This work was granted access to the HPC resources of the SACADO MeSU
#  platform at Sorbonne Universite."
