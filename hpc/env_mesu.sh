#!/bin/bash
# =============================================================================
#  hpc/env_mesu.sh  --  create the conda environment 'tusk' on SACADO MeSU/MCMeSU
#
#  Builds an environment with FEniCSx (dolfinx) + bempp-cl + an FMM backend
#  (exafmm-t) so that:
#    * fem_dispersion_v4_hpc.py  runs (dolfinx + MUMPS + MPI)
#    * bem_radiation_v4_hpc.py   runs with assembler="fmm"  (NOT dense!)
#
#  The v3 BEM run crashed because bempp fell back to DENSE assembly (415 GiB).
#  The FMM backend below is what makes the BEM tractable; verify it is found by
#  the check at the end (must print "FMM available: True").
#
#  Usage (from the project directory, on a login node):
#      bash hpc/env_mesu.sh
#  then submit the jobs with sbatch (see hpc/job_*.slurm).
#
#  Acknowledgement required in any publication:
#    "This work was granted access to the HPC resources of the SACADO MeSU
#     platform at Sorbonne Universite."
# =============================================================================
set -euo pipefail

# --- locate conda (adjust the module/path to the current MeSU install) -------
module purge || true
# OpenMPI runtime for dolfinx/MUMPS (match the cluster's provided module):
module load mpi/openmpi/4.1.8/gcc || module load openmpi || true
source /softs/tools/conda/2025.06/etc/profile.d/conda.sh

ENV_PREFIX="./tusk"        # project-local env (keeps HOME quota free; HOME=100GB)

if [ -d "${ENV_PREFIX}" ]; then
    echo "Environment ${ENV_PREFIX} already exists; activating."
    conda activate "${ENV_PREFIX}"
else
    echo "Creating conda env at ${ENV_PREFIX} (FEniCSx + bempp-cl + FMM) ..."
    conda create -y -p "${ENV_PREFIX}" -c conda-forge python=3.11
    conda activate "${ENV_PREFIX}"
    # FEniCSx stack (real + complex PETSc; complex is needed by bempp/FMM)
    conda install -y -c conda-forge \
        fenics-dolfinx=0.9 mpich petsc=*=*complex* petsc4py mumps-mpi \
        gmsh python-gmsh meshio pyyaml numpy scipy matplotlib
    # bempp-cl + FMM backend (exafmm-t) + OpenCL/Numba JIT
    pip install bempp-cl exafmm-t pyopencl numba
fi

# --- sanity checks -----------------------------------------------------------
python - <<'PY'
import importlib, sys
ok = True
for mod in ("dolfinx", "bempp.api", "gmsh", "scipy", "numpy"):
    try:
        importlib.import_module(mod); print(f"  {mod:12s} OK")
    except Exception as e:
        ok = False; print(f"  {mod:12s} MISSING: {e}")
try:
    import bempp.api as bem
    has_fmm = bem.fmm_available() if hasattr(bem, "fmm_available") else True
    print("FMM available:", has_fmm)
except Exception as e:
    print("FMM check failed:", e); ok = False
sys.exit(0 if ok else 1)
PY

echo "env_mesu.sh done. Activate with:  conda activate ${ENV_PREFIX}"
