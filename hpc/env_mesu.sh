#!/bin/bash
# =============================================================================
#  hpc/env_mesu.sh  --  prepare the conda environment for the v4 pipeline on
#                       SACADO MeSU / MCMeSU (Sorbonne Universite)
#
#  PHILOSOPHY: probe, report, do as little as possible.
#  By default this script reuses the existing 'narwhal' env (built for v3)
#  and only DIAGNOSES what's available; it does NOT call `conda install` on
#  the login node (the login node's small RAM lets conda's solver OOM-crash
#  -- this was observed on MeSU 2025).  When something is missing it prints
#  exactly the manual `pip install` or `conda install` command to run.
#
#  The BEM script (bem_radiation_v4_hpc.py) auto-detects the best assembler
#  available at runtime:
#       fmm    (if 'exafmm' is importable)
#       hmat   (built into bempp-cl; works with no extra package)
#       dense  (memory-guarded ; only used at tiny mesh size)
#  So the v4 BEM run is tractable as long as bempp-cl ITSELF imports.
#
#  Usage:
#      bash hpc/env_mesu.sh                  # reuse env 'narwhal' (default)
#      TUSK_ENV=myenv bash hpc/env_mesu.sh   # reuse another named env
#      TUSK_ENV=./tusk bash hpc/env_mesu.sh  # use a project-local prefix env
#      TUSK_ENV=narwhal AUTO_INSTALL=1 bash hpc/env_mesu.sh
#                                            # opt-in: try pip install of
#                                            # missing pieces (NOT conda)
#
#  Acknowledgement required in any publication:
#    "This work was granted access to the HPC resources of the SACADO MeSU
#     platform at Sorbonne Universite."
# =============================================================================
set -uo pipefail                        # NOT -e: we never want to abort early

# --- locate conda ------------------------------------------------------------
module purge || true
# Do NOT load a system MPI module: the conda env's dolfinx ships its own MPICH
# and loading another MPI on top causes PMI conflicts at MPI_Init.
source /softs/tools/conda/2025.06/etc/profile.d/conda.sh

# --- pick which env to use ---------------------------------------------------
ENV_TARGET="${TUSK_ENV:-narwhal}"
AUTO_INSTALL="${AUTO_INSTALL:-0}"

env_kind() {
    local target="$1"
    if [[ "$target" == /* || "$target" == .* ]]; then
        if [ -d "$target" ]; then echo "prefix-exists"
        else echo "prefix-missing"; fi
        return
    fi
    if conda env list | awk '{print $1}' | grep -Fxq "$target"; then
        echo "name-exists"
    else
        echo "name-missing"
    fi
}

KIND="$(env_kind "$ENV_TARGET")"
echo "Target env: '${ENV_TARGET}'  (status: ${KIND})"

case "$KIND" in
    name-exists|prefix-exists)
        echo "Reusing existing env."
        conda activate "$ENV_TARGET" || { echo "activate failed"; exit 1; }
        ;;
    name-missing|prefix-missing)
        echo "Env '${ENV_TARGET}' not found."
        echo "Cannot create it from this script -- conda's solver tends to"
        echo "run out of memory on the MeSU login node.  Please ask the admin"
        echo "to provide one, or create on a compute node via 'salloc'."
        exit 1
        ;;
esac

# --- probe what's already in the env (DO NOT install on login node) ----------
echo ""
echo "Probing packages in '${ENV_TARGET}' ..."

# Try multiple import names for bempp-cl: the import path changed between
# 0.4.x (bempp.api) and the current upstream main (bempp_cl.api).
BEMPP_NAME=""
for cand in bempp.api bempp_cl.api; do
    if python -c "import ${cand}" >/dev/null 2>&1; then
        BEMPP_NAME="${cand}"
        break
    fi
done

# Even if the import fails (e.g. missing transitive dep), the package may be
# installed.  `pip show bempp-cl` answers that without running an import.
BEMPP_PIP=""
if python -m pip show bempp-cl >/dev/null 2>&1; then
    BEMPP_PIP="$(python -m pip show bempp-cl 2>/dev/null | awk '/^Version:/{print $2}')"
fi

declare -A STATE
for mod_disp in "dolfinx:dolfinx" "gmsh:gmsh" "scipy:scipy" "numpy:numpy" \
                "yaml:yaml" "exafmm:exafmm"; do
    disp="${mod_disp%%:*}"; mod="${mod_disp##*:}"
    if python -c "import ${mod}" >/dev/null 2>&1; then
        STATE["$disp"]="ok"
        printf "  %-12s OK\n" "$disp"
    else
        STATE["$disp"]="missing"
        printf "  %-12s MISSING\n" "$disp"
    fi
done

# Probe the FEM mesh-IO entry point: dolfinx renamed dolfinx.io.gmshio to
# dolfinx.io.gmsh in v0.10, so the import path the FEM script needs is one of
# these two.  Detecting it on the login node prevents the FEM SLURM job from
# starting up only to die at the import line.
GMSHIO_PATH=""
for cand in "dolfinx.io.gmshio" "dolfinx.io.gmsh"; do
    if python -c "import ${cand}" >/dev/null 2>&1; then
        GMSHIO_PATH="${cand}"; break
    fi
done
if [ -n "$GMSHIO_PATH" ]; then
    printf "  %-12s OK  (path: %s)\n" "gmsh I/O" "$GMSHIO_PATH"
    STATE["gmsh-io"]="ok"
else
    printf "  %-12s NOT IMPORTABLE  (FEM job will fail at mesh-load)\n" "gmsh I/O"
    STATE["gmsh-io"]="missing"
fi

if [ -n "$BEMPP_NAME" ]; then
    printf "  %-12s OK  (importable as '%s')\n" "bempp-cl" "$BEMPP_NAME"
elif [ -n "$BEMPP_PIP" ]; then
    printf "  %-12s INSTALLED but NOT IMPORTABLE  (pip version %s)\n" \
        "bempp-cl" "$BEMPP_PIP"
    echo "       -> something in its dependency chain is broken in this env."
    echo "       Below is the actual error from a clean import attempt:"
    echo "       ---"
    python -c "import bempp.api" 2>&1 | tail -10 | sed 's/^/         /'
    echo "       ---"
    echo "       Most common causes: numba/pyopencl version mismatch, or a"
    echo "       missing OpenCL runtime.  Quick fixes to try (in this env):"
    echo "           pip install --upgrade 'numba>=0.57' 'pyopencl>=2023.1'"
    echo "           pip install --force-reinstall --no-deps bempp-cl"
else
    printf "  %-12s NOT INSTALLED\n" "bempp-cl"
fi

# --- gentle remediation suggestions ------------------------------------------
need_pip=()
need_attention=()

if [ -z "$BEMPP_NAME" ]; then
    if [ -n "$BEMPP_PIP" ]; then
        need_attention+=("bempp-cl is installed but won't import -- check the error above")
    else
        need_pip+=("bempp-cl")
    fi
fi
[ "${STATE[exafmm]}" = "missing" ] && \
    need_attention+=("exafmm (FMM backend) not installed -- BEM will fall back to hmat (still memory-safe)")
[ "${STATE[dolfinx]}" = "missing" ] && \
    need_attention+=("dolfinx not installed -- the FEM job won't run")
[ "${STATE[gmsh]}"    = "missing" ] && need_pip+=("gmsh")

echo ""
if [ ${#need_pip[@]} -gt 0 ]; then
    echo "Pieces missing that pip CAN install (run manually if you want them):"
    for p in "${need_pip[@]}"; do echo "    pip install ${p}"; done
    if [ "$AUTO_INSTALL" = "1" ]; then
        echo "AUTO_INSTALL=1 -> attempting pip install (pip is light; no conda solver)"
        python -m pip install "${need_pip[@]}" \
            || echo "  pip install failed; see above"
    else
        echo "(set AUTO_INSTALL=1 to attempt these automatically; conda is NOT used)"
    fi
fi

if [ ${#need_attention[@]} -gt 0 ]; then
    echo ""
    echo "Notes:"
    for n in "${need_attention[@]}"; do echo "  - ${n}"; done
fi

# --- summary -----------------------------------------------------------------
echo ""
echo "Summary:"
echo "  conda env active : ${ENV_TARGET}"
echo "  dolfinx (FEM)    : ${STATE[dolfinx]}"
if [ -n "$BEMPP_NAME" ]; then
    echo "  bempp-cl (BEM)   : ok  (import: ${BEMPP_NAME})"
else
    echo "  bempp-cl (BEM)   : NOT USABLE -- see notes above"
fi
echo "  exafmm (FMM)     : ${STATE[exafmm]}  (hmat fallback is OK)"

echo ""
echo "Next steps:"
echo "  sbatch hpc/job_fem_dispersion_v4.slurm    (needs dolfinx)"
echo "  sbatch hpc/job_bem_radiation_v4.slurm     (needs bempp-cl; FMM optional)"
echo "(the SLURM scripts read \$TUSK_ENV; set it before sbatch to override.)"
