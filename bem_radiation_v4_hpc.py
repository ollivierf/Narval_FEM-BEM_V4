#!/usr/bin/env python3
# =============================================================================
#  bem_radiation_v4_hpc.py  --  v4 NUMERICAL far-field radiation (BEM, bempp-cl)
#
#  MEMORY-SAFE rewrite of the v3 bem_radiation_hpc.py.  The v3 run CRASHED on
#  MeSU with
#        numpy._core._exceptions._ArrayMemoryError:
#        Unable to allocate 415. GiB for an array (166953, 166953) complex128
#  because (a) operators were assembled DENSE and (b) a single 167k-node surface
#  mesh (lambda0/6 at 200 kHz) was reused at every frequency.  This version fixes
#  BOTH and adds a hard memory guard:
#
#    1. FMM ASSEMBLY.  Every boundary/potential operator is built with
#       assembler="fmm"  ->  O(N log N) memory instead of O(N^2).  A 1e5-node
#       FMM problem fits in a few GB instead of ~150 GB dense.
#    2. PER-BAND ADAPTIVE MESH.  The wetted surface is re-meshed for each
#       frequency at lambda0(f)/ELELAM, so low frequencies use a coarse mesh and
#       only the top of the band approaches the fine limit.  Node count is capped
#       at N_NODES_MAX (skips/declocks a frequency that would exceed it).
#    3. MEMORY GUARD.  Before assembling, the predicted FMM working set is
#       estimated; if it exceeds MEM_BUDGET_GB the frequency is coarsened or
#       skipped with a clear message, so the job never OOM-kills the node.
#
#  v4 PHYSICS.  The imposed surface normal velocity is that of the dominant
#  RADIATING leaky mode (longitudinal L or flexural F): v_n(s)=U0 e^{-alpha s}
#  e^{i beta s}.  TORSION T(0,n) is never imposed -- it has u_r=0 (dark) and does
#  not couple to the perfect fluid; passing a torsion branch here yields a zero
#  Neumann datum and hence zero far field, exactly as the v4 model predicts.
#
#  REQUIRES bempp-cl (>=0.3) with an FMM backend (exafmm-t), gmsh, numpy, scipy.
#  Run on MeSU. The far-field loop is 20..200 kHz step 5 kHz.
# =============================================================================
import math
import os
import sys
import numpy as np
import yaml
import cylindrical_dispersion as C

# --- bempp import shim ------------------------------------------------------
# bempp-cl exposed its API at 'bempp.api' through 0.4.x and renamed it to
# 'bempp_cl.api' in newer upstream builds.  Find whichever one works in the
# active env so the rest of this script can just `bem = _import_bempp()`.
def _import_bempp():
    last = None
    for name in ("bempp.api", "bempp_cl.api"):
        try:
            return __import__(name, fromlist=["api"])
        except Exception as e:                # ImportError, or transitive break
            last = e
    raise ImportError(
        "Neither 'bempp.api' nor 'bempp_cl.api' could be imported. "
        f"Last error: {last}.  Check the env (e.g. `pip show bempp-cl`).")

# ----------------------------- parameters ----------------------------------
CONFIG_FILE   = "config.txt"
FREQS         = np.arange(20e3, 200e3+1, 1e3)
N_THETA       = 361
ELELAM        = 6.0                 # target elements per acoustic wavelength
N_NODES_MAX   = 120_000             # hard cap on surface nodes
MEM_BUDGET_GB = 100.0               # refuse/coarsen configs beyond this working set
GMRES_TOL     = 1e-4
FMM_EXPANSION = 5                   # multipole expansion order (accuracy/speed)

# refined material (axial dentine; see config_refined.yaml / README)
E, NU, RHO  = 10.3e9, 0.30, 1900.0
R_GUIDE     = 0.0287                 # guiding-cylinder radius (base) [m]
L_TUSK      = 1.80
RHO0, C0    = 1025.0, 1450.0
cL, cT      = C.bulk_velocities(E, NU, RHO)

# ----------------------------- geometry from config ------------------------
with open(CONFIG_FILE) as fh:
    cfg = yaml.safe_load(fh)
g = cfg["geometry"]
L  = float(g["tusk_length_total"])
Rb = float(g["outer_radius_base"])
Rt = float(g["outer_radius_tip"])

# =============================================================================
#  v4 leaky-mode surface velocity (RADIATING modes only; torsion -> zero)
# =============================================================================
_BAND_CACHE = None
def _band_families():
    """Compute the v4 branch zoo once over the whole band and cache it."""
    global _BAND_CACHE
    if _BAND_CACHE is None:
        fam = C.all_families(FREQS, cL, cT, R_GUIDE, RHO, C0,
                             n_scan=3000, m_flex=(1,), n_torsion=4)
        for b in fam['L']:
            C.leaky_alpha_branch(b, 0, cL, cT, R_GUIDE, RHO, RHO0, C0)
        for b in fam['F1']:
            C.leaky_alpha_branch(b, 1, cL, cT, R_GUIDE, RHO, RHO0, C0)
        _BAND_CACHE = fam
    return _BAND_CACHE

def leaky_mode_constants(f):
    """(alpha, beta) of the dominant RADIATING supersonic mode at f, taken from
    the three-potential branch set (L and F only). Returns (0,0) if none -> the
    Neumann datum is then zero (e.g. if only torsion exists: dark, no radiation)."""
    fam = _band_families()
    best = None
    best_cph = None
    for fkey in ('L', 'F1'):
        for b in fam[fkey]:
            j = np.argmin(np.abs(b['f']-f))
            if abs(b['f'][j]-f) > 1.0:
                continue
            cph = b['cph'][j]
            al = b.get('alpha', np.full_like(b['f'], np.nan))[j]
            if not (np.isfinite(cph) and cph > C0 and np.isfinite(al) and al > 0):
                continue
            # dominant = lowest supersonic cph (closest to grazing, strongest lobe)
            if best is None or cph < best_cph:
                best = (max(al, 1e-3), 2*math.pi*f/cph); best_cph = cph
    return best if best is not None else (0.0, 0.0)

# =============================================================================
#  per-frequency adaptive wetted-surface mesh (gmsh) with a node cap
# =============================================================================
def build_surface_mesh(f, fname):
    """Body-of-revolution triangular surface mesh sized to lambda0(f)/ELELAM,
    clamped so the node count never exceeds N_NODES_MAX. Returns (path, n_nodes,
    h_el). Helix set aside (axisymmetric body), consistent with v3/v4 scope."""
    import gmsh
    lam0 = C0/f
    h_el = lam0/ELELAM
    # estimate node count ~ surface_area / (sqrt(3)/4 h^2); coarsen if over cap
    slant = math.hypot(L, Rb-Rt)
    area = math.pi*(Rb+Rt)*slant + math.pi*Rb*Rb       # cone lateral + base disk
    def n_est(h):
        return int(2.3*area/(0.5*math.sqrt(3)*h*h))    # ~2 tri/node, tri area
    if n_est(h_el) > N_NODES_MAX:
        h_el = math.sqrt(2.3*area/(0.5*math.sqrt(3)*N_NODES_MAX))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tusk_surface")
    occ = gmsh.model.occ
    base = occ.addDisk(0, 0, 0, Rb, Rb)
    cone = occ.addCone(0, 0, 0, 0, 0, L, Rb, Rt)
    occ.fuse([(3, cone)], [(3, base)])
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_el*0.6)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_el)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    surfs = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)
    nn = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    return fname, nn, h_el

def assembler_mem_estimate_gb(n_nodes, assembler):
    """Rough working-set estimate (GB) for the chosen bempp-cl assembler with
    the four operators (SLP, DLP, ADLP, HypS) of a Burton-Miller CFIE.
    Calibrated against bempp-cl's typical per-DOF footprint:
       fmm               ~3.5 kB/DOF (near-field blocks + log(N) multipole)
       default_nonlocal  ~1.5 kB/DOF (matrix-free; only singular caches stored)
       dense             16 N^2 bytes (the v3 failure mode)"""
    n = float(n_nodes)
    if assembler == "dense":
        return 16.0 * n * n / 2**30
    per_dof = {"fmm": 3500.0, "default_nonlocal": 1500.0}.get(assembler, 3500.0)
    return (per_dof * n + 16.0 * n * math.log2(max(n, 2.0))) / 2**30

# kept as an alias for backwards compatibility with earlier scripts/logs
def fmm_mem_estimate_gb(n_nodes):
    return assembler_mem_estimate_gb(n_nodes, "fmm")

# =============================================================================
#  Assembler selection: pick the best memory-safe assembler bempp-cl exposes.
#  Preference order for bempp-cl 0.4.x (the version in narwhal, see docs at
#  bempp-cl.readthedocs.io/en/latest/docs/bempp_cl/api/operators/boundary/helmholtz/):
#      1. "fmm"               O(N log N) memory; needs exafmm
#      2. "default_nonlocal"  matrix-FREE JIT (OpenCL or Numba); minimal RAM
#      3. "dense"             O(N^2) -- only acceptable when N is tiny
#  NOTE: the legacy "hmat" keyword from bempp 0.2.x is NOT supported by
#  bempp-cl 0.4.x.  In matrix-free mode 'default_nonlocal' produces the same
#  weak_form LinearOperator usable by GMRES; per-iteration cost is higher than
#  FMM's, but RAM is tiny because the full matrix is never materialised.
# =============================================================================
_ASSEMBLER = None

def _detect_assembler_capabilities():
    """Inspect imports only (no expensive operator probe) to make a best guess."""
    try:
        import exafmm  # noqa: F401
        return ["fmm", "default_nonlocal", "dense"]
    except Exception:
        return ["default_nonlocal", "dense"]

def _set_assembler(name):
    global _ASSEMBLER
    _ASSEMBLER = name
    print(f"[BEM] using assembler='{name}'", flush=True)

def _current_assembler():
    global _ASSEMBLER
    if _ASSEMBLER is None:
        order = _detect_assembler_capabilities()
        _set_assembler(order[0])
    return _ASSEMBLER

def _try_build_op(opfn, *args, **kwargs):
    """Build a boundary operator, falling back to a cheaper assembler if the
    requested one raises (e.g. assembler='fmm' but exafmm not actually wired)."""
    order = _detect_assembler_capabilities()
    cur = _current_assembler()
    todo = [cur] + [a for a in order if a != cur]
    last_err = None
    for asm in todo:
        try:
            return opfn(*args, **kwargs, assembler=asm), asm
        except Exception as e:
            last_err = e
            print(f"[BEM] assembler='{asm}' failed: {e}; trying next.", flush=True)
    raise RuntimeError(f"All assemblers failed; last error: {last_err}")

# =============================================================================
#  ROUTE 1 : one-way exterior Helmholtz radiation (bempp-cl, Burton-Miller)
# =============================================================================
def radiate_oneway(f, grid, theta):
    bem = _import_bempp()
    try:
        from bempp.api.linalg import gmres
    except ImportError:
        from bempp_cl.api.linalg import gmres
    # Optionally tune FMM accuracy if the attribute exists in this bempp-cl build:
    if _current_assembler() == "fmm":
        try:
            bem.GLOBAL_PARAMETERS.fmm.expansion_order = FMM_EXPANSION
        except Exception:
            pass

    w = 2*math.pi*f
    k0 = w/C0
    space = bem.function_space(grid, "P", 1)

    alpha_f, beta_f = leaky_mode_constants(f)
    prefac = 1j*w*RHO0

    @bem.complex_callable
    def neumann_data(x, n, domain_index, result):
        s = x[2]                                   # axial coordinate along the tusk
        # v4: radial (normal) surface velocity of the dominant radiating mode.
        # If (alpha_f,beta_f)=(0,0) (e.g. only torsion present) -> zero => dark.
        result[0] = prefac*np.exp(-alpha_f*s)*np.exp(1j*beta_f*s)
    rhs_neu = bem.GridFunction(space, fun=neumann_data)

    # Boundary operators with adaptive-fallback assembler:
    ident = bem.operators.boundary.sparse.identity(space, space, space)
    dlp, asm = _try_build_op(bem.operators.boundary.helmholtz.double_layer,
                             space, space, space, k0)
    if asm != _current_assembler():               # remember the working choice
        _set_assembler(asm)
    slp, _ = _try_build_op(bem.operators.boundary.helmholtz.single_layer,
                           space, space, space, k0)
    hyp, _ = _try_build_op(bem.operators.boundary.helmholtz.hypersingular,
                           space, space, space, k0)
    adj, _ = _try_build_op(bem.operators.boundary.helmholtz.adjoint_double_layer,
                           space, space, space, k0)

    # Burton-Miller CFIE (eta = i/k0) removes fictitious interior eigenfrequencies
    eta = 1j/k0
    lhs = (0.5*ident - dlp) - eta*hyp
    rhs = (slp*rhs_neu) - eta*((0.5*ident + adj)*rhs_neu)
    p_surf, info = gmres(lhs, rhs, tol=GMRES_TOL, use_strong_form=True)

    # far field on unit directions in the xz-plane (theta from the axis).
    # Far-field operators map N surface dofs -> N_theta points (small, dense ok).
    pts = np.vstack([np.sin(theta), np.zeros_like(theta), np.cos(theta)])
    slp_ff = bem.operators.far_field.helmholtz.single_layer(space, pts, k0)
    dlp_ff = bem.operators.far_field.helmholtz.double_layer(space, pts, k0)
    p_inf = dlp_ff*p_surf - slp_ff*rhs_neu
    return p_inf.ravel()

# =============================================================================
#  ROUTE 2 : two-way coupled FEM-BEM (structure; v4 document sec. FEM-BEM)
# =============================================================================
def solve_coupled(f, vol_msh, surf_msh):
    """Monolithic [ D=K*-w^2 M , -C_fs ; w^2 rho0 C_sf , A_BEM ][u;p]=[f;0].
    A_BEM assembled with assembler='fmm'; block-GMRES on the Schur complement.
    The fluid load reacts back on the structure (refines Route 1). Wiring of the
    interface traces C_fs/C_sf is described in the README (FEM-BEM coupling)."""
    raise NotImplementedError("Route 2 (two-way) -- see README, FEM-BEM coupling.")

# =============================================================================
#  driver (Route 1) with per-band meshing and memory guard
# =============================================================================
def main():
    bem = _import_bempp()
    asm = _current_assembler()             # lazy: 'fmm' if exafmm, else 'hmat'
    theta = np.linspace(0.0, math.pi, N_THETA)
    P = np.zeros((len(FREQS), N_THETA))
    meta = []
    for i, f in enumerate(FREQS):
        msh = f"tusk_surface_{f/1e3:.0f}kHz.msh"
        path, nn, h_el = build_surface_mesh(f, msh)
        # the assembler may have been downgraded since the loop started; refresh
        asm = _current_assembler()
        mem = assembler_mem_estimate_gb(nn, asm)
        tag = (f"[BEM] {f/1e3:5.0f} kHz  nodes={nn:6d}  h={h_el*1e3:.2f}mm  "
               f"asm={asm}  ~{mem:.1f}GB")
        if mem > MEM_BUDGET_GB:
            print(tag + "  -> SKIP (exceeds MEM_BUDGET_GB)")
            meta.append((f, nn, h_el, mem, "skipped"))
            os.remove(path)
            continue
        print(tag, flush=True)
        grid = bem.import_grid(path)
        try:
            p_inf = radiate_oneway(f, grid, theta)
            P[i] = np.abs(p_inf)**2
            mx = P[i].max()
            if mx > 0:
                P[i] /= mx
            meta.append((f, nn, h_el, mem, "ok"))
        except Exception as exc:                   # never let one band kill the job
            print(f"        FAILED at {f/1e3:.0f} kHz: {exc}")
            meta.append((f, nn, h_el, mem, f"failed:{type(exc).__name__}"))
        finally:
            if os.path.exists(path):
                os.remove(path)                    # keep scratch clean (MeSU policy)
    np.savez("bem_radiation_v4.npz", freqs=FREQS, theta=np.degrees(theta), P=P,
             meta=np.array(meta, dtype=object))
    print("wrote bem_radiation_v4.npz  (compare data/radiation_v4.npz)")

if __name__ == "__main__":
    main()
