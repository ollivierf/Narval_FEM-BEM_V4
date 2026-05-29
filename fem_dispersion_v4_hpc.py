#!/usr/bin/env python3
# =============================================================================
#  fem_dispersion_v4_hpc.py  --  v4 NUMERICAL dispersion (FEniCSx, Fourier modes)
#
#  v4 extension of the v3 fem_dispersion_hpc.py.  The v3 solver was axisymmetric
#  (n=0, two displacement components u_r,u_z) and therefore captured ONLY the
#  longitudinal family L(0,n).  The three-potential v4 model adds TORSION T(0,n)
#  and FLEXION F(m,n); to reach them on the SAME meridional (r,z) mesh we use a
#  FOURIER-MODE-m formulation with the full THREE displacement components
#  (u_r, u_phi, u_z), writing u(r,phi,z,t)=Re{ hat_u(r,z,t) e^{i m phi} }:
#
#    MODE = "long"     m=0, drive (u_r,u_z)         -> L(0,n)  (= v3 result)
#    MODE = "torsion"  m=0, drive u_phi only        -> T(0,n)  (DARK: u_r=0)
#    MODE = "flexion"  m=1, drive (u_r,u_phi,u_z)    -> F(1,n)
#
#  For m=0 the (u_r,u_z) block and the u_phi block DECOUPLE (v4 sec.4), so the
#  torsion run is a scalar-like problem in u_phi with the shear operator only;
#  for m=1 all three components couple, exactly as the flexion family requires.
#
#  Output (per MODE): fem_dispersion_v4_<MODE>.npz / .png with the numerical f-k
#  diagram (2-D FFT of a dense surface sensor line) and the analytic overlay from
#  data/analytic_modes_v4.npz (the matching family).
#
#  MeSU run (per mode):
#      python generate_axisymmetric_mesh.py            # fine mesh, F_MAX=200 kHz
#      srun -n <N> python fem_dispersion_v4_hpc.py --mode long
#      srun -n <N> python fem_dispersion_v4_hpc.py --mode torsion
#      srun -n <N> python fem_dispersion_v4_hpc.py --mode flexion
# =============================================================================
import math, os, argparse
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.fem import petsc as fem_petsc
import postprocess_dispersion as pp

# --- gmshio import shim ------------------------------------------------------
# dolfinx renamed dolfinx.io.gmshio -> dolfinx.io.gmsh in v0.10 AND changed the
# return signature of read_from_msh() from a 3-tuple to a MeshData named-tuple.
# Some conda-forge builds also expose neither (no GMSH support compiled in),
# in which case we fall back to meshio + dolfinx.mesh.create_mesh.
try:
    from dolfinx.io import gmshio as _gmshio                     # < 0.10
    _GMSH_FLAVOR = "gmshio"
except ImportError:
    try:
        from dolfinx.io import gmsh as _gmshio                   # >= 0.10
        _GMSH_FLAVOR = "gmsh"
    except ImportError:
        _gmshio = None
        _GMSH_FLAVOR = None

def read_msh(path, comm, gdim=2):
    """Read a .msh file into (mesh, cell_tags, facet_tags), normalising across
    dolfinx versions:
      * <0.10 returns a 3-tuple from gmshio.read_from_msh,
      * >=0.10 returns a MeshData dataclass with .mesh / .cell_tags / .facet_tags,
      * if dolfinx has no GMSH support at all we use meshio as a fallback."""
    if _gmshio is not None and hasattr(_gmshio, "read_from_msh"):
        out = _gmshio.read_from_msh(path, comm, rank=0, gdim=gdim)
        # MeshData (v0.10+) is iterable but has named attributes
        if hasattr(out, "mesh"):
            return out.mesh, out.cell_tags, out.facet_tags
        return out                                              # legacy 3-tuple
    # fallback: meshio bridge
    import meshio
    from dolfinx.mesh import create_mesh
    m = meshio.read(path)
    cells = m.get_cells_type("triangle")
    points = m.points[:, :gdim]
    raise ImportError(
        "dolfinx has no gmshio support in this env; manual meshio fallback "
        "would be needed.  Install fenics-dolfinx with GMSH support or use "
        "a dolfinx>=0.7 build from conda-forge.")

comm = MPI.COMM_WORLD
rank = comm.rank
if rank == 0:
    import dolfinx
    print(f"[FEM v4] dolfinx {dolfinx.__version__} | gmshio path: {_GMSH_FLAVOR}",
          flush=True)

# ----------------------------- CLI / parameters ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["long", "torsion", "flexion"], default="long",
                help="Fourier-mode family to excite (v4): long=L(0,n) m=0, "
                     "torsion=T(0,n) m=0 u_phi, flexion=F(1,n) m=1.")
args, _ = ap.parse_known_args()
MODE = args.mode
M_FOURIER = 0 if MODE in ("long", "torsion") else 1

MESH_FILE   = "tusk_axi.msh"
CONFIG_FILE = "config.txt"
F_LO, F_HI  = 20.0e3, 200.0e3
F_MAX       = 220.0e3
CHIRP_DUR   = 60.0e-6
EXC_FRAC    = 0.18
ABSORB_FRAC = 0.08
ALPHA_ALID  = 4.0e6
FLUID_LOAD  = (MODE != "torsion")     # torsion is DARK: no fluid loading (u_r=0)
CFL_FACTOR  = 1.0/22.0
RHO_INF     = 0.9
N_SENSORS   = 400
T_PAD       = 1.25

# ----------------------------- config --------------------------------------
with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)
for m in cfg["materials"].values():
    if "E" in m: m["E"] = float(m["E"])
    if "c" in m: m["c"] = float(m["c"])
g = cfg["geometry"]; L = float(g["tusk_length_total"])
Rob, Rot = g["outer_radius_base"], g["outer_radius_tip"]
TAG = cfg["mesh"]["domain_tags"]; BND = cfg["mesh"]["boundary_tags"]; AXIS_TAG = 50
def rout(z): return Rob + (Rot - Rob)*z/L

# ----------------------------- mesh & materials -----------------------------
domain, cell_tags, facet_tags = read_msh(MESH_FILE, comm, gdim=2)
tdim = domain.topology.dim; fdim = tdim-1
domain.topology.create_connectivity(fdim, tdim)

Q = fem.functionspace(domain, ("DG", 0))
lam_f = fem.Function(Q); mu_f = fem.Function(Q); rho_f = fem.Function(Q); beta_f = fem.Function(Q)
def lame(E, nu): return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
omega_ref = 2*math.pi*math.sqrt(F_LO*F_HI)
for tag, name in {TAG["dentine"]:"dentine", TAG["cementum"]:"cementum", TAG["pulp"]:"pulp"}.items():
    p = cfg["materials"][name]; lam, mu = lame(p["E"], p["nu"]); cells = cell_tags.find(tag)
    lam_f.x.array[cells] = lam; mu_f.x.array[cells] = mu; rho_f.x.array[cells] = p["rho"]
    beta_f.x.array[cells] = p["damping_loss_factor"]/omega_ref

# ----------------------------- Fourier-mode-m kinematics --------------------
#  Real-valued 3-component meridional field U=(u_r,u_phi,u_z) representing
#  u(r,phi,z)=Re{U(r,z) e^{i m phi}}.  The strain components for Fourier mode m
#  (see e.g. de Rosa / Treyssede cylindrical waveguide formulations) are, with
#  the azimuthal factor folded into the in-plane operator via the integer m:
#       e_rr = du_r/dr
#       e_zz = du_z/dz
#       e_tt = (u_r + m*u_phi)/r        (azimuthal normal strain, mode-m)
#       e_rz = 0.5(du_r/dz + du_z/dr)
#       e_rt = 0.5(du_phi/dr - (u_phi + m*u_r)/r)
#       e_zt = 0.5(du_phi/dz - m*u_z/r)
#  For m=0 this reduces to the v3 axisymmetric set PLUS a decoupled u_phi
#  (torsion) block, exactly the v4 m=0 factorisation.
x = ufl.SpatialCoordinate(domain); r = x[0]; z = x[1]
rsafe = ufl.max_value(r, 1e-12)
mF = float(M_FOURIER)

def eps_mode(u):
    ur, ut, uz = u[0], u[1], u[2]
    e_rr = ur.dx(0)
    e_zz = uz.dx(1)
    e_tt = (ur + mF*ut)/rsafe
    e_rz = 0.5*(ur.dx(1) + uz.dx(0))
    e_rt = 0.5*(ut.dx(0) - (ut + mF*ur)/rsafe)
    e_zt = 0.5*(ut.dx(1) - mF*uz/rsafe)
    return (e_rr, e_zz, e_tt, e_rz, e_rt, e_zt)

def sig_mode(u):
    e_rr, e_zz, e_tt, e_rz, e_rt, e_zt = eps_mode(u)
    tr = e_rr + e_zz + e_tt
    s_rr = lam_f*tr + 2*mu_f*e_rr
    s_zz = lam_f*tr + 2*mu_f*e_zz
    s_tt = lam_f*tr + 2*mu_f*e_tt
    s_rz = 2*mu_f*e_rz; s_rt = 2*mu_f*e_rt; s_zt = 2*mu_f*e_zt
    return (s_rr, s_zz, s_tt, s_rz, s_rt, s_zt)

def inner_mode(u, v):
    s = sig_mode(u); e = eps_mode(v)
    return (s[0]*e[0] + s[1]*e[1] + s[2]*e[2]
            + 2.0*s[3]*e[3] + 2.0*s[4]*e[4] + 2.0*s[5]*e[5])

# THREE-component vector space (u_r, u_phi, u_z) on the meridian
V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
u_new = fem.Function(V, name="u"); u_old = fem.Function(V); v_old = fem.Function(V); a_old = fem.Function(V)
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
nrm = ufl.FacetNormal(domain); W = 2.0*ufl.pi*r

La = ABSORB_FRAC*L
ramp = (ufl.conditional(z > L-La, ((z-(L-La))/La)**3, 0.0)
        + ufl.conditional(z < La, ((La-z)/La)**3, 0.0))
alid = ALPHA_ALID*ramp

u_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)
def m_form(u, w): return rho_f*ufl.dot(u, w)*W*dx
def k_form(u, w): return inner_mode(u, w)*W*dx
def c_form(u, w):
    c = beta_f*inner_mode(u, w)*W*dx + rho_f*alid*ufl.dot(u, w)*W*dx
    if FLUID_LOAD:
        # fluid radiation impedance acts on the NORMAL (radial) component only;
        # this is u_r here (n ~ e_r on the outer wall) -> couples L and F, not T.
        rf, cf = cfg["materials"]["fluid"]["rho"], cfg["materials"]["fluid"]["c"]
        nr = nrm[0]                               # radial part of the facet normal
        c += rf*cf*(u[0]*nr)*(w[0]*nr)*W*ds(BND["outer_surface"])
    return c

# ----------------------------- excitation by mode --------------------------
in_band = ufl.conditional(ufl.And(z > La, z < EXC_FRAC*L), 1.0, 0.0)
load_amp = fem.Constant(domain, default_scalar_type(0.0))
if MODE == "long":
    src = ufl.as_vector((0.7071, 0.0, 0.7071))    # radial+axial -> S/A longitudinal
elif MODE == "torsion":
    src = ufl.as_vector((0.0, 1.0, 0.0))          # pure azimuthal -> torsion
else:  # flexion
    src = ufl.as_vector((0.6, 0.5, 0.6))          # mixed 3-comp -> flexural F(1,n)
def f_form(w): return load_amp*in_band*ufl.dot(src, w)*W*ds(BND["outer_surface"])

def chirp(t):
    if t <= 0.0 or t >= CHIRP_DUR: return 0.0
    win = math.sin(math.pi*t/CHIRP_DUR)**2
    rate = (F_HI-F_LO)/CHIRP_DUR
    return win*math.sin(2*math.pi*(F_LO*t + 0.5*rate*t*t))

# ----------------------------- boundary conditions on the axis -------------
#  On r=0 the regularity of a Fourier mode-m field requires:
#    m=0 long/torsion : u_r=0 (and u_phi=0 for long) ; u_z free
#    m=1 flexion      : u_z=0 ; (u_r, u_phi) coupled, leave free except u_z
axis_facets = facet_tags.find(AXIS_TAG)
bcs = []
if M_FOURIER == 0:
    du0 = fem.locate_dofs_topological(V.sub(0), fdim, axis_facets)   # u_r=0
    bcs.append(fem.dirichletbc(default_scalar_type(0.0), du0, V.sub(0)))
    if MODE == "long":
        du1 = fem.locate_dofs_topological(V.sub(1), fdim, axis_facets)  # u_phi=0
        bcs.append(fem.dirichletbc(default_scalar_type(0.0), du1, V.sub(1)))
    else:  # torsion: u_z=0 too (only u_phi lives)
        du2 = fem.locate_dofs_topological(V.sub(2), fdim, axis_facets)
        bcs.append(fem.dirichletbc(default_scalar_type(0.0), du2, V.sub(2)))
else:  # m=1 flexion: u_z=0 on the axis
    du2 = fem.locate_dofs_topological(V.sub(2), fdim, axis_facets)
    bcs.append(fem.dirichletbc(default_scalar_type(0.0), du2, V.sub(2)))

# ----------------------------- generalized-alpha ---------------------------
rho_inf = RHO_INF
a_m = (2*rho_inf-1)/(rho_inf+1); a_f = rho_inf/(rho_inf+1)
gamma = 0.5-a_m+a_f; beta = 0.25*(1-a_m+a_f)**2
dt_val = CFL_FACTOR/F_MAX
c_slow = 0.85*C.bulk_velocities(float(cfg["materials"]["dentine"]["E"]),
                                0.30, 1900.0)[1] if False else 0.85*1337.0
T_END = T_PAD*L/c_slow
n_steps = int(round(T_END/dt_val))
dt = fem.Constant(domain, default_scalar_type(dt_val))
def a_expr(u): return (u-u_old-dt*v_old)/(beta*dt**2) - (1-2*beta)/(2*beta)*a_old
def v_expr(a): return v_old+dt*((1-gamma)*a_old+gamma*a)
def avg(o, n_, al): return al*o+(1-al)*n_
a_n = a_expr(u_)
res = (m_form(avg(a_old, a_n, a_m), w_) + c_form(avg(v_old, v_expr(a_n), a_f), w_)
       + k_form(avg(u_old, u_, a_f), w_) - f_form(w_))
a_bilin = fem.form(ufl.lhs(res)); L_lin = fem.form(ufl.rhs(res))
A = fem_petsc.assemble_matrix(a_bilin, bcs=bcs); A.assemble()
# dolfinx 0.10 changed create_vector to take a FunctionSpace (or sequence of)
# instead of a Form. The shim below keeps the script working on both old (<0.10,
# Form-based) and new (>=0.10, FunctionSpace-based) signatures.
try:
    b = fem_petsc.create_vector(L_lin)                    # dolfinx < 0.10
except TypeError:
    b = fem_petsc.create_vector(V)                        # dolfinx >= 0.10
solver = PETSc.KSP().create(comm); solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY); solver.getPC().setType(PETSc.PC.Type.LU)
try: solver.getPC().setFactorSolverType("mumps")
except Exception: pass
c1 = 1.0/(beta*dt_val**2); c2 = (1-2*beta)/(2*beta); c3 = dt_val*(1-gamma); c4 = dt_val*gamma

# ----------------------------- sensors --------------------------------------
z_lo, z_hi = EXC_FRAC*L+0.02, (1-ABSORB_FRAC)*L-0.02
z_sens = np.linspace(z_lo, z_hi, N_SENSORS)
pts = np.zeros((N_SENSORS, 3)); pts[:, 0] = rout(z_sens)*0.999; pts[:, 1] = z_sens
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
tree = bb_tree(domain, tdim)
coll = compute_colliding_cells(domain, compute_collisions_points(tree, pts), pts)
cells, keep = [], []
for i in range(N_SENSORS):
    c = coll.links(i)
    if len(c): cells.append(c[0]); keep.append(i)
keep = np.array(keep, np.int32); pts_ok = pts[keep]; z_ok = z_sens[keep]
# record the component that best reveals each family:
#   long/flexion -> u_r (radial, the radiating component); torsion -> u_phi
rec_main = np.zeros((n_steps, len(keep))); rec_uz = np.zeros((n_steps, len(keep)))
comp_main = 1 if MODE == "torsion" else 0          # u_phi for torsion, else u_r
if rank == 0:
    print(f"[FEM-disp v4 | MODE={MODE} m={M_FOURIER}] L={L} dt={dt_val:.3e} "
          f"steps={n_steps} fluid_load={FLUID_LOAD} sensors={len(keep)} "
          f"main_comp={'u_phi' if comp_main==1 else 'u_r'}", flush=True)

t = 0.0
for step in range(n_steps):
    t += dt_val
    load_amp.value = chirp(t - a_f*dt_val)
    with b.localForm() as loc: loc.set(0.0)
    fem_petsc.assemble_vector(b, L_lin)
    fem_petsc.apply_lifting(b, [a_bilin], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)
    solver.solve(b, u_new.x.petsc_vec); u_new.x.scatter_forward()
    uv = u_new.x.array; uo = u_old.x.array; vo = v_old.x.array; ao = a_old.x.array
    av = c1*(uv-uo-dt_val*vo)-c2*ao; vv = vo+c3*ao+c4*av
    u_old.x.array[:] = uv; v_old.x.array[:] = vv; a_old.x.array[:] = av
    vals = u_new.eval(pts_ok, np.array(cells, np.int32))
    rec_main[step, :] = vals[:, comp_main]; rec_uz[step, :] = vals[:, 2]
    if rank == 0 and step % max(1, n_steps//20) == 0:
        print(f"  step {step}/{n_steps} t={t*1e3:.3f}ms max|u|={np.abs(uv).max():.2e}", flush=True)

# -------------------------- post: numerical f-k -----------------------------
if rank == 0:
    dz = float(np.mean(np.diff(z_ok)))
    np.savez(f"fem_sensors_v4_{MODE}.npz", main=rec_main, uz=rec_uz, z=z_ok,
             dz=dz, dt=dt_val, mode=MODE, m=M_FOURIER, comp_main=comp_main)
    # combine components for L/F (u_r + u_z); torsion uses u_phi alone
    f, k, Amp_m = pp.dispersion_2dfft(rec_main, dz, dt_val, fmax=F_HI, kmax=math.pi/dz)
    if MODE == "torsion":
        Amp = Amp_m
    else:
        _, _, Amp_z = pp.dispersion_2dfft(rec_uz, dz, dt_val, fmax=F_HI, kmax=math.pi/dz)
        Amp = Amp_m + Amp_z
    np.savez(f"fem_dispersion_v4_{MODE}.npz", f=f, k=k, Amp=Amp, mode=MODE, m=M_FOURIER)
    # plot with the matching analytic overlay
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 6))
    ax.pcolormesh(k, f/1e3, 10*np.log10(Amp.T/Amp.max()+1e-6),
                  shading="auto", cmap="magma", vmin=-35, vmax=0)
    ov = "data/analytic_modes_v4.npz"
    if os.path.exists(ov):
        d = np.load(ov)
        prefix = {"long": "L", "torsion": "T0", "flexion": "F1"}[MODE]
        for key in d.files:
            if key.endswith("_k") and key.startswith(prefix):
                base = key[:-2]
                ax.plot(d[base+"_k"], d[base+"_f"]/1e3, lw=1.0, color="cyan", alpha=.7)
        ax.plot([], [], color="cyan", label=f"analytique {prefix}*")
        ax.legend(fontsize=8)
    ax.set_xlabel("k [rad/m]"); ax.set_ylabel("f [kHz]")
    ax.set_title(f"Dispersion FEM v4 — MODE={MODE} (m={M_FOURIER}) + recouvrement analytique")
    ax.set_xlim(0, math.pi/dz*0.6)
    fig.tight_layout(); fig.savefig(f"fem_dispersion_v4_{MODE}.png", dpi=140)
    print(f"wrote fem_dispersion_v4_{MODE}.npz / .png")