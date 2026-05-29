#!/usr/bin/env python3
# =============================================================================
#  generate_surface_mesh_v4.py
#  Standalone wetted-surface mesh generator (body of revolution) for the v4 BEM,
#  with the SAME per-band sizing + node cap used internally by
#  bem_radiation_v4_hpc.py.  Useful for diagnostics with check_mesh.py before a
#  full run, and to confirm the FMM problem size stays well below the dense limit
#  that crashed v3 (415 GiB at 167k nodes).
#
#  Usage:
#      python3 generate_surface_mesh_v4.py [F_kHz]      # default 200 kHz
#      python3 check_mesh.py tusk_surface_v4.msh        # inspect node count / mem
# =============================================================================
import math, sys
import gmsh
import yaml

F_MAX  = (float(sys.argv[1])*1e3) if len(sys.argv) > 1 else 200e3
C0     = 1450.0
ELELAM = 6.0
N_NODES_MAX = 120_000

with open("config.txt") as f:
    cfg = yaml.safe_load(f)
g = cfg["geometry"]
L  = float(g["tusk_length_total"])
Rb = float(g["outer_radius_base"])
Rt = float(g["outer_radius_tip"])

lam0 = C0/F_MAX
h_el = lam0/ELELAM
slant = math.hypot(L, Rb-Rt)
area = math.pi*(Rb+Rt)*slant + math.pi*Rb*Rb
def n_est(h): return int(2.3*area/(0.5*math.sqrt(3)*h*h))
capped = n_est(h_el) > N_NODES_MAX
if capped:
    h_el = math.sqrt(2.3*area/(0.5*math.sqrt(3)*N_NODES_MAX))

gmsh.initialize()
gmsh.model.add("tusk_surface_v4")
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
gmsh.model.setPhysicalName(2, 1, "tusk_wetted_surface")
gmsh.model.mesh.generate(2)
gmsh.write("tusk_surface_v4.msh")
nn = len(gmsh.model.mesh.getNodes()[0])
gmsh.finalize()

mem_fmm    = (3500.0*nn + 16.0*nn*math.log2(max(nn, 2)))/2**30
mem_nonloc = (1500.0*nn + 16.0*nn*math.log2(max(nn, 2)))/2**30
mem_dense  = (nn**2*16)/2**30
print(f"tusk_surface_v4.msh: ~{nn} nodes, h~{h_el*1e3:.2f} mm "
      f"(lambda0/{ELELAM:.0f} at {F_MAX/1e3:.0f} kHz){' [CAPPED]' if capped else ''}")
print(f"  estimated working set:  fmm ~ {mem_fmm:.2f} GB  |  "
      f"default_nonlocal ~ {mem_nonloc:.2f} GB  |  dense ~ {mem_dense:.0f} GB  "
      f"(<- v3 failure mode)")
