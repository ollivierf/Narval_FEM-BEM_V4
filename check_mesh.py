#!/usr/bin/env python3
"""Inspecte un .msh surfacique en agregeant TOUS les blocs de triangles
(format MSH 4.x = plusieurs blocs par entite). Detecte les noeuds orphelins.
f_max valide a 6 et 4 elements / longueur d'onde (c0=1450 m/s)."""
import sys, numpy as np, meshio
C0 = 1450.0
fn = sys.argv[1] if len(sys.argv) > 1 else "tusk_surface.msh"
m = meshio.read(fn)
pts = m.points

# lister TOUS les blocs
print(f"fichier       : {fn}")
print("blocs cellules:")
for cb in m.cells:
    print(f"   - {cb.type:12s} x {len(cb.data)}")

# agreger TOUS les blocs de triangles
tri_blocks = [cb.data for cb in m.cells if cb.type == "triangle"]
if not tri_blocks:
    print("Aucun bloc 'triangle' -> pas un maillage surfacique 3D.")
    sys.exit(1)
tris = np.vstack(tri_blocks)

n_pts_file = len(pts)
used = np.unique(tris)                  # noeuds reellement references
n_used = len(used)
n_orphan = n_pts_file - n_used

e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
L = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
hmin, hmean, hmax = L.min(), L.mean(), L.max()
bbox = pts.max(0) - pts.min(0)

print(f"\nnoeuds fichier: {n_pts_file}")
print(f"noeuds utilises: {n_used}")
print(f"noeuds orphelins: {n_orphan}  <-- si >0, bempp gonfle les DOFs pour rien")
print(f"triangles total: {len(tris)}")
print(f"ratio tri/noeud: {len(tris)/n_used:.2f}  (≈2 pour une surface saine)")
print(f"bbox (x,y,z)  : {bbox[0]*1e3:.1f} x {bbox[1]*1e3:.1f} x {bbox[2]*1e3:.1f} mm")
print(f"arete min/moy/max: {hmin*1e3:.2f} / {hmean*1e3:.2f} / {hmax*1e3:.2f} mm")
print(f"matrice dense (noeuds utilises): {(n_used**2*16)/2**30:.1f} GiB")
print(f"f_max valide  : {C0/(6*hmax)/1e3:.0f} kHz (6 el/l) | {C0/(4*hmax)/1e3:.0f} kHz (4 el/l)")