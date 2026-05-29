#!/usr/bin/env python3
# =============================================================================
#  run_analytic_F11.py
#  F(1,1)-ONLY analytic driver -- the dispersion + leakage counterpart of
#  run_analytic_v4.py, restricted to the flexural fundamental.
#
#  Mirrors the run_analytic_v4.py / radiation_farfield_v4.py split:
#    * THIS SCRIPT computes dispersion (cph, cg, f-k) and leakage (alpha,
#      thetaM) for F(1,1) and writes data/F11_modes.npz.
#    * radiation_farfield_F11.py reads that npz and computes the far-field
#      radiation (polar grid, f-angle map, individual polars).
#
#  Why a dedicated script?  F(1,1) is the lowest flexural branch (m=1, n=1) of
#  the three-potential cylinder (v4 doc).  It is supersonic over essentially the
#  whole 20-200 kHz band, so it is the most "leaky-antenna-like" of all the modes
#  in the zoo, and is the only flexural mode the analytic engine traces robustly
#  (higher F overtones are delegated to the FEM).  Isolating it here lets the
#  experimenter inspect its dispersion, group velocity, Mach angle, and leakage
#  without the visual clutter of L(0,n) and the dark torsion family.
#
#  Outputs (suffix _F11 everywhere, no overwriting of the full-zoo outputs):
#    figures/dispersion_cph_F11.png   cph(f), with cT, cR, cbar, c0
#    figures/dispersion_cg_F11.png    group velocity cg(f)
#    figures/fk_analytic_F11.png      f-k diagram for F(1,1) + water line
#    figures/leaky_alpha_F11.png      alpha(f) and Mach angle thetaM(f)
#    data/F11_modes.npz               f, k, cph, cg, alpha, thetaM
#
#  v4 PHYSICS NOTE.  F(1,1) carries u_r != 0 -> it radiates whenever cph>c0
#  (which is true everywhere in this band).  Contrast with torsion T(0,n), whose
#  u_r=0 makes it acoustically dark; nothing torsion-related appears here.
# =============================================================================
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cylindrical_dispersion as C

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- material & geometry (consistent with run_analytic_v4.py) --------------
E, NU, RHO = 10.3e9, 0.30, 1900.0       # axial dentine
RHO0, C0   = 1025.0, 1450.0             # arctic seawater
R_GUIDE    = 0.0287                      # base outer radius [m]
F_LO, F_HI, F_STEP = 20e3, 200e3, 1e3
CPH_MIN, CPH_MAX   = 300.0, 9000.0
N_SCAN     = 3000

cL, cT = C.bulk_velocities(E, NU, RHO)
cR     = C.rayleigh_speed(cT, NU)
cbar   = C.bar_speed(E, RHO)
freqs  = np.arange(F_LO, F_HI+1, F_STEP)
print(f"cL={cL:.0f} cT={cT:.0f} cR={cR:.0f} c_bar={cbar:.0f} c0={C0:.0f}")

# --- trace F(1,1) ONLY (skip the L and T families for speed) --------------
# Seed the fundamental from the lowest frequency's flexion scan, then continue
# locally -- same robust path the all_families() driver takes for F.
w0 = 2*np.pi*F_LO
rt0 = C.flexion_roots_at_freq(F_LO, 1, cL, cT, R_GUIDE, RHO,
                              CPH_MIN, CPH_MAX, N_SCAN)
if len(rt0) == 0:
    raise SystemExit("No F(1,1) seed found at the lowest frequency.")
k_seed = float(np.max(rt0))                 # highest k = lowest cph = fundamental
br = C.trace_fundamental(
        freqs,
        lambda w, k: C._det_flexion(1, w, k, cL, cT, R_GUIDE, RHO),
        k_seed, cph_window=0.35)
br['name'] = "F11"; br['radiates'] = True
C.leaky_alpha_branch(br, 1, cL, cT, R_GUIDE, RHO, RHO0, C0)
print(f"F(1,1): npts={len(br['f'])}  "
      f"cph {br['cph'].min():.0f}->{br['cph'].max():.0f}  "
      f"alpha min/max = {np.nanmin(br['alpha']):.2e} / {np.nanmax(br['alpha']):.2e}")

# --- save raw data ---------------------------------------------------------
np.savez("data/F11_modes.npz",
         f=br['f'], k=br['k'], cph=br['cph'], cg=br['cg'],
         alpha=br['alpha'], thetaM=br['thetaM'],
         cL=cL, cT=cT, cR=cR, cbar=cbar, c0=C0, R=R_GUIDE)

# common style
COL_F11 = "#1a9850"
LW      = 2.2

# ============================ FIG 1 : cph(f) ===============================
fig, ax = plt.subplots(figsize=(8.4, 5.4))
ax.plot(br['f']/1e3, br['cph'], color=COL_F11, lw=LW, label="F(1,1)")
for y, lab, c in [(cR, "$c_R$", "gray"), (cbar, "$c_{barre}$", "purple"),
                  (cT, "$c_T$", "0.4")]:
    ax.axhline(y, color=c, ls=":", lw=0.9)
    ax.text(F_HI/1e3*1.005, y, lab, va="center", fontsize=8, color=c)
ax.axhline(C0, color="k", lw=1.4); ax.text(F_HI/1e3*1.005, C0, "$c_0$", va="center", fontsize=8)
ax.fill_between([F_LO/1e3, F_HI/1e3], C0, 4000, color="gold", alpha=.08)
ax.text(60, C0+900, "zone rayonnante  $c_\\phi>c_0$\n"
        "(F(1,1) y reste en basse bande seulement)",
        color="darkgoldenrod", fontsize=9)
ax.set_xlim(F_LO/1e3, F_HI/1e3); ax.set_ylim(CPH_MIN, 4000)
ax.set_xlabel("fréquence [kHz]"); ax.set_ylabel("vitesse de phase $c_\\phi$ [m/s]")
ax.set_title("Dispersion analytique v4 — F(1,1) fondamental flexural")
ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("figures/dispersion_cph_F11.png", dpi=140); plt.close(fig)
print("wrote figures/dispersion_cph_F11.png")

# ============================ FIG 2 : f-k ==================================
fig, ax = plt.subplots(figsize=(7.6, 6.0))
ax.plot(br['k'], br['f']/1e3, color=COL_F11, lw=LW, label="F(1,1)")
ax.plot(2*np.pi*freqs/C0, freqs/1e3, "k-", lw=1.3, label="$k=\\omega/c_0$ (ligne d'eau)")
ax.plot(2*np.pi*freqs/cT, freqs/1e3, color="0.4", ls=":", lw=1.0, label="$k=\\omega/c_T$")
ax.set_xlabel("nombre d'onde axial $k$ [rad/m]"); ax.set_ylabel("fréquence [kHz]")
ax.set_title("Diagramme f–k — F(1,1)")
ax.set_xlim(0, 2*np.pi*F_HI/cR*1.05); ax.set_ylim(F_LO/1e3, F_HI/1e3)
ax.legend(fontsize=9); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("figures/fk_analytic_F11.png", dpi=140); plt.close(fig)
print("wrote figures/fk_analytic_F11.png")

# ============================ FIG 3 : cg(f) ================================
fig, ax = plt.subplots(figsize=(8.4, 5.0))
m = np.isfinite(br['cg']) & (br['cg'] > 0) & (br['cg'] < 6000)
ax.plot(br['f'][m]/1e3, br['cg'][m], color=COL_F11, lw=LW, label="F(1,1)")
ax.axhline(cT, color="0.4", ls=":", lw=0.9)
ax.text(F_HI/1e3*1.005, cT, "$c_T$", fontsize=8, color="0.4")
ax.set_xlabel("fréquence [kHz]"); ax.set_ylabel("vitesse de groupe $c_g$ [m/s]")
ax.set_title("Vitesse de groupe — F(1,1)")
ax.set_xlim(F_LO/1e3, F_HI/1e3); ax.set_ylim(0, 3500)
ax.legend(fontsize=10); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("figures/dispersion_cg_F11.png", dpi=140); plt.close(fig)
print("wrote figures/dispersion_cg_F11.png")

# ===================== FIG 4 : leaky alpha + Mach angle ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
al = br['alpha']; th = br['thetaM']
good = np.isfinite(al) & (al > 0)
ax1.semilogy(br['f'][good]/1e3, al[good], color=COL_F11, lw=LW, label="F(1,1)")
ax1.set_xlabel("fréquence [kHz]"); ax1.set_ylabel(r"atténuation par fuite $\alpha_s$ [Np/m]")
ax1.set_title("Atténuation de fuite — F(1,1) (mode rayonnant : $u_r\\neq0$)")
ax1.grid(alpha=.25, which="both"); ax1.legend(fontsize=10)

ax2.plot(br['f'][good]/1e3, np.degrees(th[good]), color=COL_F11, lw=LW, label="F(1,1)")
ax2.set_xlabel("fréquence [kHz]"); ax2.set_ylabel(r"angle de Mach $\theta_M$ [deg]")
ax2.set_title(r"Angle de rayonnement $\theta_M=\arcsin(c_0/c_\phi)$ — F(1,1)")
ax2.set_ylim(0, 90); ax2.grid(alpha=.25); ax2.legend(fontsize=10)
fig.tight_layout(); fig.savefig("figures/leaky_alpha_F11.png", dpi=140); plt.close(fig)
print("wrote figures/leaky_alpha_F11.png")

print("\nNext step: 'python3 radiation_farfield_F11.py' for the far-field "
      "radiation figures (uses data/F11_modes.npz).")
