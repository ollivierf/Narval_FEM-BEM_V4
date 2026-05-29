#!/usr/bin/env python3
# =============================================================================
#  run_analytic_v4.py   --   v4 analytical guided-wave characterisation
#
#  Counterpart of the v3 run_analytic.py, rewritten for the THREE-POTENTIAL
#  cylindrical model (cylindrical_dispersion.py).  Produces, over 20-200 kHz:
#
#    figures/dispersion_cph_v4.png   cph(f) for L(0,n), T(0,n), F(1,1) + limits
#    figures/dispersion_cg_v4.png    group velocity cg(f)
#    figures/fk_analytic_v4.png      f-k diagram, three families colour-coded
#    figures/leaky_alpha_v4.png      leakage alpha(f), Mach angle thetaM(f)
#                                    (RADIATING modes only: L and F; T excluded)
#    data/analytic_modes_v4.npz      raw branches for FEM/BEM overlay
#    data/leaky_modes_v4.npz         (f, alpha, thetaM, cph) per radiating branch
#
#  KEY v4 POINTS made visible in the figures:
#    * T(0,1) is a horizontal line at cT (non-dispersive) and is drawn DASHED-GREY
#      and labelled "dark" -- it does not radiate (u_r = 0).
#    * the leaky panel contains NO torsion branch, by construction.
#    * radiation requires BOTH cph>c0 AND u_r != 0 -> only L and F appear there.
#
#  Geometry: solid cylinder of the OUTER radius (base) as the guiding section,
#  consistent with the v4 cylindrical formulation (the thin-wall plate proxy of
#  v3 is replaced by the true cylinder).  Material: refined axial dentine.
# =============================================================================
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cylindrical_dispersion as C

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- refined biomechanical inputs (see README / config_refined.yaml) --------
E, NU, RHO = 10.3e9, 0.30, 1900.0       # dentine (axial/stiff direction)
RHO0, C0   = 1025.0, 1450.0             # arctic seawater (~-1.5 C, S~34.5)
R_GUIDE    = 0.0287                      # outer radius at base [m] (guiding cylinder)
F_LO, F_HI, F_STEP = 10e3, 150e3, 1e3
CPH_MIN, CPH_MAX   = 300.0, 3000.0
N_SCAN     = 3000
N_TORSION  = 4
M_FLEX     = (1,)                         # azimuthal orders for flexion fundamentals

cL, cT = C.bulk_velocities(E, NU, RHO)
cR     = C.rayleigh_speed(cT, NU)
cbar   = C.bar_speed(E, RHO)
print(f"cL={cL:.0f} cT={cT:.0f} cR={cR:.0f} c_bar={cbar:.0f} c0={C0:.0f}")
print(f"torsion T(0,1) = cT = {cT:.0f} m/s (non-dispersive, DARK)")

freqs = np.arange(F_LO, F_HI+1, F_STEP)
fam = C.all_families(freqs, cL, cT, R_GUIDE, RHO, C0,
                     cph_min=CPH_MIN, cph_max=CPH_MAX, n_scan=N_SCAN,
                     m_flex=M_FLEX, n_torsion=N_TORSION)
print(f"L: {len(fam['L'])} branches | T: {len(fam['T'])} | "
      + " ".join(f"F{m}: {len(fam[f'F{m}'])}" for m in M_FLEX))

# --- save raw branches ------------------------------------------------------
save = dict(cL=cL, cT=cT, cR=cR, cbar=cbar, c0=C0, R=R_GUIDE)
for key, brs in fam.items():
    for b in brs:
        save[f"{b['name']}_f"]   = b['f']
        save[f"{b['name']}_k"]   = b['k']
        save[f"{b['name']}_cph"] = b['cph']
np.savez("data/analytic_modes_v4.npz", **save)

# ============================ FIG 1 : cph(f) ===============================
fig, ax = plt.subplots(figsize=(8.6, 5.8))
colL = plt.cm.Blues(np.linspace(0.45, 0.95, max(len(fam['L']), 1)))
colF = plt.cm.Greens(np.linspace(0.55, 0.9, max(len(fam['F1']), 1)))
for n, b in enumerate(fam['L']):
    ax.plot(b['f']/1e3, b['cph'], color=colL[n], lw=1.8,
            label="L(0,n)" if n == 0 else None)
for n, b in enumerate(fam['F1']):
    ax.plot(b['f']/1e3, b['cph'], color="#1a9850", lw=2.4, ls="-",
            label="F(1,1) flexion")
# torsion: dark modes, dashed grey
for n, b in enumerate(fam['T']):
    style = dict(color="0.45", lw=1.6, ls="--")
    ax.plot(b['f']/1e3, b['cph'], **style,
            label="T(0,n) (mode sombre)" if n == 0 else None)
ax.text(F_HI/1e3*0.62, cT+60, "T(0,1)=$c_T$  (sombre, $u_r$=0)",
        color="0.3", fontsize=8.5)
# reference speeds
for y, lab, c in [(cR, "$c_R$", "gray"), (cbar, "$c_{barre}$", "purple"),
                  (cT, "$c_T$", "0.4")]:
    ax.axhline(y, color=c, ls=":", lw=0.9)
    ax.text(F_HI/1e3*1.005, y, lab, va="center", fontsize=8, color=c)
ax.axhline(C0, color="k", lw=1.4); ax.text(F_HI/1e3*1.005, C0, "$c_0$", va="center", fontsize=8)
ax.fill_between([F_LO/1e3, F_HI/1e3], C0, 7000, color="gold", alpha=.06)
ax.text(55, C0+1700, "zone rayonnante  $c_\\phi>c_0$  (si $u_r\\neq0$)",
        color="darkgoldenrod", fontsize=9)
ax.set_xlim(F_LO/1e3, F_HI/1e3); ax.set_ylim(CPH_MIN, 7000)
ax.set_xlabel("fréquence [kHz]"); ax.set_ylabel("vitesse de phase $c_\\phi$ [m/s]")
ax.set_title("Dispersion v4 — cylindre de dentine (R=%.1f mm) : L, T, F" % (R_GUIDE*1e3))
ax.legend(ncol=2, fontsize=8.5, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("figures/dispersion_cph_v4.png", dpi=140); plt.close(fig)
print("wrote figures/dispersion_cph_v4.png")

# ============================ FIG 2 : f-k ==================================
fig, ax = plt.subplots(figsize=(7.8, 6.2))
for n, b in enumerate(fam['L']):
    ax.plot(b['k'], b['f']/1e3, color="#2171b5", lw=1.4,
            label="L(0,n)" if n == 0 else None)
for b in fam['F1']:
    ax.plot(b['k'], b['f']/1e3, color="#1a9850", lw=2.0, label="F(1,1)")
for n, b in enumerate(fam['T']):
    ax.plot(b['k'], b['f']/1e3, color="0.45", lw=1.4, ls="--",
            label="T(0,n) (sombre)" if n == 0 else None)
ax.plot(2*np.pi*freqs/C0, freqs/1e3, "k-", lw=1.3, label="$k=\\omega/c_0$ (ligne d'eau)")
ax.set_xlabel("nombre d'onde axial $k$ [rad/m]"); ax.set_ylabel("fréquence [kHz]")
ax.set_title("Diagramme f–k v4 (trois familles)")
ax.set_xlim(0, 2*np.pi*F_HI/cR*1.05); ax.set_ylim(F_LO/1e3, F_HI/1e3)
ax.legend(fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("figures/fk_analytic_v4.png", dpi=140); plt.close(fig)
print("wrote figures/fk_analytic_v4.png")

# ============================ FIG 3 : cg(f) ================================
fig, ax = plt.subplots(figsize=(8.6, 5.2))
for n, b in enumerate(fam['L']):
    m = np.isfinite(b['cg']) & (b['cg'] > 0) & (b['cg'] < 6000)
    ax.plot(b['f'][m]/1e3, b['cg'][m], color=colL[n], lw=1.6,
            label="L(0,n)" if n == 0 else None)
for b in fam['F1']:
    m = np.isfinite(b['cg']) & (b['cg'] > 0) & (b['cg'] < 6000)
    ax.plot(b['f'][m]/1e3, b['cg'][m], color="#1a9850", lw=2.2, label="F(1,1)")
for n, b in enumerate(fam['T']):
    m = np.isfinite(b['cg']) & (b['cg'] > 0)
    ax.plot(b['f'][m]/1e3, b['cg'][m], color="0.45", lw=1.4, ls="--",
            label="T(0,n) (sombre)" if n == 0 else None)
ax.axhline(cT, color="0.4", ls=":", lw=0.9); ax.text(F_HI/1e3*1.005, cT, "$c_T$", fontsize=8, color="0.4")
ax.set_xlabel("fréquence [kHz]"); ax.set_ylabel("vitesse de groupe $c_g$ [m/s]")
ax.set_title("Vitesse de groupe v4"); ax.set_xlim(F_LO/1e3, F_HI/1e3); ax.set_ylim(0, 4000)
ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("figures/dispersion_cg_v4.png", dpi=140); plt.close(fig)
print("wrote figures/dispersion_cg_v4.png")

# ===================== FIG 4 : leaky alpha + Mach (RADIATING only) =========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
leak_data = {}
radiating = [("L", fam['L'], 0, "#2171b5", "-")] \
          + [("F1", fam['F1'], 1, "#1a9850", "-")]
for fkey, brs, m, col, ls in radiating:
    for b in brs:
        C.leaky_alpha_branch(b, m, cL, cT, R_GUIDE, RHO, RHO0, C0)
        al, th = b["alpha"], b["thetaM"]
        good = np.isfinite(al) & (al > 0)
        if good.any():
            ax1.semilogy(b["f"][good]/1e3, al[good], color=col, ls=ls, lw=1.5,
                         label=b['name'])
            ax2.plot(b["f"][good]/1e3, np.degrees(th[good]), color=col, ls=ls, lw=1.5,
                     label=b['name'])
            leak_data[b['name']] = (b["f"][good], al[good], th[good], b["cph"][good])
ax1.set_xlabel("fréquence [kHz]"); ax1.set_ylabel(r"atténuation par fuite $\alpha_s$ [Np/m]")
ax1.set_title("Atténuation de fuite — modes RAYONNANTS (L, F)\n(torsion exclue : $u_r=0$)")
ax1.grid(alpha=.25, which="both"); ax1.legend(fontsize=7.5, ncol=2)
ax2.set_xlabel("fréquence [kHz]"); ax2.set_ylabel(r"angle de Mach $\theta_M$ [deg]")
ax2.set_title(r"Angle de rayonnement $\theta_M=\arcsin(c_0/c_\phi)$")
ax2.grid(alpha=.25); ax2.legend(fontsize=7.5, ncol=2); ax2.set_ylim(0, 90)
fig.tight_layout(); fig.savefig("figures/leaky_alpha_v4.png", dpi=140); plt.close(fig)
print("wrote figures/leaky_alpha_v4.png")

flat = {}
for nm, (fa, al, th, cp) in leak_data.items():
    for q, v in zip(("f", "alpha", "thetaM", "cph"), (fa, al, th, cp)):
        flat[f"{nm}_{q}"] = v
np.savez("data/leaky_modes_v4.npz", **flat)
print("wrote data/leaky_modes_v4.npz (radiating modes:", list(leak_data.keys()), ")")
