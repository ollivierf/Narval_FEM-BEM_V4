#!/usr/bin/env python3
# =============================================================================
#  radiation_farfield_v4.py   --   v4 far-field radiation (leaky-wave model)
#
#  Counterpart of v3 radiation_farfield.py, on the THREE-POTENTIAL cylinder.
#  For each frequency 20..200 kHz (step 5 kHz) every RADIATING supersonic mode
#  (cph>c0 AND u_r != 0) radiates from a finite aperture L with complex axial
#  wavenumber k_z = beta - i alpha.  The meridional far field is the leaky-wave
#  antenna integral (v4 document, sec. radiation):
#
#      p(theta) ~ sin(theta) * (1 - exp(-(alpha+i Delta)L)) / (alpha+i Delta)
#      Delta(theta) = k0 cos(theta) - beta,   k0 = omega/c0
#      main lobe at  theta_M = arccos(c0/cph)        (from the tusk axis)
#
#  v4 DIFFERENCE FROM v3:  the mode set now contains the longitudinal family
#  L(0,n) AND the flexural fundamental F(1,1) (both have u_r != 0), while the
#  TORSION family T(0,n) is EXCLUDED -- it carries u_r = 0 and is acoustically
#  dark (no Mach radiation in a perfect fluid).  This is enforced by only
#  looping over fam['L'] and fam['F1'] and never fam['T'].
#
#  Outputs:
#    figures/radiation_polar_grid_v4.png   polars at a sample of frequencies
#    figures/radiation_fmap_v4.png         frequency-angle map (LWA signature)
#    figures/radiation_polar_all_v4/*.png  one polar per 5 kHz step
#    data/radiation_v4.npz                 theta grid + |p|^2(f,theta) + lobe pts
# =============================================================================
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cylindrical_dispersion as C

os.makedirs("figures/radiation_polar_all_v4", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- inputs (consistent with run_analytic_v4.py) ---------------------------
E, NU, RHO = 10.3e9, 0.30, 1900.0
RHO0, C0   = 1025.0, 1450.0
R_GUIDE    = 0.0287
L_APER     = 1.80                 # radiating aperture length [m]
cL, cT     = C.bulk_velocities(E, NU, RHO)

FREQS  = np.arange(20e3, 200e3+1, 1e3)
THETA  = np.linspace(0.1, 179.9, 720)
th_rad = np.radians(THETA)

# --- precompute the radiating branches once over the whole band ------------
fam = C.all_families(FREQS, cL, cT, R_GUIDE, RHO, C0,
                     n_scan=3000, m_flex=(1,), n_torsion=4)
# attach leakage to the radiating families (L and F); torsion stays dark
for b in fam['L']:
    C.leaky_alpha_branch(b, 0, cL, cT, R_GUIDE, RHO, RHO0, C0)
for b in fam['F1']:
    C.leaky_alpha_branch(b, 1, cL, cT, R_GUIDE, RHO, RHO0, C0)

def radiating_modes_at(f):
    """list of (cph, beta, alpha) for every RADIATING mode (L,F) supersonic at f.
    Torsion is never included (u_r=0 -> dark)."""
    out = []
    for fkey in ('L', 'F1'):
        for b in fam[fkey]:
            j = np.argmin(np.abs(b['f']-f))
            if abs(b['f'][j]-f) > 1.0:
                continue
            cph = b['cph'][j]
            al = b.get('alpha', np.full_like(b['f'], np.nan))[j]
            if not np.isfinite(cph) or cph <= C0 or not np.isfinite(al) or al <= 0:
                continue
            out.append((cph, 2*np.pi*f/cph, max(al, 1e-3)))
    return out

def pattern(f, modes):
    w = 2*np.pi*f; k0 = w/C0
    P = np.zeros_like(th_rad)
    for (cph, beta, alpha) in modes:
        Delta = k0*np.cos(th_rad) - beta
        G = (1 - np.exp(-(alpha + 1j*Delta)*L_APER))/(alpha + 1j*Delta)
        P += np.abs(np.sin(th_rad)*G)**2        # incoherent equal-excitation sum
    return P

# --- compute all -----------------------------------------------------------
P = np.zeros((len(FREQS), len(THETA)))
lobe_pts = []
for i, f in enumerate(FREQS):
    modes = radiating_modes_at(f)
    if modes:
        P[i] = pattern(f, modes)
        for (cph, beta, alpha) in modes:
            lobe_pts.append((f/1e3, np.degrees(np.arccos(np.clip(C0/cph, -1, 1)))))
    P[i] /= (P[i].max() + 1e-30)
lobe_pts = np.array(lobe_pts) if lobe_pts else np.zeros((0, 2))
print(f"computed {len(FREQS)} patterns; radiating modes @70kHz:",
      len(radiating_modes_at(70e3)), "(torsion excluded by construction)")

np.savez("data/radiation_v4.npz", freqs=FREQS, theta=THETA, P=P,
         L=L_APER, c0=C0, lobe_pts=lobe_pts)

# ---- FIG A : grid of polar diagrams ---------------------------------------
# subsample at a fixed 20 kHz cadence so the grid is the same shape (~10 panels)
# whether F_STEP=5 kHz or 1 kHz; otherwise the grid would explode at 1 kHz.
step_panel = max(1, int(round(20e3/(FREQS[1]-FREQS[0]))))
sel = FREQS[::step_panel]
ncol = 3; nrow = int(np.ceil(len(sel)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.4*nrow),
                         subplot_kw={"projection": "polar"})
for ax, f in zip(axes.ravel(), sel):
    i = np.argmin(np.abs(FREQS - f))
    pdb = np.clip(10*np.log10(P[i] + 1e-6), -20, 0)
    # axisymmetric -> draw the half-polar 0..180 deg only (no mirror)
    ax.plot(th_rad, pdb, color="#0050a0", lw=1.6)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_rlim(-20, 0); ax.set_rticks([0, -3, -6, -12, -20])
    ax.set_title(f"{f/1e3:.0f} kHz", fontsize=10, pad=12)
    ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180])
for ax in axes.ravel()[len(sel):]:
    ax.axis("off")
fig.suptitle("Rayonnement champ lointain v4 — demi-polaires (axisymétrique)\n"
             "modes L+F ; torsion sombre exclue", fontsize=12)
fig.tight_layout(); fig.savefig("figures/radiation_polar_grid_v4.png", dpi=140); plt.close(fig)
print("wrote figures/radiation_polar_grid_v4.png")

# ---- FIG B : frequency-angle map ------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 5.4))
im = ax.pcolormesh(THETA, FREQS/1e3, 10*np.log10(P + 1e-6),
                   shading="auto", cmap="turbo", vmin=-20, vmax=0)
if len(lobe_pts):
    ax.scatter(lobe_pts[:, 1], lobe_pts[:, 0], s=9, c="white", edgecolor="k",
               linewidth=.3, label=r"lobes $\arccos(c_0/c_\phi)$ par mode (L,F)")
fig.colorbar(im, label="|p|² normalisé [dB]", ticks=[0, -3, -6, -12, -20])
ax.set_xlabel("angle depuis l'axe $\\theta$ [deg]"); ax.set_ylabel("fréquence [kHz]")
ax.set_title("Carte fréquence–angle v4 (signature LWA, modes rayonnants)")
ax.legend(fontsize=8, loc="upper right"); ax.set_xlim(0, 180)
fig.tight_layout(); fig.savefig("figures/radiation_fmap_v4.png", dpi=140); plt.close(fig)
print("wrote figures/radiation_fmap_v4.png")

# ---- individual polars (half-polar, 20 dB dynamics) -----------------------
for i, f in enumerate(FREQS):
    fig = plt.figure(figsize=(4.4, 2.6)); ax = fig.add_subplot(111, projection="polar")
    pdb = np.clip(10*np.log10(P[i] + 1e-6), -20, 0)
    ax.plot(th_rad, pdb, "#0050a0", lw=1.5)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_rlim(-20, 0); ax.set_rticks([0, -3, -6, -12, -20])
    ax.set_title(f"{f/1e3:.0f} kHz", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"figures/radiation_polar_all_v4/polar_{f/1e3:03.0f}kHz.png", dpi=110)
    plt.close(fig)
print(f"wrote {len(FREQS)} individual polar plots in figures/radiation_polar_all_v4/")
