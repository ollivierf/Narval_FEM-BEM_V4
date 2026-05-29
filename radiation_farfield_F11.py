#!/usr/bin/env python3
# =============================================================================
#  radiation_farfield_F11.py
#  F(1,1)-ONLY far-field radiation -- counterpart of radiation_farfield_v4.py,
#  restricted to the flexural fundamental.
#
#  Use this when you want to study the LWA pattern of F(1,1) in isolation, e.g.
#  to vary the aperture length L_APER or the angular grid without re-running
#  run_analytic_F11.py (which traces dispersion + leakage and is the slow part).
#  This script READS data/F11_modes.npz (produced by run_analytic_F11.py) and
#  computes only the radiation integral.
#
#  v4 PHYSICS NOTE.  F(1,1) carries u_r != 0 and stays supersonic over the whole
#  20-200 kHz band -- it radiates everywhere.  Contrast with torsion T(0,n)
#  which has u_r=0 (acoustically dark) and is never included in any radiation
#  script, here or in radiation_farfield_v4.py.
#
#  Outputs:
#    figures/radiation_polar_grid_F11.png   polars at a sample of frequencies
#    figures/radiation_fmap_F11.png         frequency-angle map (LWA signature)
#    figures/radiation_polar_all_F11/       one polar per 5 kHz step (37 files)
#    data/F11_radiation.npz                 theta, |p|^2(f,theta), lobe points
# =============================================================================
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/radiation_polar_all_F11", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- parameters (consistent with radiation_farfield_v4.py) -----------------
L_APER = 1.80                            # radiating aperture length [m]
N_THETA = 720
COL_F11 = "#1a9850"

# --- load the F(1,1) branch saved by run_analytic_F11.py -------------------
MODES_FILE = "data/F11_modes.npz"
if not os.path.exists(MODES_FILE):
    raise SystemExit(
        f"{MODES_FILE} not found. Run 'python3 run_analytic_F11.py' first to "
        "produce the F(1,1) dispersion + leakage data this script depends on.")
d = np.load(MODES_FILE)
freqs = d['f']; cph = d['cph']; alpha = d['alpha']; thetaM = d['thetaM']
C0 = float(d['c0'])
print(f"loaded F(1,1): {len(freqs)} freqs, "
      f"cph {cph.min():.0f}->{cph.max():.0f} m/s, "
      f"alpha {np.nanmin(alpha):.2e}->{np.nanmax(alpha):.2e} Np/m")

THETA  = np.linspace(0.1, 179.9, N_THETA)
th_rad = np.radians(THETA)

# --- leaky-wave aperture integral (same form as radiation_farfield_v4.py) --
#   p(theta) ~ sin(theta) * (1 - exp(-(alpha+i Delta)L)) / (alpha+i Delta)
#   Delta(theta) = k0 cos(theta) - beta,   beta = w/cph,   k0 = w/c0
P = np.zeros((len(freqs), len(THETA)))
lobe_pts = []
for i, f in enumerate(freqs):
    if not (np.isfinite(cph[i]) and cph[i] > C0
            and np.isfinite(alpha[i]) and alpha[i] > 0):
        continue
    w = 2*np.pi*f; k0 = w/C0; beta = w/cph[i]; al = max(alpha[i], 1e-3)
    Delta = k0*np.cos(th_rad) - beta
    G = (1 - np.exp(-(al + 1j*Delta)*L_APER))/(al + 1j*Delta)
    P[i] = np.abs(np.sin(th_rad)*G)**2
    mx = P[i].max()
    if mx > 0:
        P[i] /= mx
    lobe_pts.append((f/1e3, np.degrees(np.arccos(np.clip(C0/cph[i], -1, 1)))))
lobe_pts = np.array(lobe_pts) if lobe_pts else np.zeros((0, 2))
np.savez("data/F11_radiation.npz", freqs=freqs, theta=THETA, P=P,
         L=L_APER, c0=C0, lobe_pts=lobe_pts)
print(f"computed {len(freqs)} F(1,1) patterns "
      f"(radiating @ {len(lobe_pts)} of {len(freqs)} frequencies)")

# ---- FIG A: polar grid (sample) -------------------------------------------
# subsample at a fixed 20 kHz cadence (~10 panels) independent of F_STEP
step_panel = max(1, int(round(20e3/(freqs[1]-freqs[0]))))
sel = freqs[::step_panel]
ncol = 3; nrow = int(np.ceil(len(sel)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.4*nrow),
                         subplot_kw={"projection": "polar"})
for ax, f in zip(axes.ravel(), sel):
    i = np.argmin(np.abs(freqs - f))
    pdb = np.clip(10*np.log10(P[i] + 1e-6), -20, 0)
    # axisymmetric -> draw the half-polar 0..180 deg only (no mirror)
    ax.plot(th_rad, pdb, color=COL_F11, lw=1.8)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_rlim(-20, 0); ax.set_rticks([0, -3, -6, -12, -20])
    ax.set_title(f"{f/1e3:.0f} kHz", fontsize=10, pad=12)
    ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180])
for ax in axes.ravel()[len(sel):]:
    ax.axis("off")
fig.suptitle("Rayonnement champ lointain — F(1,1) — demi-polaires (axisymétrique)\n"
             "axe du tusk = 0°, |p|² dB", fontsize=12)
fig.tight_layout(); fig.savefig("figures/radiation_polar_grid_F11.png", dpi=140); plt.close(fig)
print("wrote figures/radiation_polar_grid_F11.png")

# ---- FIG B: frequency-angle map -------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 5.4))
im = ax.pcolormesh(THETA, freqs/1e3, 10*np.log10(P + 1e-6),
                   shading="auto", cmap="turbo", vmin=-20, vmax=0)
if len(lobe_pts):
    ax.scatter(lobe_pts[:, 1], lobe_pts[:, 0], s=14, c="white", edgecolor="k",
               linewidth=.4, label=r"lobe $\arccos(c_0/c_\phi)$")
fig.colorbar(im, label="|p|² normalisé [dB]", ticks=[0, -3, -6, -12, -20])
ax.set_xlabel("angle depuis l'axe $\\theta$ [deg]"); ax.set_ylabel("fréquence [kHz]")
ax.set_title("Carte fréquence–angle du rayonnement F(1,1) (signature LWA)")
ax.legend(fontsize=9, loc="upper right"); ax.set_xlim(0, 180)
fig.tight_layout(); fig.savefig("figures/radiation_fmap_F11.png", dpi=140); plt.close(fig)
print("wrote figures/radiation_fmap_F11.png")

# ---- FIG C: individual polars (half-polar, 20 dB dynamics) -----------------
for i, f in enumerate(freqs):
    fig = plt.figure(figsize=(4.4, 2.6)); ax = fig.add_subplot(111, projection="polar")
    pdb = np.clip(10*np.log10(P[i] + 1e-6), -20, 0)
    ax.plot(th_rad, pdb, COL_F11, lw=1.5)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_rlim(-20, 0); ax.set_rticks([0, -3, -6, -12, -20])
    ax.set_title(f"F(1,1) — {f/1e3:.0f} kHz", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"figures/radiation_polar_all_F11/polar_{f/1e3:03.0f}kHz.png", dpi=110)
    plt.close(fig)
print(f"wrote {len(freqs)} individual polar plots in figures/radiation_polar_all_F11/")
