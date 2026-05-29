#!/usr/bin/env python3
# =============================================================================
#  cylindrical_dispersion.py   --   v4 THREE-POTENTIAL dispersion engine
#
#  Replaces the flat-plate Rayleigh-Lamb core (analytic_dispersion.py, v3) by
#  the EXACT elastodynamic dispersion of a solid/annular CYLINDER, built on the
#  Helmholtz decomposition into THREE potentials (cf. v4 document, sec.4 & annex):
#
#      u = grad(Phi)            (P  , irrotational      -> dilatation)
#        + curl(Psi e_s)        (SH , azimuthal         -> torsion)
#        + curlcurl(chi e_s)    (SV , divergence-free   -> in-plane shear)
#
#  Each potential obeys a Helmholtz equation; the three are coupled ONLY through
#  the radial-traction boundary conditions, giving det M(k,w)=0.  The azimuthal
#  order m selects the family:
#
#      m = 0 :  det factorises  ->  L(0,n) longitudinal   (Phi,chi coupled)
#                                   T(0,n) torsion         (Psi alone, DARK mode)
#      m >=1 :  no factorisation ->  F(m,n) flexion        (Phi,chi,Psi coupled)
#
#  KEY v4 RESULTS encoded here (see annex of the v4 document):
#    * Torsion fundamental T(0,1) is EXACTLY non-dispersive: cph = cT for all w.
#    * Higher torsion T(0,n>=2) cut off at zeros of J2 : fc = j_{2,n-1} cT/(2 pi R).
#    * Torsion has u_r = 0 on the surface -> it does NOT radiate in a perfect
#      fluid (acoustically dark) and is invisible to a normal-incidence LDV.
#    * Radiation requires BOTH cph>c0 (real Mach angle) AND U_r != 0.
#
#  All SI.  Verified against: cT (torsion), bar speed sqrt(E/rho) (L(0,1) low-f),
#  Pochhammer-Chree longitudinal, and the v3 plate limit at high kR (thin wall).
#
#  Free solid cylinder (radius R) uses regular Bessel J_m only.  An annular /
#  multilayer wall (pulp channel) is handled by build_layer_matrix() with J_m
#  and Y_m and a 6x6 transfer assembly (Lowe 1995); used by the HPC FEM as the
#  analytic overlay, and by the BEM as the surface-velocity polarisation.
# =============================================================================
import numpy as np
from scipy.special import jv, yv, jn_zeros, jve, yve

# ----------------------------------------------------------------------------
# 0. isotropic bulk velocities and the elastic moduli inversion (v4 eq. inversion)
# ----------------------------------------------------------------------------
def bulk_velocities(E, nu, rho):
    cL = np.sqrt(E*(1-nu)/(rho*(1+nu)*(1-2*nu)))     # dilatational (P)
    cT = np.sqrt(E/(2*rho*(1+nu)))                    # shear (S)
    return cL, cT

def rayleigh_speed(cT, nu):                           # Bergmann approximation
    return cT*(0.862+1.14*nu)/(1+nu)

def bar_speed(E, rho):                                # L(0,1) low-frequency limit
    return np.sqrt(E/rho)

def moduli_from_speeds(cL, cT, rho):                  # v4 inversion (EXP-1)
    E = rho*cT**2*(3*cL**2-4*cT**2)/(cL**2-cT**2)
    nu = (cL**2-2*cT**2)/(2*(cL**2-cT**2))
    return E, nu

# ----------------------------------------------------------------------------
# 2. TORSION  T(0,n)  -- exact, analytic (Psi alone, m=0)
#    BC sigma_rphi(R)=0  <=>  beta R J0(beta R) - 2 J1(beta R) = 0  <=>  J2(beta R)=0
#    Fundamental: beta->0 => cph = cT (non-dispersive).
# ----------------------------------------------------------------------------
def torsion_branches(freqs, cT, R, n_modes=4):
    """Return list of torsion branches T(0,1..n_modes) as dicts f[],k[],cph[].
    T(0,1): cph=cT exactly (non-dispersive).  T(0,n>=2): J2(beta R)=0 cutoffs."""
    w = 2*np.pi*np.asarray(freqs, float)
    branches = []
    # T(0,1): non-dispersive, exists at all frequencies
    k1 = w/cT
    branches.append(dict(name="T01", f=np.asarray(freqs, float),
                         k=k1, cph=np.full_like(w, cT),
                         cg=np.full_like(w, cT), radiates=False))
    # T(0,n>=2): beta R = j_{2,n-1};  k^2 = kT^2 - (j/R)^2 ; cutoff where k->0
    j2 = jn_zeros(2, n_modes-1)
    for n, jz in enumerate(j2, start=2):
        beta = jz/R
        kT = w/cT
        k2 = kT*kT - beta*beta
        good = k2 > 0
        kk = np.sqrt(np.where(good, k2, np.nan))
        cph = np.where(good, w/kk, np.nan)
        # group velocity cg = c_T^2 k / w  (from w^2 = cT^2(k^2+beta^2))
        cg = np.where(good, cT*cT*kk/w, np.nan)
        branches.append(dict(name=f"T0{n}", f=np.asarray(freqs, float),
                             k=kk, cph=cph, cg=cg, radiates=False))
    return branches

# ----------------------------------------------------------------------------
# 3. LONGITUDINAL L(0,n) and FLEXION F(m,n) -- free-cylinder frequency equations.
#    Both are evaluated with exponentially-scaled Bessel functions (jve) so the
#    determinant stays O(1) and smooth for cph below cT or cL (large-imaginary
#    transverse arguments).  m=0 factorises into the longitudinal (Phi,chi) pair
#    [Pochhammer-Chree] and the torsion (Psi) factor [handled above]; m>=1 keeps
#    the full 3x3 (Phi,chi,Psi) coupling [Pochhammer flexural].
# ----------------------------------------------------------------------------
def _row_normalize_c(M):
    """Divide each row by its max-abs entry (a positive real), preserving the
    determinant's argument up to a positive factor (keeps sign changes robust)."""
    s = np.max(np.abs(M), axis=1)
    s[s == 0] = 1.0
    return M/s[:, None]

def _det_longitudinal(w, k, cL, cT, R, rho):
    """m=0 LONGITUDINAL L(0,n): canonical Pochhammer-Chree frequency equation for
    the free solid cylinder (Achenbach 1973 / Graff 1975), the exact m=0 (Phi,chi)
    factor of the three-potential system [v4 annex eq. 42]. Evaluated with
    exponentially-SCALED Bessel functions (jve) so it stays O(1) and smooth for
    cph below cT or cL (large-imaginary transverse arguments), which is essential
    for reliable root tracing.

        2 (alpha/R)(beta^2+k^2) J1(aR) J1(bR)
          - (beta^2-k^2)^2 J0(aR) J1(bR)
          - 4 k^2 alpha beta J1(aR) J0(bR) = 0
    """
    a = np.sqrt(complex((w/cL)**2 - k*k))
    b = np.sqrt(complex((w/cT)**2 - k*k))
    aR = a*R; bR = b*R
    # scaled Bessel: Jn(z) = jve(n,z)*exp(|Im z|); the common exp factors cancel
    # because every term below is first order in a J(aR) and first order in a J(bR).
    J0a = jve(0, aR); J1a = jve(1, aR)
    J0b = jve(0, bR); J1b = jve(1, bR)
    val = (2.0*(a/R)*(b*b + k*k)*J1a*J1b
           - (b*b - k*k)**2 * J0a*J1b
           - 4.0*k*k*a*b*J1a*J0b)
    return val.real

def _det_flexion(m, w, k, cL, cT, R, rho):
    """m>=1 FLEXION F(m,n): full 3x3 frequency determinant of the free solid
    cylinder, all three potentials coupled (Phi,chi,Psi); Pochhammer flexural
    equation (Pao & Mindlin 1960, Graff 1975). Column A holds P-Bessel J(aR),
    columns B,C hold S-Bessel J(bR); using SCALED jve per column factors out the
    exp growth uniformly (the determinant picks up one exp|Im aR| and two
    exp|Im bR|, a positive factor that does not move the roots)."""
    a = np.sqrt(complex((w/cL)**2 - k*k))
    b = np.sqrt(complex((w/cT)**2 - k*k))
    aR = a*R; bR = b*R
    kT2 = (w/cT)**2; R2 = R*R
    Ja = jve(m, aR); Jap = 0.5*(jve(m-1, aR) - jve(m+1, aR))
    Jb = jve(m, bR); Jbp = 0.5*(jve(m-1, bR) - jve(m+1, bR))
    rr_A = (2*m*m/R2 - (kT2 - 2*k*k))*Ja - (2.0/R)*(a*Jap)
    rr_B = 2*k*((m*m/R2 - b*b)*Jb - (1.0/R)*(b*Jbp))
    rr_C = 2*m*((1.0/R)*(b*Jbp) - (1.0/R2)*Jb)
    rp_A = 2*m*((1.0/R2)*Ja - (1.0/R)*(a*Jap))
    rp_B = 2*k*m*((1.0/R2)*Jb - (1.0/R)*(b*Jbp))
    rp_C = (b*b - 2*m*m/R2)*Jb + (2.0/R)*(b*Jbp)
    rs_A = 2*k*(a*Jap)
    rs_B = (kT2 - 2*k*k)*(b*Jbp)
    rs_C = (m/R)*k*Jb
    M = np.array([[rr_A, rr_B, rr_C],
                  [rp_A, rp_B, rp_C],
                  [rs_A, rs_B, rs_C]], dtype=complex)
    M = _row_normalize_c(M)
    return np.linalg.det(M).real

def _roots_in_k(detfun, w, cL, cT, R, rho, cph_min, cph_max, n_scan):
    """Find k-roots of detfun at fixed w by sign changes of the (scaled, O(1))
    determinant. A thin guard band around the bulk speeds cT and cL is skipped,
    where the frequency equation is singular by construction (J<->I transition)."""
    k_lo = w/cph_max; k_hi = w/cph_min
    ks = np.linspace(k_lo, k_hi, n_scan)
    vals = np.array([detfun(w, k, cL, cT, R, rho) for k in ks])
    sign = np.sign(vals)
    idx = np.where(np.diff(sign) != 0)[0]
    def near_bulk(k):
        c = w/k
        return (abs(c - cT) < 0.004*cT) or (abs(c - cL) < 0.004*cL)
    roots = []
    for i in idx:
        ka, kb = ks[i], ks[i+1]
        fa = vals[i]
        for _ in range(60):
            km = 0.5*(ka+kb); fm = detfun(w, km, cL, cT, R, rho)
            if fm == 0 or (kb-ka) < 1e-7*km:
                break
            if (fa < 0) != (fm < 0):
                kb = km
            else:
                ka, fa = km, fm
        kr = 0.5*(ka+kb)
        if not near_bulk(kr):
            roots.append(kr)
    return np.array(sorted(roots))

def longitudinal_roots_at_freq(f, cL, cT, R, rho, cph_min, cph_max, n_scan=4000):
    w = 2*np.pi*f
    return _roots_in_k(lambda w, k, cL, cT, R, rho: _det_longitudinal(w, k, cL, cT, R, rho),
                       w, cL, cT, R, rho, cph_min, cph_max, n_scan)

def flexion_roots_at_freq(f, m, cL, cT, R, rho, cph_min, cph_max, n_scan=4000):
    w = 2*np.pi*f
    return _roots_in_k(lambda w, k, cL, cT, R, rho: _det_flexion(m, w, k, cL, cT, R, rho),
                       w, cL, cT, R, rho, cph_min, cph_max, n_scan)

# ----------------------------------------------------------------------------
# 4. branch continuation (shared by L and F families) -- continuity in cph
# ----------------------------------------------------------------------------
def trace_fundamental(freqs, detfun_at, k_seed, cph_window=0.5):
    """Robustly trace ONE branch (e.g. the F(1,1) flexural fundamental) by local
    continuation: at each frequency, bracket the single root nearest the previous
    k, searching only a narrow window around the linear prediction. This sidesteps
    the dense spurious-root cluster of the brute-force scan in cph in [cT,cL].

    detfun_at(w,k) -> real determinant; k_seed = starting wavenumber at freqs[0]."""
    fs = np.asarray(freqs, float)
    out_f, out_k, out_c = [], [], []
    k_prev = k_seed
    k_prevprev = None
    for f in fs:
        w = 2*np.pi*f
        # predict next k by linear extrapolation if we have two points
        if k_prevprev is not None and len(out_f) >= 2:
            k_pred = 2*out_k[-1] - out_k[-2]
        else:
            k_pred = k_prev
        # search window around prediction
        dk = cph_window*abs(k_pred) + 1e-3
        ka, kb = max(k_pred-dk, 1e-3), k_pred+dk
        ks = np.linspace(ka, kb, 400)
        vals = np.array([detfun_at(w, k) for k in ks])
        sgn = np.sign(vals)
        idx = np.where(np.diff(sgn) != 0)[0]
        if len(idx) == 0:
            break                                   # branch lost (e.g. cutoff)
        # pick the sign-change bracket whose midpoint is closest to k_pred
        mids = 0.5*(ks[idx]+ks[idx+1])
        j = idx[np.argmin(np.abs(mids - k_pred))]
        kA, kB = ks[j], ks[j+1]; fA = vals[j]
        for _ in range(60):
            km = 0.5*(kA+kB); fm = detfun_at(w, km)
            if fm == 0 or (kB-kA) < 1e-8*km:
                break
            if (fA < 0) != (fm < 0):
                kB = km
            else:
                kA, fA = km, fm
        kr = 0.5*(kA+kB)
        out_f.append(f); out_k.append(kr); out_c.append(w/kr)
        k_prevprev = k_prev; k_prev = kr
    out_f = np.array(out_f); out_k = np.array(out_k); out_c = np.array(out_c)
    cg = np.gradient(2*np.pi*out_f, out_k) if len(out_f) > 2 else np.full_like(out_f, np.nan)
    return dict(f=out_f, k=out_k, cph=out_c, cg=cg)

def trace_family(freqs, roots_at_freq, family_label, gap_tol=0.15):
    branches = []
    for f in freqs:
        rt = roots_at_freq(f)
        w = 2*np.pi*f
        cph = w/rt if len(rt) else np.array([])
        used = [False]*len(rt)
        for br in branches:
            if not br['_open']:
                continue
            last = br['cph'][-1]
            best, bi = None, -1
            for j, c in enumerate(cph):
                if used[j]:
                    continue
                rel = abs(c-last)/last
                if rel < gap_tol and (best is None or rel < best):
                    best, bi = rel, j
            if bi >= 0:
                used[bi] = True
                br['f'].append(f); br['k'].append(rt[bi]); br['cph'].append(cph[bi])
            else:
                br['_open'] = False
        for j, c in enumerate(cph):
            if not used[j]:
                branches.append(dict(f=[f], k=[rt[j]], cph=[c], _open=True))
    for n, br in enumerate(branches):
        br['f'] = np.array(br['f']); br['k'] = np.array(br['k']); br['cph'] = np.array(br['cph'])
        br.pop('_open', None)
        if len(br['f']) > 2:
            br['cg'] = np.gradient(2*np.pi*br['f'], br['k'])
        else:
            br['cg'] = np.full_like(br['f'], np.nan)
    branches.sort(key=lambda b: b['f'][0])
    branches = [b for b in branches if len(b['f']) >= 1]
    for n, br in enumerate(branches):
        br['name'] = f"{family_label}{n}"
        br['radiates'] = True            # L and F carry u_r != 0
    return branches

# ----------------------------------------------------------------------------
# 5. leaky attenuation alpha(f) and Mach angle for a RADIATING branch.
#    Only modes with U_r != 0 radiate; cph>c0 required for a real Mach angle.
#    Surface u_r amplitude from the eigenvector (A,B,C) of M at the root.
# ----------------------------------------------------------------------------
def _surface_ur(m, w, k, cL, cT, R, rho):
    """|U_r|^2 at r=R (per unit potential amplitude), from the null-vector of the
    radial-traction system. For m=0 the longitudinal pair (Phi,chi) is used and
    U_r = A alpha J1(alpha R) + B i k beta J1(beta R) (chi part); torsion (Psi)
    contributes ZERO to U_r -> dark mode. For m>=1 the 3x3 null-vector is used."""
    a = np.sqrt(complex((w/cL)**2 - k*k))
    b = np.sqrt(complex((w/cT)**2 - k*k))
    if m == 0:
        # LONGITUDINAL surface u_r from the 2x2 (Phi,chi) null vector.
        # Rows are sigma_rr, sigma_rs (=0); columns are (A=Phi, B=chi).
        # Using scaled J1 (= -J' for order 0) keeps entries O(1).
        kT2 = (w/cT)**2
        M2 = np.array([
            [(2*k*k - kT2)*jv(0, a*R),      2j*k*b*(-jv(1, b*R))],
            [2j*k*a*(-jv(1, a*R)),          (kT2 - 2*k*k)*(-jv(1, b*R))]
        ], dtype=complex)
        _, _, Vt = np.linalg.svd(M2)
        A, B = Vt[-1]
        # u_r = A alpha J0'(alpha R) + B i k beta J0'(beta R), with J0'=-J1
        Ur = A*a*(-jv(1, a*R)) + B*1j*k*b*(-jv(1, b*R))
        return float(abs(Ur)**2)
    else:
        Ja = jv(m, a*R); Jap = 0.5*(jv(m-1, a*R) - jv(m+1, a*R))
        Jb = jv(m, b*R); Jbp = 0.5*(jv(m-1, b*R) - jv(m+1, b*R))
        R2 = R*R; kT2 = (w/cT)**2
        rr_A = (2*m*m/R2 - (kT2 - 2*k*k))*Ja - (2.0/R)*(a*Jap)
        rr_B = 2*k*((m*m/R2 - b*b)*Jb - (1.0/R)*(b*Jbp))
        rr_C = 2*m*((1.0/R)*(b*Jbp) - (1.0/R2)*Jb)
        rp_A = 2*m*((1.0/R2)*Ja - (1.0/R)*(a*Jap))
        rp_B = 2*k*m*((1.0/R2)*Jb - (1.0/R)*(b*Jbp))
        rp_C = (b*b - 2*m*m/R2)*Jb + (2.0/R)*(b*Jbp)
        rs_A = 2*k*(a*Jap); rs_B = (kT2 - 2*k*k)*(b*Jbp); rs_C = (m/R)*k*Jb
        M = _row_normalize_c(np.array([[rr_A, rr_B, rr_C],
                                       [rp_A, rp_B, rp_C],
                                       [rs_A, rs_B, rs_C]], dtype=complex))
        _, _, Vt = np.linalg.svd(M)
        A, B, Cc = Vt[-1]
        Ur = A*(a*Jap) + B*1j*k*(b*Jbp) + Cc*(m/R)*Jb
        return float(abs(Ur)**2)

def leaky_alpha_branch(branch, m, cL, cT, R, rho, rho0, c0):
    f = branch['f']; k = branch['k']; cph = branch['cph']; cg = branch['cg']
    alpha = np.full_like(f, np.nan); thetaM = np.full_like(f, np.nan)
    for i in range(len(f)):
        if not np.isfinite(cg[i]) or cg[i] <= 0 or cph[i] <= c0:
            continue
        w = 2*np.pi*f[i]
        ur2 = _surface_ur(m, w, k[i], cL, cT, R, rho)
        if ur2 <= 0:
            continue
        thetaM[i] = np.arcsin(c0/cph[i])
        I_rad = 0.5*rho0*c0*w*w*ur2/np.cos(thetaM[i])
        P_tr = 0.5*rho*w*w*ur2*R*cg[i]      # transported power scale (uniform-wall)
        alpha[i] = I_rad/(2*P_tr) if P_tr > 0 else np.nan
    branch['alpha'] = alpha; branch['thetaM'] = thetaM
    return branch

# ----------------------------------------------------------------------------
# 6. high-level driver: all three families on a frequency grid
# ----------------------------------------------------------------------------
def all_families(freqs, cL, cT, R, rho, c0, cph_min=300., cph_max=9000.,
                 n_scan=4000, m_flex=(1,), n_torsion=4, min_pts=5):
    """Return dict with 'L', 'T', 'F<m>' branch lists -- the v4 mode zoo.

    L(0,n)  : longitudinal, brute-force scan + cph-continuation (Pochhammer-Chree).
    T(0,n)  : torsion, exact analytic (J2 cutoffs); T(0,1) non-dispersive at cT.
    F(m,1)  : flexural FUNDAMENTAL only, traced by robust local continuation.
              Higher flexural overtones F(m,n>=2) are intentionally delegated to
              the FEM (fem_dispersion_hpc.py, Fourier mode m): the free-cylinder
              flexural determinant has a dense spurious-root cluster in cph in
              [cT,cL] that makes brute-force enumeration unreliable, whereas a
              single fundamental follows cleanly from a low-frequency seed."""
    out = {}
    # longitudinal L(0,n)
    L = trace_family(freqs,
                     lambda f: longitudinal_roots_at_freq(f, cL, cT, R, rho,
                                                          cph_min, cph_max, n_scan),
                     "L", gap_tol=0.15)
    L = [b for b in L if len(b['f']) >= min_pts]
    for n, b in enumerate(L):
        b['name'] = f"L{n}"
    out['L'] = L
    # torsion T(0,n) -- analytic, always present
    out['T'] = torsion_branches(freqs, cT, R, n_modes=n_torsion)
    # flexion fundamental F(m,1) for each requested azimuthal order m
    for m in m_flex:
        # seed the fundamental at the lowest frequency from the global scan
        f0 = freqs[0]; w0 = 2*np.pi*f0
        rt0 = flexion_roots_at_freq(f0, m, cL, cT, R, rho, cph_min, cph_max, n_scan)
        if len(rt0) == 0:
            out[f'F{m}'] = []
            continue
        k_seed = np.max(rt0)                       # lowest cph = highest k = fundamental
        br = trace_fundamental(freqs,
                               lambda w, k, mm=m: _det_flexion(mm, w, k, cL, cT, R, rho),
                               k_seed, cph_window=0.35)
        br['name'] = f"F{m}1"; br['radiates'] = True
        out[f'F{m}'] = [br] if len(br['f']) >= min_pts else []
    return out


if __name__ == "__main__":
    # quick self-test: torsion fundamental must be exactly cT, and the
    # longitudinal low-frequency branch must approach the bar speed.
    E, nu, rho = 10.3e9, 0.30, 1900.0
    cL, cT = bulk_velocities(E, nu, rho)
    R = 0.0287
    print(f"cL={cL:.0f} cT={cT:.0f} cR={rayleigh_speed(cT,nu):.0f} "
          f"c_bar={bar_speed(E,rho):.0f}")
    T = torsion_branches([20e3, 70e3, 200e3], cT, R)
    print("T(0,1) cph (should equal cT):", T[0]['cph'])
    print("T(0,2) cutoff kHz (J2):", jn_zeros(2,1)[0]*cT/(2*np.pi*R)/1e3)
    # inversion round-trip
    Eb, nub = moduli_from_speeds(cL, cT, rho)
    print(f"inversion round-trip: E={Eb/1e9:.2f} GPa (10.3), nu={nub:.3f} (0.30)")
