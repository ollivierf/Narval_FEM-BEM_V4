import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dispersion_2dfft(U, dx, dt, fmax=None, kmax=None, win=True):
    """U[nt, nx] time-space sensor signal -> (f, k, |FFT|) for +x going waves."""
    nt, nx = U.shape
    if win:
        U = U * np.hanning(nt)[:,None] * np.hanning(nx)[None,:]
    # zero-pad for smoother spectra
    Nt = 1<<int(np.ceil(np.log2(nt)))+1
    Nx = 1<<int(np.ceil(np.log2(nx)))+1
    F = np.fft.fft2(U, s=(Nt,Nx))
    f = np.fft.fftfreq(Nt, dt)
    k = np.fft.fftfreq(Nx, dx)*2*np.pi          # rad/m
    F = np.fft.fftshift(F); f=np.fft.fftshift(f); k=np.fft.fftshift(k)
    A = np.abs(F)
    # +x-propagating waves live in the (+f, -k) quadrant with numpy's
    # exp(-i...) convention; take it and report |k|>=0.
    fi = f>=0; ki = k<=0
    f = f[fi]; k = -k[ki][::-1]
    A = A[np.ix_(fi, ki)][:, ::-1]
    if fmax: m=f<=fmax; f,A=f[m],A[m]
    if kmax: m=k<=kmax; k,A=k[m],A[:,m]
    return f, k, A

def plot_dispersion(f,k,A,overlay=None,title="k-f dispersion",out="dispersion.png"):
    plt.figure(figsize=(7,5))
    plt.pcolormesh(k, f/1e3, 20*np.log10(A/A.max()+1e-9),
                   shading="auto", cmap="magma", vmin=-40, vmax=0)
    plt.colorbar(label="|FFT| [dB]")
    if overlay:
        for name,(kk,ff) in overlay.items():
            plt.plot(kk, ff/1e3, "--", lw=1.3, label=name)
        plt.legend(fontsize=8, loc="lower right")
    plt.xlabel("wavenumber k [rad/m]"); plt.ylabel("frequency [kHz]")
    plt.title(title); plt.tight_layout(); plt.savefig(out, dpi=130)
    print("wrote", out)

# ----------------- SELF-TEST on synthetic 2-mode data -----------------
if __name__=="__main__":
    f0=25e3; nt=2000; dt=1.0/(20*40e3); nx=120; dx=0.012
    t=np.arange(nt)*dt; x=np.arange(nx)*dx
    Ncyc=6; tc=Ncyc/(2*f0)
    def burst(tt): return np.where((tt>0)&(tt<Ncyc/f0),
              np.sin(2*np.pi*f0*tt)*np.sin(np.pi*f0*tt/Ncyc)**2,0.0)
    # mode S0-like: fast non-dispersive c=2700; mode A0-like: slow dispersive c~sqrt(f)
    cS=2700.0
    U=np.zeros((nt,nx))
    for j,xj in enumerate(x):
        U[:,j]+= 1.0*burst(t-xj/cS)                       # S0
        cA=900.0*np.sqrt(f0/20e3)                          # A0 group~const here
        U[:,j]+= 0.8*burst(t-xj/cA)                        # A0
    f,k,A=dispersion_2dfft(U,dx,dt,fmax=60e3,kmax=300)
    # theory lines at f0
    overlay={"S0 c=2700": (np.array([2*np.pi*f0/cS]),np.array([f0])),
             "A0 c=900":  (np.array([2*np.pi*f0/(900*np.sqrt(f0/20e3))]),np.array([f0]))}
    # full lines vs f
    ff=np.linspace(5e3,60e3,200)
    overlay={"S0 (c=2700 m/s)":(2*np.pi*ff/cS, ff),
             "A0 (~900 m/s)":  (2*np.pi*ff/(900*np.sqrt(ff/20e3)), ff)}
    plot_dispersion(f,k,A,overlay,title="SELF-TEST synthetic 2-mode",out="selftest_dispersion.png")
    # check peak k at f0 for the fast mode
    fi=np.argmin(abs(f-f0)); kp=k[np.argmax(A[fi])]
    print(f"recovered dominant k at {f0/1e3:.0f}kHz: {kp:.1f} rad/m  (S0 expects {2*np.pi*f0/cS:.1f})")
