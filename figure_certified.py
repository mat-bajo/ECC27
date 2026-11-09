"""
fig_certified.py  --  single script for the figure

    necessary test (Toeplitz matrix K_r > 0)  +  certified region (sufficient bound)  +  spectral abscissa
in the plane (kappa, a),   a = c k2 / lam^2.

Neutral system   y'(t) = -kappa y(t) + a y'(t-h) + b y(t-2h),      b = k1 k2,   h = 1/lam,   |a| < 1.

    orange : flagged by the necessary test (K_r not > 0, or |a| >= 1) AND truly unstable  -> unstable, proved
    green  : certified (not flagged AND below a certified line), OR flagged by the necessary test but actually stable (false alarm)
    grey   : not flagged, not certified
    yellow : threshold of the spectral-method theorem  c_crit = (lam/|k2|) inf_w |(kappa + i lam w) e^{2iw} - k1 k2| / |w|
    dashed : sufficient bound a < sqrt(1-(b/kappa)^2)
    dotted : zero level of the computed spectral abscissa

usage:  python fig_certified.py [r] [k1]      (defaults r = 5, k1 = 1; writes fig_certified_r{r}_k1_{k1}.pdf / .png)
"""
import sys, os, numpy as np
from scipy.linalg import expm
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = int(sys.argv[1]) if len(sys.argv) > 1 else 5
K1 = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LAM, K2 = 1.0, 1.0
H, B = 1.0 / LAM, K1 * K2                             # delay h = 1/lam and coefficient b of y(t-2h)
XS   = np.linspace(0.2, 10, 56)                       # abscissa  x = kappa/|k1 k2|  (kappa = x |k1 k2|)
AS   = np.linspace(0.02, 1.15, 48)                    # ordinate a = c k2/lam^2  (never exactly 1)

# ---------------------------------------------------------------- Lyapunov matrix (neutral case, W = 1)
def lyap_neutral(kap, a, b, h):
    """u(tau), |tau| <= 2h, of  y' = -kap y + a y'(t-h) + b y(t-2h), |a|<1;  None if Lyapunov condition fails."""
    E  = np.array([[1, -a, 0, 0], [0, 1, -a, 0], [0, -a, 1, 0], [0, 0, -a, 1.]])
    M0 = np.array([[-kap, 0, b, 0], [0, -kap, 0, b], [-b, 0, kap, 0], [0, -b, 0, kap]])
    M  = np.linalg.solve(E, M0)
    S  = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]]); P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.]])
    N  = np.vstack([S @ expm(M * h) - P, [-a * b, -kap, a * kap, b]])
    if np.linalg.cond(N) > 1e12: return None
    c = np.linalg.solve(N, [0, 0, 0, -0.5])
    lam, V = np.linalg.eig(M); al = np.linalg.solve(V, c.astype(complex))
    def u(tau):
        t = np.abs(np.atleast_1d(np.asarray(tau, float))); out = np.empty_like(t); m1 = t <= h
        out[m1]  = (np.exp(np.outer(t[m1], lam))      * (V[1, :] * al)).sum(1).real
        out[~m1] = (np.exp(np.outer(t[~m1] - h, lam)) * (V[0, :] * al)).sum(1).real
        return out
    return u

def flagged(kap, a, b, h, r):
    """True <=> instability proved by the necessary test  K_r = [u(|i-j| 2h/(r-1))] not > 0."""
    if abs(a) >= 1: return True                       # difference operator not stable
    u = lyap_neutral(kap, a, b, h)
    if u is None: return True                         # Lyapunov condition fails (+-root pair)
    K = u(np.arange(r) * 2 * h / (r - 1))[np.abs(np.subtract.outer(np.arange(r), np.arange(r)))]
    return bool(np.linalg.eigvalsh(K).min() <= 0)

# ---------------------------------------------------------------- spectral abscissa
def abscissa(kap, a, b, h):
    """Rightmost root of  l(1 - a e^{-lh}) + kap - b e^{-2lh}  (Newton from a grid + from the neutral chain
    Re l = ln|a|/h, whose high-frequency roots can be the rightmost ones when a -> 1).  a >= 1: ln(a)/h (essential spectrum)."""
    if abs(a) >= 1: return np.log(abs(a)) / h + 1e-9
    X, Y = np.meshgrid(np.linspace(-3, 4, 12), np.linspace(0, max(30, 12 / h), 160))
    k = np.arange(0, min(int((abs(kap) + abs(b)) / (1 - abs(a)) * 1.2 * h / (2 * np.pi)) + 5, 800))
    l = np.concatenate([(X + 1j * Y).ravel(), np.log(abs(a)) / h + 2j * np.pi * k / h,
                        np.log(abs(a)) / h + 0.3 + 1j * (2 * np.pi * k + np.pi) / h])
    with np.errstate(all='ignore'):
        for _ in range(60):
            e = np.exp(-l * h); D = l * (1 - a * e) + kap - b * e**2; dD = (1 - a * e) + a * h * l * e + 2 * h * b * e**2
            l = np.where(np.abs(l) < 1e5, l - D / dD, l)
        e = np.exp(-l * h); D = l * (1 - a * e) + kap - b * e**2
    ok = np.isfinite(D) & (np.abs(D) < 1e-9) & (np.abs(l) < 1e5)
    return l.real[ok].max() if ok.any() else -1.0

def y_spec(x, W=3000.0, dw=0.004):
    """Ordinate c_crit |k2|/lam^2 = (1/lam) inf_{w>0} |(kappa + i lam w) e^{2iw} - k1 k2| / w  at x = kappa/|k1k2| > 1.
    Grid on (0, W]; beyond W the ratio is >= lam - (kappa+|b|)/W, which is used as a (conservative) tail bound."""
    w = np.arange(dw, W, dw); e2 = np.exp(2j * w); iwe = 1j * LAM * w * e2
    out = []
    for xi in np.atleast_1d(x):
        kap = xi * abs(B); f = np.abs(kap * e2 + iwe - B) / w
        out.append(min(f.min(), LAM - (kap + abs(B)) / W) / LAM)
    return np.array(out)

def work(p):
    kap = p[0] * abs(B)                               # p[0] = kappa/|k1 k2|
    return flagged(kap, p[1], B, H, R), abscissa(kap, p[1], B, H)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    tag = f"r{R}_k1_{K1:g}"; cache = f"fig_certified_{tag}.npz"
    pts = [(x, a) for a in AS for x in XS]
    if os.path.exists(cache): z = np.load(cache); F, A = z['F'], z['A']
    else:
        with Pool() as p: res = p.map(work, pts, chunksize=20)
        F = np.array([r[0] for r in res]).reshape(len(AS), len(XS)); A = np.array([r[1] for r in res]).reshape(F.shape)
        np.savez(cache, F=F, A=A)
    XX, AA = np.meshgrid(XS, AS)
    bound = lambda x: np.sqrt(np.clip(1 - 1 / x**2, 0, None))      # sufficient bound, x = kappa/|k1 k2|
    xx = np.linspace(1 + 1e-3, XS[-1], 300); ys = y_spec(xx)           # spectral-method curve (defined for x > 1 only)
    spec = lambda x: np.interp(x, xx, ys)
    stable = A < 0
    cert = ~F & (XX > 1) & (AA < np.maximum(bound(XX), spec(XX)))      # certified (kept for the consistency checks)
    green  = cert | (F & stable)                      # certified, or flagged by the test but actually stable (false alarm)
    orange = F & ~stable                              # flagged AND truly unstable
    grey   = ~F & ~cert                               # not flagged, not certified

    print(f"r = {R}: {F.size} points | orange {orange.sum()} | green {green.sum()} | grey {grey.sum()}")
    print("checks  - flagged but stable (false alarm, now green):", int((F & stable).sum()),
          "| certified but unstable:", int((cert & ~stable).sum()),
          "| retained but unstable:", int((~F & ~stable).sum()),
          "| stable and not certified:", int((stable & ~cert).sum()))

    below = (XX > 1) & (AA < spec(XX))
    print(f"spectral-method curve: max = {ys.max():.4f} (must be <= 1) | min(curve - dashed) = {(ys - bound(xx)).min():+.4f} | "
          f"points below it that are unstable: {int((below & ~stable).sum())} | stable points above it: {int((stable & ~below).sum())}")
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(4.8, 3.7), constrained_layout=True)
    for m, c in [(orange, '#D55E00'), (grey, '#c9c9c9'), (green, '#009E73')]:
        ax.scatter(XX[m], AA[m], s=9, color=c, linewidths=0)
    ax.contour(XS, AS, A, levels=[0], colors='#FFD700', linestyles=':', linewidths=2)
    ax.plot(xx, ys, 'r-', lw=2); ax.plot(xx, bound(xx), 'k--', lw=2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.15)
    ax.set_xlabel(r'$\kappa/|k_1k_2|$'); ax.set_ylabel(r'$|k_2| c /\lambda^2$')
    fig.savefig(f"fig_certified_{tag}.pdf"); fig.savefig(f"fig_certified_{tag}.png", dpi=200)