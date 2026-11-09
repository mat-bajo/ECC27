
"""
Numerical comparison of the three critical coupling bounds (constant lambda).

  c_spec : Amirat et al., Laplace-domain bound. NOT closed form: a minimum over the
           unbounded frequency axis of an oscillating function -> needs a truncated,
           discretised frequency grid (+ local refinement).
  c_lyap : Amirat et al., Lyapunov bound (closed form).
  c_new  : this paper, Lyapunov bound on the neutral-delay reformulation (closed form).

Single plot comparing four values k1 = -1, -0.25, 0.25, 1 (k2 = lambda = 1).
The four spectral bounds are computed separately, while the two Lyapunov
bounds are plotted only once, as they coincide under the chosen normalisation.

Normalised quantity plotted: |k2| * c_crit / lambda^2
Horizontal axis: kappa / |k1*k2|.

Run: python3 bounds_comparison.py -> writes bounds.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def c_spec(kappa, k1k2, k2=1.0, lam=1.0, wmax=None,
           dw=4e-3, refine=20):
    """Truncated frequency-domain minimisation."""
    b = abs(k1k2)
    if wmax is None:
        wmax = 8*kappa**2/(lam*b) + 100

    w = np.arange(1e-3, wmax, dw)

    g = lambda x: abs(
        (1j*lam*x + kappa)*np.exp(2j*x) - k1k2
    ) / x

    f = np.abs(
        (1j*lam*w + kappa)*np.exp(2j*w) - k1k2
    ) / w

    best = f.min()

    for i in np.argsort(f)[:refine]:
        r = minimize_scalar(
            g,
            bounds=(w[max(i-1, 0)],
                    w[min(i+1, len(w)-1)]),
            method="bounded",
            options={"xatol": 1e-10}
        )
        best = min(best, r.fun)

    return lam/abs(k2)*best


def c_lyap(kappa, k1, k2, lam=1.0):
    """Amirat et al., Lyapunov bound."""
    b = abs(k1*k2)
    if kappa <= b*np.e**2:
        return lam**2*np.sqrt(
            abs(k1)/(abs(k2)*kappa)
        )*np.log(kappa/b)
    return 2*lam**2/(np.e*abs(k2))


def c_new(kappa, k1, k2, lam=1.0):
    """Lyapunov bound on the neutral-delay reformulation."""
    return lam**2*np.sqrt(
        kappa**2 - (k1*k2)**2
    )/(kappa*abs(k2))


if __name__ == "__main__":

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2
    })

    fig, ax = plt.subplots(figsize=(6.9, 3.5))

    u = 1 + np.logspace(-2, np.log10(9), 140)
    k2, lam = 1.0, 1.0

    k1_values = [-1.0, -0.25, 0.25, 1.0]
    colors = ["C0", "C1", "C2", "C3"]

    # Four spectral bounds
    for k1, color in zip(k1_values, colors):
        b = abs(k1*k2)
        kap = u*b

        sp = np.array([
            c_spec(K, k1*k2, k2, lam)
            for K in kap
        ])

        ax.plot(
            u, sp,
            color=color, ls="-", lw=1.4,
            label=rf"$c_{{\rm freq}}^\ast,\ k_1={k1:g}$"
        )

    # Lyapunov bounds (plotted only once)
    k1_ref = 1.0
    kap = u*abs(k1_ref*k2)

    ly = np.array([
        c_lyap(K, k1_ref, k2, lam)
        for K in kap
    ])

    nw = np.array([
        c_new(K, k1_ref, k2, lam)
        for K in kap
    ])

    ax.plot(
        u, nw, "k--", lw=1.7,
        label=r"$c_{\rm Lyap\,TDS}^\ast$"
    )

    ax.plot(
        u, ly,"-.", color="#808080", lw=1.5,
        label=r"$c_{\rm Lyap}^\ast$"
    )

    # Reference horizontal lines
    ax.axhline(1, color="0.5", lw=0.6, ls=":")
    ax.axhline(2/np.e, color="0.5", lw=0.6, ls=":")

    ax.text(9.9, 0.98, r"$1$",
            ha="right", va="top", fontsize=8)

    ax.text(9.9, 2/np.e-0.02, r"$2/e$",
            ha="right", va="top", fontsize=8)

    ax.set_xlabel(r"$\kappa/|k_1k_2|$")
    ax.set_ylabel(r"$|k_2|\,c^\ast/\lambda^2$")

    ax.set_xlim(1, 10)
    ax.set_ylim(0, 1.06)
    ax.grid(alpha=0.25, lw=0.4)

    # Single legend
    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=True,
        framealpha=0.9
    )

    fig.tight_layout(pad=0.5)
    fig.savefig("bounds.pdf", bbox_inches="tight")
    fig.savefig("bounds.png", dpi=200)
    plt.close(fig)

    print("saved bounds.pdf")

