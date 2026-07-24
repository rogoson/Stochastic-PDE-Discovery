import matplotlib.pyplot as plt
import numpy as np
from diffeqpy import de


def f(u, p, t):
    return 1.0


def g(u, p, t):
    return 1.0


u0 = 0.0
tspan = (0.0, 1.0)
dt = 0.001

# Use the same noise process for all solvers for fair comparison
prob = de.SDEProblem(f, g, u0, tspan)

configs = [
    ("EM (fixed, dt=0.001)", de.solve(prob, de.EM(), dt=dt, adaptive=False)),
    ("SRIW1 (fixed, dt=0.001)", de.solve(prob, de.SRIW1(), dt=dt, adaptive=False)),
    ("SRIW1 (adaptive, dtmax=0.01)", de.solve(prob, de.SRIW1(), dt=dt, dtmax=0.01)),
    ("SRIW1 (adaptive, dtmax=0.05)", de.solve(prob, de.SRIW1(), dt=dt, dtmax=0.05)),
    ("SRIW1 (adaptive, no dtmax)", de.solve(prob, de.SRIW1(), dt=dt)),
]

fig, axes = plt.subplots(1, len(configs), figsize=(16, 4), sharey=True)

for ax, (label, sol) in zip(axes, configs):
    t = np.array(sol.t)
    u = np.array(sol.u)
    ax.plot(t, u, color="darkorange", linewidth=0.8)
    ax.set_title(label, fontsize=8)
    ax.set_xlabel("Time $t$")
    ax.grid(True, alpha=0.3)
    # Annotate number of steps taken
    ax.text(
        0.05,
        0.95,
        f"Steps: {len(t)-1}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

axes[0].set_ylabel("$W_t$")
fig.suptitle(
    "Pure Brownian motion: effect of solver and step-size control on path resolution",
    fontsize=10,
    y=1.02,
)
plt.tight_layout()
plt.savefig("worthwhileImages/solver_comparison.pdf", bbox_inches="tight")
plt.show()
