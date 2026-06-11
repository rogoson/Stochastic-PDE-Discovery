import numpy as np
import matplotlib.pyplot as plt
from diffeqpy import de
from juliacall import Main as jl

# simple 1D SDE: du = 0*dt + 1*dW (pure Brownian motion, true sigma^2 = 1)
# so drift is zero and we can isolate the noise structure

jl.seval("""
function zero_drift(du, u, p, t)
    du[1] = 0.0
end
function unit_noise(du, u, p, t)  
    du[1] = 1.0
end
""")

n_paths = 500
dt = 0.002
T = 1.0
n_steps = int(T / dt)

prob = de.SDEProblem(jl.zero_drift, jl.unit_noise, [0.0], (0.0, T), (0.0,))
ensemble = de.EnsembleProblem(prob)

# EM fixed
sol_em = de.solve(
    ensemble,
    de.EM(),
    de.EnsembleThreads(),
    trajectories=n_paths,
    adaptive=False,
    saveat=dt,
    dt=dt,
)

# SRIW1 adaptive with saveat
sol_adaptive = de.solve(
    ensemble, de.SRIW1(), de.EnsembleThreads(), trajectories=n_paths, saveat=dt, dt=dt
)


# extract increments
def get_increments(sol, n_paths, n_steps):
    increments = []
    for i in range(n_paths):
        u = np.array([np.array(s) for s in sol.u[i].u]).flatten()
        inc = np.diff(u)
        # get times and get total time just to ensure that the trajectories are the same length in time
        times = np.array([t for t in sol.u[i].t])
        total_time = times[-1] - times[0]
        print(
            f"Path {i}: total time = {total_time:.4f}, expected {T:.4f}, increments shape: {inc.shape}"
        )
        increments.extend(inc.tolist())
    return np.array(increments)


inc_em = get_increments(sol_em, n_paths, n_steps)
inc_adaptive = get_increments(sol_adaptive, n_paths, n_steps)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# plot 1: histogram of increments
axes[0].hist(
    inc_em, bins=50, alpha=0.5, density=True, label=f"EM (var={np.var(inc_em):.4f})"
)
axes[0].hist(
    inc_adaptive,
    bins=50,
    alpha=0.5,
    density=True,
    label=f"SRIW1 adaptive (var={np.var(inc_adaptive):.4f})",
)
axes[0].axvline(x=0, color="k", linestyle="--")
axes[0].set_title("Increment distributions")
axes[0].set_xlabel("u(t+dt) - u(t)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# plot 2: variance of increments - should be dt=0.002 for both
axes[1].bar(
    ["EM", "SRIW1 adaptive", "True (dt)"],
    [np.var(inc_em), np.var(inc_adaptive), dt],
    color=["steelblue", "darkorange", "red"],
    alpha=0.7,
)
axes[1].set_title(f"Increment variance (should be dt={dt})")
axes[1].set_ylabel("Variance")
axes[1].grid(True, alpha=0.3)

# plot 3: KM estimate = mean of squared increments / dt
km_em = np.mean(inc_em**2) / dt
km_adaptive = np.mean(inc_adaptive**2) / dt
axes[2].bar(
    ["EM", "SRIW1 adaptive", "True σ²=1"],
    [km_em, km_adaptive, 1.0],
    color=["steelblue", "darkorange", "red"],
    alpha=0.7,
)
axes[2].set_title("KM diffusion estimate")
axes[2].set_ylabel("σ² estimate")
axes[2].grid(True, alpha=0.3)

plt.suptitle(
    "Pure Brownian motion: EM vs SRIW1 adaptive increment structure", fontweight="bold"
)
plt.tight_layout()
plt.savefig("worthwhileImages/increment_comparison.pdf", dpi=150, bbox_inches="tight")
plt.show()

print(f"EM increment variance: {np.var(inc_em):.6f} (expected {dt})")
print(f"SRIW1 adaptive increment variance: {np.var(inc_adaptive):.6f} (expected {dt})")
print(f"EM KM estimate: {km_em:.4f} (expected 1.0)")
print(f"SRIW1 adaptive KM estimate: {km_adaptive:.4f} (expected 1.0)")


# plot a single path at high resolution for both solvers
prob_single = de.SDEProblem(jl.zero_drift, jl.unit_noise, [0.0], (0.0, 0.1), (0.0,))

# EM fixed - save at every internal step
sol_em_dense = de.solve(prob_single, de.EM(), adaptive=False, dt=0.0001, saveat=0.0001)

# SRIW1 adaptive - let it choose its own steps completely freely
sol_sriw1_dense = de.solve(
    prob_single,
    de.SRIW1(),
    adaptive=True,
    abstol=1e-8,
    reltol=1e-8,
    saveat=0.0001,
)

t_em = np.array(sol_em_dense.t)
u_em = np.array([float(u[0]) for u in sol_em_dense.u])

t_sriw1 = np.array(sol_sriw1_dense.t)
u_sriw1 = np.array([float(u[0]) for u in sol_sriw1_dense.u])

print(f"EM saved points: {len(t_em)}")
print(f"SRIW1 saved points: {len(t_sriw1)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# full paths
axes[0, 0].plot(t_em, u_em, linewidth=0.8, color="steelblue")
axes[0, 0].set_title("EM (fixed step) — full path", fontweight="bold")
axes[0, 0].set_xlabel("t")
axes[0, 0].set_ylabel("u(t)")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t_sriw1, u_sriw1, linewidth=0.8, color="darkorange")
axes[0, 1].set_title("SRIW1 (adaptive) — full path", fontweight="bold")
axes[0, 1].set_xlabel("t")
axes[0, 1].set_ylabel("u(t)")
axes[0, 1].grid(True, alpha=0.3)

# zoom into first 0.01 seconds
mask_em = t_em <= 0.01
mask_sriw1 = t_sriw1 <= 0.01

axes[1, 0].plot(
    t_em[mask_em],
    u_em[mask_em],
    linewidth=0.8,
    color="steelblue",
    marker="o",
    markersize=2,
)
axes[1, 0].set_title("EM (fixed step) — zoomed [0, 0.01]", fontweight="bold")
axes[1, 0].set_xlabel("t")
axes[1, 0].set_ylabel("u(t)")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(
    t_sriw1[mask_sriw1],
    u_sriw1[mask_sriw1],
    linewidth=0.8,
    color="darkorange",
    marker="o",
    markersize=2,
)
axes[1, 1].set_title("SRIW1 (adaptive) — zoomed [0, 0.01]", fontweight="bold")
axes[1, 1].set_xlabel("t")
axes[1, 1].set_ylabel("u(t)")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(
    "Pure Brownian motion: path comparison EM vs SRIW1 adaptive", fontweight="bold"
)
plt.tight_layout()
plt.savefig("worthwhileImages/path_comparison.pdf", dpi=150, bbox_inches="tight")
plt.show()

# print increment statistics for both
inc_em = np.diff(u_em)
inc_sriw1 = np.diff(u_sriw1)
print(f"\nEM increment variance: {np.var(inc_em):.6f} (expected {0.0001})")
print(f"SRIW1 increment variance: {np.var(inc_sriw1):.6f} (expected {0.0001})")
print(f"\nEM path range: [{u_em.min():.4f}, {u_em.max():.4f}]")
print(f"SRIW1 path range: [{u_sriw1.min():.4f}, {u_sriw1.max():.4f}]")


print(np.diff(sol_sriw1_dense.t)[:50])

print("min dt =", np.min(np.diff(sol_sriw1_dense.t)))
print("max dt =", np.max(np.diff(sol_sriw1_dense.t)))
print("mean dt =", np.mean(np.diff(sol_sriw1_dense.t)))
