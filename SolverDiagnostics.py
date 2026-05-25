import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
from juliacall import Main as jl
from diffeqpy import de

with open("./parameters.yaml") as f:
    yamlParameters = yaml.safe_load(f)

TOTAL_SAMPLES = yamlParameters["common"]["N_samples"]
N = yamlParameters["common"][
    "J"
]  # number of spatial points (periodic, so points = intervals)
L = yamlParameters["common"]["x_end"]
dx = L / N  # h = a/J = 20/64
x = np.linspace(0, L - dx, N)  # 64 points from 0 to 20, 64 intervals
endTime = yamlParameters["common"]["t_end"]
timesteps = yamlParameters["common"]["timesteps"]
dt = endTime / timesteps
timeSpan = (yamlParameters["common"]["t_start"], endTime)

initialConditions = {
    "heat": list(np.sin(x)),
    "allen_cahn": list(np.sin(x)),
    "nagumo": list(np.cos(x)),
}

jl.seval(f"""
using Random; Random.seed!(0)

function heat_drift_jl(du, u, p, t)
    epsilon, dx, sigma = p
    dx2 = dx^2
    du[1] = epsilon * (u[{N}] - 2u[1] + u[2]) / dx2
    for i in 2:{N-1}
        du[i] = epsilon * (u[i-1] - 2u[i] + u[i+1]) / dx2
    end
    du[{N}] = epsilon * (u[{N-1}] - 2u[{N}] + u[1]) / dx2
end

function allen_cahn_drift_jl(du, u, p, t)
    epsilon, dx, sigma, u_coeff, u3_coeff = p
    dx2 = dx^2
    du[1] = epsilon * (u[{N}] - 2u[1] + u[2]) / dx2 + u_coeff * u[1] + u3_coeff * u[1]^3
    for i in 2:{N-1}
        du[i] = epsilon * (u[i-1] - 2u[i] + u[i+1]) / dx2 + u_coeff * u[i] + u3_coeff * u[i]^3
    end
    du[{N}] = epsilon * (u[{N-1}] - 2u[{N}] + u[1]) / dx2 + u_coeff * u[{N}] + u3_coeff * u[{N}]^3
end

function nagumo_drift_jl(du, u, p, t)
    epsilon, dx, sigma, u_coeff, u2_coeff, u3_coeff = p
    dx2 = dx^2
    du[1] = epsilon * (u[{N}] - 2u[1] + u[2]) / dx2 + u_coeff * u[1] + u2_coeff * u[1]^2 + u3_coeff * u[1]^3
    for i in 2:{N-1}
        du[i] = epsilon * (u[i-1] - 2u[i] + u[i+1]) / dx2 + u_coeff * u[i] + u2_coeff * u[i]^2 + u3_coeff * u[i]^3
    end
    du[{N}] = epsilon * (u[{N-1}] - 2u[{N}] + u[1]) / dx2 + u_coeff * u[{N}] + u2_coeff * u[{N}]^2 + u3_coeff * u[{N}]^3
end

function oned_noise_jl(du, u, p, t)
    sigma = p[3]
    for i in 1:{N}
        du[i] = sigma 
    end
end
""")


def getParameters(method):
    epsilon = yamlParameters[method]["correct"]["epsilon"]
    sigma = yamlParameters[method]["correct"]["sigma"]
    dx = yamlParameters["common"]["dx"]
    if method == "allen_cahn":
        u_coeff = yamlParameters[method]["correct"]["u_coeff"]
        u3_coeff = yamlParameters[method]["correct"]["u3_coeff"]
        p = (epsilon, dx, sigma, u_coeff, u3_coeff)
    elif method == "nagumo":
        u_coeff = yamlParameters[method]["correct"]["u_coeff"]
        u2_coeff = yamlParameters[method]["correct"]["u2_coeff"]
        u3_coeff = yamlParameters[method]["correct"]["u3_coeff"]
        p = (epsilon, dx, sigma, u_coeff, u2_coeff, u3_coeff)
    elif method == "heat":
        p = (epsilon, dx, sigma)
    else:
        print("Unknown method: ", method)
        return
    return p


def convergence_study(method="heat", dt_multipliers=[1, 0.5, 0.25, 0.1, 0.05]):
    """
    Show KM diffusion estimate converges to true sigma^2 as dt -> 0
    for EM and SRIW1 fixed step.
    """
    DIAG_SAMPLES = 200
    p = getParameters(method)
    driftEquations = {
        "heat": jl.heat_drift_jl,
        "allen_cahn": jl.allen_cahn_drift_jl,
        "nagumo": jl.nagumo_drift_jl,
    }

    base_dt = yamlParameters["common"]["dt"]
    expected = yamlParameters[method]["correct"]["sigma"] ** 2

    solvers = {
        "EM": de.EM(),
        "SRIW1 (fixed)": de.SRIW1(),
    }

    results = {name: [] for name in solvers}
    dt_values = [base_dt * m for m in dt_multipliers]

    for dt_val in dt_values:
        # adjust saveat and timesteps to keep same total time
        saveat_val = dt_val

        baseProblem = de.SDEProblem(
            driftEquations[method],
            jl.oned_noise_jl,
            initialConditions[method],
            timeSpan,
            p,
        )
        ensembleProblem = de.EnsembleProblem(baseProblem)

        for name, solver in solvers.items():
            sol = de.solve(
                ensembleProblem,
                solver,
                de.EnsembleThreads(),
                trajectories=DIAG_SAMPLES,
                saveat=saveat_val,
                dt=dt_val,
                adaptive=False,
            )
            uStore = np.zeros((N, int(endTime / dt_val) + 1, DIAG_SAMPLES))
            for i in range(DIAG_SAMPLES):
                U = np.array([np.array(u) for u in sol.u[i].u])
                uStore[:, :, i] = U.T

            y = uStore[:, 1:, :] - uStore[:, :-1, :]
            xdiff_mean = (1 / dt_val) * np.mean(y * y)
            results[name].append(xdiff_mean)
            print(f"{name}, dt={dt_val:.4f}: KM estimate = {xdiff_mean:.4f}")

    # plot
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in results.items():
        ax.plot(dt_values, vals, marker="o", linewidth=2, label=name)
    ax.axhline(
        y=expected,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True σ²={expected}",
    )
    dataDir = os.path.join("worthwhileImages")
    os.makedirs(dataDir, exist_ok=True)
    ax.set_xlabel("dt", fontweight="bold")
    ax.set_ylabel("KM diffusion estimate", fontweight="bold")
    ax.set_title(f"Convergence of KM estimate — {method}", fontweight="bold")
    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.invert_xaxis()  # coarse -> fine left to right
    plt.tight_layout()
    plt.savefig(
        f"worthwhileImages/convergence_{method}.pdf", dpi=150, bbox_inches="tight"
    )
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    return results


convergence_study("heat")
convergence_study("allen_cahn")
convergence_study("nagumo")
