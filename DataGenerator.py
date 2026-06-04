from diffeqpy import de
from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import yaml
import os

np.random.seed(0)

# parameters

with open("parameters.yaml") as f:
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


def random_smooth_ic(n, m, sigma=3):
    u0 = np.random.randn(n, m)
    u0 = gaussian_filter(u0, sigma=sigma)
    return u0


initialConditions = {
    "heat": random_smooth_ic(len(x), 1).flatten(),
    "allen_cahn": random_smooth_ic(len(x), 1).flatten(),
    "nagumo": random_smooth_ic(len(x), 1).flatten(),
}

# normal heat update, but copying boundary conditions from matlab code (2x thing next to start/end)
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
# continuous (not scaling the noise by /sqrt(dx) to spread noise out - which would end with a higher estimate overall actually)


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


def generateData(method=None):
    if method is None:
        print("Please specify a system to generate data for.")
        return

    method = method.lower()
    p = getParameters(method)

    driftEquations = {
        "heat": jl.heat_drift_jl,
        "allen_cahn": jl.allen_cahn_drift_jl,
        "nagumo": jl.nagumo_drift_jl,
    }
    baseProblem = de.SDEProblem(
        driftEquations[method],
        jl.oned_noise_jl,
        initialConditions.get(method, [0.0] * N),
        timeSpan,
        p,
    )
    ensembleProblem = de.EnsembleProblem(baseProblem)
    solution = de.solve(  # sr1w1 messed up diffusion estimates
        ensembleProblem,
        de.SRIW1(),  # try with adaptive = false!!!!!
        de.EnsembleThreads(),
        trajectories=TOTAL_SAMPLES,
        adaptive=False,
        saveat=dt,
        dt=dt,
    )
    # check in more detail how SRIW1 improves on EM

    # shape, matching Mathpati code, but not paper - minimal difference
    uStorage = np.zeros((N, timesteps + 1, TOTAL_SAMPLES))
    for sampleNo in range(TOTAL_SAMPLES):
        if sampleNo % 100 == 0:
            print("Sample number: ", sampleNo)
        U = np.array([np.array(u) for u in solution.u[sampleNo].u])
        uStorage[:, :, sampleNo] = U.T

    # consecutive differences: u(t+1) - u(t), shape (64, 500, 2000)
    y = uStorage[:, 1:, :] - uStorage[:, :-1, :]

    # first KM moment (drift estimate), shape (64, 500)
    xdt = (1 / dt) * np.mean(y, axis=2)

    # second KM moment (diffusion squared estimate), shape (64, 500)
    xdiff = (1 / dt) * np.mean(y * y, axis=2)

    # save in data/ directory, accounting for if the dir doesnt exist. create data/ if it doesn't exist
    dataDir = os.path.join("data")
    os.makedirs(dataDir, exist_ok=True)

    np.save(
        os.path.join(dataDir, f"{method.capitalize()}_dx_{N}m_{timesteps}t.npy"),
        uStorage,
    )
    np.save(
        os.path.join(dataDir, f"{method.capitalize()}_xdt_{N}m_{timesteps}t.npy"),
        xdt,
    )
    np.save(
        os.path.join(dataDir, f"{method.capitalize()}_xdiff_{N}m_{timesteps}t.npy"),
        xdiff,
    )

    print("uStorage shape:", uStorage.shape)  # (64, 501, 2000)
    print("xdt shape:", xdt.shape)  # (64, 500)
    print("xdiff shape:", xdiff.shape)  # (64, 500)
