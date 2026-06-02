from diffeqpy import de
from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

np.random.seed(0)

with open("parameters.yaml") as f:
    yamlParameters = yaml.safe_load(f)

TOTAL_SAMPLES = yamlParameters["common"]["prediction_samples"]  # prediction ensembles
N = yamlParameters["common"]["J"]
L = yamlParameters["common"]["x_end"]
dx = L / N
x = np.linspace(0, L - dx, N)
endTime = yamlParameters["common"]["t_end"]
timesteps = yamlParameters["common"]["timesteps"]
dt = endTime / timesteps
timeSpan = (yamlParameters["common"]["t_start"], endTime)

K = 10  # use 10 chebyschev terms
xCheb = 2 * x / L - 1

initialConditions = {
    "heat": list(
        np.sin(2 * np.pi * x / L)
    ),  # sin(0)=0, sin(20*2pi/20)=sin(2pi)=0 - fixing period
    "allen_cahn": list(np.sin(2 * np.pi * x / L)),
    "nagumo": list(np.cos(2 * np.pi * x / L)),  # cos(0)=1, cos(2pi)=1
}


def chebyschevInitial(xCheb, K, rng):  # mixture of chebyschev initials
    T = np.polynomial.chebyshev.chebvander(
        xCheb, K
    )  # evaluate all polynomials on all points
    w = rng.randn(K + 1) / (
        np.arange(K + 1) + 1
    )  # dampen higher freqencies to reduce noisiness of initials
    return T @ w


initialConditions = []
for i in range(TOTAL_SAMPLES):
    rng = np.random.RandomState(i)
    initialConditions.append(chebyschevInitial(xCheb, K, rng).tolist())

jl.Main.initialConditions = initialConditions
jl.Main.dxVal = dx

jl.seval(f"""
using SciMLBase, Random

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

function prob_func_True(prob, context)
    i = context.sim_id
    u0_i = collect(Float64, Main.initialConditions[i])
    remake(prob, u0=u0_i, seed=i)
end

function prob_func_pred(prob, context)
    i = context.sim_id
    u0_i = collect(Float64, Main.initialConditions[i])
    p_i = Tuple(Main.predParams[i])
    remake(prob, u0=u0_i, p=p_i, seed=i)  # same seed i as True
end
""")


def getParameters(method):
    epsilon = yamlParameters[method]["correct"]["epsilon"]
    sigma = yamlParameters[method]["correct"]["sigma"]
    if method == "allen_cahn":
        u_coeff = yamlParameters[method]["correct"]["u_coeff"]
        u3_coeff = yamlParameters[method]["correct"]["u3_coeff"]
        return (epsilon, dx, sigma, u_coeff, u3_coeff)
    elif method == "nagumo":
        u_coeff = yamlParameters[method]["correct"]["u_coeff"]
        u2_coeff = yamlParameters[method]["correct"]["u2_coeff"]
        u3_coeff = yamlParameters[method]["correct"]["u3_coeff"]
        return (epsilon, dx, sigma, u_coeff, u2_coeff, u3_coeff)
    elif method == "heat":
        return (epsilon, dx, sigma)
    else:
        print("Unknown method: ", method)
        return None


def samplePredParameters(method, nSamples):
    # sample predicted parameters from VB posterior for each trajectory
    with open("parameters.yaml") as f:
        yamlParameters = yaml.safe_load(f)
    disc = yamlParameters[method]["discovered"]
    masterRng = np.random.RandomState(42)
    epsilon = masterRng.normal(
        disc["epsilon"], np.sqrt(disc["epsilon_variance"]), nSamples
    )

    # override dx for allen cahn, 400hz used instead of 500 in the paper

    if any(
        param is None for param in disc.values()
    ):  # wrong but probably the most graceful way to handle this - can't generate new julia code on the fly for a broken model
        print(
            f"Warning: no discovered parameters for {method}, using True parameters for predictions."
        )
        return [getParameters(method) for _ in range(nSamples)]

    if method == "heat":
        sigma = np.sqrt(
            np.abs(
                masterRng.normal(
                    disc["sigma_squared"],
                    np.sqrt(disc["sigma_squared_variance"]),
                    nSamples,
                )
            )
        )
        return [(epsilon[i], dx, sigma[i]) for i in range(nSamples)]
    elif method == "allen_cahn":
        sigma = np.sqrt(
            np.abs(
                masterRng.normal(
                    disc["sigma_squared"],
                    np.sqrt(disc["sigma_squared_variance"]),
                    nSamples,
                )
            )
        )
        u_coeff = masterRng.normal(
            disc["u_coeff"], np.sqrt(disc["u_coeff_variance"]), nSamples
        )
        u3_coeff = masterRng.normal(
            disc["u3_coeff"], np.sqrt(disc["u3_coeff_variance"]), nSamples
        )
        return [
            (epsilon[i], dx, sigma[i], u_coeff[i], u3_coeff[i]) for i in range(nSamples)
        ]
    elif method == "nagumo":
        sigma = np.sqrt(
            np.abs(
                masterRng.normal(
                    disc["sigma_squared"],
                    np.sqrt(disc["sigma_squared_variance"]),
                    nSamples,
                )
            )
        )  # because it discovers the squared form
        u_coeff = masterRng.normal(
            disc["u_coeff"], np.sqrt(disc["u_coeff_variance"]), nSamples
        )
        u2_coeff = masterRng.normal(
            disc["u2_coeff"], np.sqrt(disc["u2_coeff_variance"]), nSamples
        )
        u3_coeff = masterRng.normal(
            disc["u3_coeff"], np.sqrt(disc["u3_coeff_variance"]), nSamples
        )
        return [
            (epsilon[i], dx, sigma[i], u_coeff[i], u2_coeff[i], u3_coeff[i])
            for i in range(nSamples)
        ]


def generatePrediction(method=None):
    if method is None:
        print("Please specify a system.")
        return

    method = method.lower()
    driftEquations = {
        "heat": jl.heat_drift_jl,
        "allen_cahn": jl.allen_cahn_drift_jl,
        "nagumo": jl.nagumo_drift_jl,
    }

    pTrue = getParameters(method)
    predParams = samplePredParameters(method, TOTAL_SAMPLES)
    jl.Main.predParams = predParams

    dataDir = os.path.join("data")
    os.makedirs(dataDir, exist_ok=True)

    for label, probFunc in [
        ("correct", jl.prob_func_True),
        ("pred", jl.prob_func_pred),
    ]:
        baseProblem = de.SDEProblem(
            driftEquations[method],
            jl.oned_noise_jl,
            initialConditions[0],  # overridden by prob_func
            timeSpan,
            pTrue,
        )
        ensembleProblem = de.EnsembleProblem(baseProblem, prob_func=probFunc)
        solution = de.solve(
            ensembleProblem,
            de.EM(),
            de.EnsembleSerial(),  # serial for reproducibility with seeded noise
            trajectories=TOTAL_SAMPLES,
            saveat=dt,
            dt=dt,
        )

        uStorage = np.zeros((N, timesteps + 1, TOTAL_SAMPLES))
        for sampleNo in range(TOTAL_SAMPLES):
            if sampleNo % 50 == 0:
                print(f"{label} sample: {sampleNo}")
            U = np.array([np.array(u) for u in solution.u[sampleNo].u])
            uStorage[:, :, sampleNo] = U.T

        np.save(
            os.path.join(dataDir, f"{method}_prediction_{label}.npy"),
            uStorage,
        )
        print(f"{label} shape:", uStorage.shape)
