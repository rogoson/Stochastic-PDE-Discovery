from diffeqpy import de
from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt
import os

# parameters
TOTAL_SAMPLES = 200
epsilonTrue = 1
sigmaTrue = 1

N = 64  # number of spatial points (periodic, so points = intervals)
L = 20
dx = L / N  # h = a/J = 20/64
x = np.linspace(0, L - dx, N)
endTime = 1.0
timesteps = 500
dt = endTime / timesteps
timeSpan = (0.0, endTime)
u0 = list(np.sin(x))
K = 10  # use 10 chebyschev terms
xCheb = 2 * x / L - 1
pTrue = (epsilonTrue, dx, sigmaTrue)


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

epsilonPred = 1.00367123
sigmaPred = 1.01054335
masterRng = np.random.RandomState(42)
epsArr = masterRng.normal(epsilonPred, np.sqrt(2.90836367e-05), TOTAL_SAMPLES)
sigArr = masterRng.normal(np.sqrt(sigmaPred), np.sqrt(3.25445242e-08), TOTAL_SAMPLES)

jl.Main.initialConditions = initialConditions
jl.Main.epsArr = epsArr.tolist()
jl.Main.sigArr = sigArr.tolist()
jl.Main.dxVal = dx

jl.seval(f"""
function heat_drift_jl(du, u, p, t)
    epsilon, dx, sigma = p
    dx2 = dx^2
    du[1] = epsilon * (u[{N}] - 2u[1] + u[2]) / dx2
    for i in 2:{N-1}
        du[i] = epsilon * (u[i-1] - 2u[i] + u[i+1]) / dx2
    end
    du[{N}] = epsilon * (u[{N-1}] - 2u[{N}] + u[1]) / dx2
end

function heat_noise_jl(du, u, p, t)
    epsilon, dx, sigma = p
    for i in 1:{N}
        du[i] = sigma 
    end
end

using SciMLBase
function prob_func_true(prob, context)
    i = context.sim_id
    u0_i = collect(Float64, Main.initialConditions[i])
    remake(prob, u0=u0_i)
end

function prob_func_pred(prob, context)
    i = context.sim_id
    u0_i = collect(Float64, Main.initialConditions[i])
    p_i = (Main.epsArr[i], Main.dxVal, Main.sigArr[i])
    remake(prob, u0=u0_i, p=p_i)
end
""")

u0_base = initialConditions[0]

baseProblemTrue = de.SDEProblem(
    jl.heat_drift_jl, jl.heat_noise_jl, u0_base, timeSpan, pTrue
)
ensembleProblemTrue = de.EnsembleProblem(baseProblemTrue, prob_func=jl.prob_func_true)

solution = de.solve(
    ensembleProblemTrue,
    de.EM(),
    de.EnsembleThreads(),
    trajectories=TOTAL_SAMPLES,
    saveat=dt,
    dt=dt,
)
uStorage = np.zeros((N, timesteps + 1, TOTAL_SAMPLES))
for sampleNo in range(TOTAL_SAMPLES):
    if sampleNo % 100 == 0:
        print("Sample number: ", sampleNo)
    U = np.array([np.array(u) for u in solution.u[sampleNo].u])
    uStorage[:, :, sampleNo] = U.T

higherDirectory = os.path.dirname(os.path.dirname(__file__))

np.save(os.path.join(higherDirectory, "Heat_prediction_sol_true.npy"), uStorage)
print("True solution shape:", uStorage.shape)

# predicted solution - sampled parameters
pPredBase = (epsilonPred, dx, sigmaPred)  # check this
baseProblemPred = de.SDEProblem(
    jl.heat_drift_jl, jl.heat_noise_jl, u0_base, timeSpan, pPredBase
)
ensembleProblemPred = de.EnsembleProblem(baseProblemPred, prob_func=jl.prob_func_pred)
solutionPred = de.solve(
    ensembleProblemPred,
    de.EM(),
    de.EnsembleThreads(),
    trajectories=TOTAL_SAMPLES,
    saveat=dt,
    dt=dt,
)

uStoragePred = np.zeros((N, timesteps + 1, TOTAL_SAMPLES))
for sampleNo in range(TOTAL_SAMPLES):
    U = np.array([np.array(u) for u in solutionPred.u[sampleNo].u])
    uStoragePred[:, :, sampleNo] = U.T

np.save(os.path.join(higherDirectory, "Heat_prediction_sol_pred.npy"), uStoragePred)
print("Pred solution shape:", uStoragePred.shape)
