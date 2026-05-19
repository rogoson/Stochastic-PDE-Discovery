from diffeqpy import de
from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt
import os

# also add a seed here

# parameters
TOTAL_SAMPLES = 2000
epsilon = 1
sigma = 1
N = 64  # number of spatial points (periodic, so points = intervals)
L = 20
dx = L / N  # h = a/J = 20/64
x = np.linspace(0, L - dx, N)  # 64 points from 0 to 20, 64 intervals
endTime = 1.0
timesteps = 500
dt = endTime / timesteps
timeSpan = (0.0, endTime)
u0 = list(np.sin(x))
p = (epsilon, dx, sigma)

# normal heat update, but copying boundary conditions from matlab code (2x thing next to start/end)
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
""")
# continuous (not scaling the noise by /sqrt(dx) to spread noise out - which would end with a higher estimate overall actually)

baseProblem = de.SDEProblem(jl.heat_drift_jl, jl.heat_noise_jl, u0, timeSpan, p)
ensembleProblem = de.EnsembleProblem(baseProblem)

solution = de.solve(  # sr1w1 messed up diffusion estimates
    ensembleProblem,
    de.EM(),
    de.EnsembleThreads(),
    trajectories=TOTAL_SAMPLES,
    saveat=dt,
    dt=dt,
)
# SOMETHING TO NOTE IN METHODOLOGY: it is that SR1w1 actually gives smoother and more accurate trajectories, such that the variance is actually much smaller and therefore messes up the estimation
# basically, it is that y_t+1 - y_t is therefore smaller by sr1w1 on average because its a smoother trajectory even though dt is the same

# shape, matching Mathpati code, but not paper - minimal difference
uStorage = np.zeros((N, timesteps + 1, TOTAL_SAMPLES))
for sampleNo in range(TOTAL_SAMPLES):
    if sampleNo % 100 == 0:
        print("Sample number: ", sampleNo)
    U = np.array([np.array(u) for u in solution.u[sampleNo].u])
    uStorage[:, :, sampleNo] = U.T

higherDirectory = os.path.dirname(os.path.dirname(__file__))

# consecutive differences: u(t+1) - u(t), shape (64, 500, 2000)
y = uStorage[:, 1:, :] - uStorage[:, :-1, :]

# first KM moment (drift estimate), shape (64, 500)
xdt = (1 / dt) * np.mean(y, axis=2)

# second KM moment (diffusion squared estimate), shape (64, 500)
xdiff = (1 / dt) * np.mean(y * y, axis=2)

np.save(os.path.join(higherDirectory, "Heat_dx_64m_500t.npy"), uStorage)
np.save(os.path.join(higherDirectory, "Heat_xdt_64m_500t.npy"), xdt)
np.save(os.path.join(higherDirectory, "Heat_xdiff_64m_500t.npy"), xdiff)

print("uStorage shape:", uStorage.shape)  # (64, 501, 2000)
print("xdt shape:", xdt.shape)  # (64, 500)
print("xdiff shape:", xdiff.shape)  # (64, 500)

# print("mean y^2 at interior (rows 1:-1):", np.mean(y[1:-1] * y[1:-1]))
# print("mean y^2 at boundary row 0:", np.mean(y[0] * y[0]))
# print("mean y^2 at boundary row 64:", np.mean(y[64] * y[64]))
# print("expected mean y^2:", sigma**2 / dx * dt)
