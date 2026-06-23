import matplotlib.pyplot as plt
from diffeqpy import de


# 1. Define the Drift (f) and Diffusion (g) equations
# Pure Brownian motion has 0 drift and 1 constant diffusion
def f(u, p, t):
    return 0.0


def g(u, p, t):
    return 1.0


# 2. Setup initial condition and time span
u0 = 0.0
tspan = (0.0, 1.0)

# 3. Formulate the Stochastic Differential Equation problem
prob = de.SDEProblem(f, g, u0, tspan)

# 4. Solve using the SRIW1 algorithm
# diffeqpy automatically makes SRIW1 available via the `de` namespace
sol = de.solve(prob, de.SRIW1())

# 5. Extract values and plot
plt.figure(figsize=(10, 4))
plt.plot(sol.t, sol.u, label="Pure Brownian Motion (SRIW1)", color="darkorange")
plt.xlabel("Time (t)")
plt.ylabel("Position (W_t)")
plt.grid(True)
plt.legend()
plt.show()
