import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------
# Simulation Settings
# ----------------------------
years = 50
t_span = (0, years)
t_eval = np.linspace(0, years, 1000)

# ----------------------------
# Initial Population Values
# ----------------------------
N0 = 3.0e6
W0 = 3.1e5
I0 = 9.5e2
R0 = 1.6e2
S0 = N0 - W0 - I0 - R0

y0 = [S0, W0, I0, R0]

# ----------------------------
# Parameters
# ----------------------------
mu = 1.52e-2
Lambda = 4.56e4
rho = 0.12
epsilon = 1.5e-2
beta = 5.0e-7       # Calibrated from KNCR
kappa = 0.04
alpha = 0.05
sigma = 0.23
delta = 0.2          # Intervention effectiveness

# ----------------------------
# SWIR Model
# ----------------------------
def swir_model(t, y, delta):
    S, W, I, R = y
    dS = (1 - rho)*Lambda + alpha*R - (beta*I + epsilon + mu)*S
    dW = rho*Lambda + epsilon*S - ((1 - delta)*beta*I + mu)*W
    dI = beta*S*I + (1 - delta)*beta*W*I - (kappa + mu + sigma)*I
    dR = kappa*I - (alpha + mu)*R
    return [dS, dW, dI, dR]

# ----------------------------
# Solve ODE
# ----------------------------
sol = solve_ivp(lambda t, y: swir_model(t, y, delta),
                t_span, y0, t_eval=t_eval)

# ----------------------------
# Plot 1: All Compartments
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label='S', linewidth=2)
plt.plot(sol.t, sol.y[1], label='W', linewidth=2)
plt.plot(sol.t, sol.y[2], label='I', linewidth=2)
plt.plot(sol.t, sol.y[3], label='R', linewidth=2)

plt.xlabel("Time (Years)")
plt.ylabel("Population")
plt.title("SWIR Lung Cancer Model Dynamics (δ=0.2)")
plt.legend()
plt.grid(True)

# Set uniform axis scales
plt.xlim(0,50)
plt.ylim(0, max(N0, max(sol.y[1]))*1.05)  # slightly above total pop
plt.savefig("fig2.png", dpi=300, bbox_inches='tight') 
plt.show()