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
beta = 5.0e-7
kappa = 0.04
alpha = 0.05
sigma = 0.23

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
# Solve ODE for δ = 0, 0.2, 0.8
# ----------------------------
deltas = [0.0, 0.2, 0.8]
solutions = {}

for d in deltas:
    sol = solve_ivp(lambda t, y: swir_model(t, y, d),
                    t_span, y0, t_eval=t_eval)
    solutions[d] = sol

# ----------------------------
# Plot: Infected Population with Peak Markers
# ----------------------------
plt.figure(figsize=(10,6))
styles = ['-', '-', '-']
colors = ['r','g','b']

for d, style, color in zip(deltas, styles, colors):
    I_vals = solutions[d].y[2]
    t_vals = solutions[d].t
    plt.plot(t_vals, I_vals, linestyle=style, color=color, linewidth=2, label=f'I(t), δ={d}')
    
    # Peak marker
    peak_idx = np.argmax(I_vals)
    plt.plot(t_vals[peak_idx], I_vals[peak_idx], 'o', color=color, markersize=8)
    plt.text(t_vals[peak_idx]+0.5, I_vals[peak_idx]*1.01, f"{int(I_vals[peak_idx]):,}", color=color)

plt.xlabel("Time (Years)")
plt.ylabel("Infected Population")
plt.title("Impact of Intervention Effectiveness on Lung Cancer (I(t))")
plt.grid(True)
plt.xlim(0,50)
plt.ylim(0, max([max(solutions[d].y[2]) for d in deltas])*1.05)
plt.legend()
plt.savefig("fig4.png", dpi=300, bbox_inches='tight') 
plt.show()
