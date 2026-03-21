import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. The SWIR Model Definition
# y[0]=S, y[1]=W, y[2]=I, y[3]=R
def swir_model(y, t, Lambda, rho, alpha, beta, epsilon, mu, delta, kappa, sigma):
    S, W, I, R = y
    
    # Differential Equations
    dSdt = (1 - rho) * Lambda + alpha * R - (beta * I + epsilon + mu) * S
    dWdt = rho * Lambda + epsilon * S - ((1 - delta) * beta * I + mu) * W
    dIdt = beta * S * I + (1 - delta) * beta * W * I - (kappa + mu + sigma) * I
    dRdt = kappa * I - (alpha + mu) * R
    
    return [dSdt, dWdt, dIdt, dRdt]

# 2. Parameters

N0 = 3000000.0
Lambda = 10.0      # Recruitment
mu = 0.0001        # Natural death
alpha = 0.00001    # Relapse
rho = 0.05         # Initial smoker proportion
beta = 1.9e-6      # Transmission rate
epsilon = 0.08     # Initiation rate
kappa = 0.9        # Recovery rate
sigma = 0.15       # Disease mortality (Cancer deaths)

# 3. Initial Conditions 
# S0=2.68M, W0=310k, I0=950, R0=160
y0 = [2688890.0, 310000.0, 950.0, 160.0]

# 4. Time Span: 0 to 20 Years
t = np.linspace(0, 20, 2000)

# 5. Intervention Scenarios to Compare
deltas = [0.2, 0.5, 0.8]
colors = ['red', 'orange', 'purple', 'green']

plt.figure(figsize=(12, 7))

print("--- Peak Analysis Table ---")
print(f"{'Delta':<10} | {'Peak Time (yr)':<15} | {'Peak Infected':<15}")
print("-" * 45)

for delta, color in zip(deltas, colors):
    # Solve the ODE for the specific delta
    sol = odeint(swir_model, y0, t, args=(Lambda, rho, alpha, beta, epsilon, mu, delta, kappa, sigma))
    I_curve = sol[:, 2] # Extract only the Infected (Red) compartment
    
    # Plot the Infected curve
    plt.plot(t, I_curve, lw=3.5, label=f'Infected (delta = {delta})', color=color)
    
    # Find the peak point
    peak_idx = np.argmax(I_curve)
    peak_time = t[peak_idx]
    peak_val = I_curve[peak_idx]
    
    # Print the values to console
    print(f"{delta:<10} | {peak_time:<15.2f} | {int(peak_val):<15,}")
    
    # Annotate the peak on the graph
    plt.annotate(f'Peak: {int(peak_val):,}', 
                 xy=(peak_time, peak_val), 
                 xytext=(peak_time + 1.2, peak_val + 50000),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                 fontsize=10, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.7))

# 6. Formatting the Graph
plt.title('Impact of Effectiveness (delta) on the Infected Population', fontsize=14, fontweight='bold')
plt.xlabel('Years', fontsize=12)
plt.ylabel('Number of Infected Individuals', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', shadow=True, fontsize=11)
plt.xlim(0, 20)
plt.ylim(0, 2000000) # Adjust height for visibility
plt.tight_layout()

# 7. Final Output
plt.show()
