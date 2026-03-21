import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. The SWIR Model (System 2)
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
Lambda = 10.0      
mu = 0.0002        
alpha = 0.00001    
rho = 0.05         
beta = 1.9e-6      
epsilon = 0.08     # Initiation rate
kappa = 0.9        # Recovery rate
sigma = 0.15       # Disease mortality 
delta = 0.50       # Weak intervention (0% effective)

# 3. Initial Conditions
# S0 = 2,688,890, W0 = 310,000, I0 = 950, R0 = 160
y0 = [2688890.0, 310000.0, 950.0, 160.0]

# 4. Time Span: 0 to 20 Years
t = np.linspace(0, 20, 2000)

# 5. Solve the ODE
sol = odeint(swir_model, y0, t, args=(Lambda, rho, alpha, beta, epsilon, mu, delta, kappa, sigma))

# 6. Plotting
plt.figure(figsize=(12, 7))
plt.plot(t, sol[:, 0], color='blue', lw=3.5, label='Susceptible (S)')
plt.plot(t, sol[:, 1], color='black', lw=3.5, label='Waterpipe Smokers (W)')
plt.plot(t, sol[:, 2], color='red', lw=3.5, label='Infected (I)')
plt.plot(t, sol[:, 3], color='lime', lw=3.5, label='Recovered (R)')


plt.title(f'SWIR Model: Moderate Intervention (delta = {delta})', fontsize=14, fontweight='bold')
plt.xlabel('Years', fontsize=12)
plt.ylabel('Number of Individuals', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', shadow=True, fontsize=11)
plt.xlim(0, 20)
plt.ylim(0, 3500000)
plt.tight_layout()

# Display the plot
plt.show()

# 7. Print verification for t=12.5 and t=20
idx_12_5 = np.argmin(np.abs(t - 12.5))
idx_20 = np.argmin(np.abs(t - 20))
print(f"Values at t=12.5: S={sol[idx_12_5,0]:.2f}, R={sol[idx_12_5,3]:.2f}")
print(f"Values at t=20.0: S={sol[idx_20,0]:.2f}, R={sol[idx_20,3]:.2f}")
