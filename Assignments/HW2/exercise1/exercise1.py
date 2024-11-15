import matplotlib.pyplot as plt
import numpy as np

    
k_B = 1.380*10**-23
T= 300
eta = 10**-3
R=10**-6
k_x = 10**-6
k_y = 9*10**-6

gamma = 6*np.pi*eta*R

D = (k_B*T)/gamma

##Q1
tau_trap_x = gamma/k_x
tau_trap_y = gamma/k_y

print("tau_trap_x: ", tau_trap_x,"tau_trap_y: ", tau_trap_y)

min_tau = np.min((tau_trap_x,tau_trap_y))
dt = np.round((min_tau/2),4)
print("dt=",dt)
t_tot = 30
N= int(t_tot/dt)



x = np.zeros(N)    
y = np.zeros(N)    
w_x=np.random.normal(0,1,N)  # Gaussian distributed random numbers 
w_y=np.random.normal(0,1,N)  # Gaussian distributed random numbers 
for i in range(N-1):
    x[i+1] = x[i] - k_x*x[i]*dt/gamma + np.sqrt(2*k_B*T*dt/gamma)*w_x[i]     
    y[i+1] = y[i] - k_y*y[i]*dt/gamma + np.sqrt(2*k_B*T*dt/gamma)*w_y[i]      


### P1
##scale to nm
plt.plot(x,y,'.',markersize=0.6)
plt.axis('equal')
plt.show()


#### P2
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for X distribution and theoretical distribution
axes[0].hist(x, bins=50, density=True, alpha=0.6, color='blue', label='X Distribution')
x_generated = np.linspace(np.min(x), np.max(x), 100)
U_x = 0.5 * k_x * x_generated**2
p_x = np.exp(-U_x / (k_B * T))
p_x /= np.trapz(p_x, x_generated)  # Normalize to match the histogram
axes[0].plot(x_generated, p_x, color='black', label='Expected X Distribution', linewidth=2)
axes[0].set_title('Probability Distributions of X, Theoretical and Calculated')
axes[0].legend()

# Plot for Y distribution and theoretical distribution
axes[1].hist(y, bins=50, density=True, alpha=0.6, color='red', label='Y Distribution')
y_generated = np.linspace(np.min(y), np.max(y), 100)
U_y = 0.5 * k_y * y_generated**2
p_y = np.exp(-U_y / (k_B * T))
p_y /= np.trapz(p_y, y_generated)  # Normalize to match the histogram
axes[1].plot(y_generated, p_y, color='black', label='Expected Y Distribution', linewidth=2)
axes[1].set_title('Probability Distributions of Y, Theoretical and Calculated')
axes[1].legend()

plt.tight_layout()
plt.show()


#### Q2
sigma_x = np.var(x)
sigma_y = np.var(y)

harmonic_trap_x = k_B*T/k_x
harmonic_trap_y = k_B*T/k_y

print("sigma_x: ",sigma_x, "Harmonic_trap_x: ", harmonic_trap_x)
print("sigma_y: ",sigma_y, "Harmonic_trap_y: ", harmonic_trap_y)
