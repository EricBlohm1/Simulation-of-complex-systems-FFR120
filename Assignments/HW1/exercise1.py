import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters for the Lennard-Jones gas.
m = 1  # Mass (units of m0).
sigma = 1  # Size (units of sigma0).
eps = 1  # Energy (unit of epsilon0).
v0 = 10  # Initial speed (units of v0 = sqrt((2 * epsilon0) / m0)).


## Parameters of disk
m_disk = 10
r_disk = 10


# Parameters for the simulation.
N_particles = 25**2  # Number of particles.

dt = 0.005   # Time step (units of t0 = sigma * sqrt(m0 /(2 * epsilon0))).

L = 260  # Box size (units of sigma0).
x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

cutoff_radius = 3 * sigma  # Cutoff_radius for neighbours list.


# Generate initial positions on a grid and orientations at random.
x0, y0 = np.meshgrid(
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
)
x0 = x0.flatten()[:N_particles]
y0 = y0.flatten()[:N_particles]
print(len(x0))

def remove_particles_in_disk(x0,y0,r_disk,cutoff):
    ## Disk start at (0,0) 
    distances = np.sqrt(x0**2+y0**2)
    indices = np.where(distances > r_disk + cutoff)[0]
    if 0 not in indices:
        indices = np.insert(indices, 0, 0)
    #len(indices) = N_particles
    return x0[indices], y0[indices], len(indices)

x0,y0, N_particles = remove_particles_in_disk(x0,y0,r_disk,cutoff_radius)
phi0 = (2 * np.random.rand(N_particles) - 1) * np.pi
print(len(x0))


##TODO only neighbours for the disk?? 
# Initialize the neighbour list.
def list_neighbours(x, y, N_particles, cutoff_radius):
    '''Prepare a neigbours list for each particle.'''
    neighbours = []
    neighbour_number = []
    for j in range(N_particles):
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)
        neighbor_indices = np.where(distances <= r_disk + cutoff_radius)
        neighbours.append(neighbor_indices)
        neighbour_number.append(len(neighbor_indices))
    return neighbours, neighbour_number

neighbours, neighbour_number = list_neighbours(x0, y0, N_particles, cutoff_radius)


# Initialize the variables for the leapfrog algorithm.
# Current time step.
x = x0
y = y0
x_half = np.zeros(N_particles)
y_half = np.zeros(N_particles)
v = v0
phi = phi0
vx = v0 * np.cos(phi0)
vy = v0 * np.sin(phi0)

# Next time step for particles.
nx = np.zeros(N_particles)
ny = np.zeros(N_particles)
nv = np.zeros(N_particles)
nphi = np.zeros(N_particles)
nvx = np.zeros(N_particles)
nvy = np.zeros(N_particles)


##TODO change to lennard jones
def total_force_cutoff(x, y, N_particles, sigma, epsilon, neighbours):
    '''
    Calculate the total force on each particle due to the interaction with a 
    neighbours list with the particles interacting through a Lennard-Jones 
    potential.
    '''
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)
    for i in range(N_particles):
        for j in list(neighbours[i][0]):
            if i != j:
                if(i == 0 or j == 0):
                    r2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                    r = np.sqrt(r2) - r_disk
                    frac = sigma / r
                    
                    # Force on i due to j.
                    F = 24 * epsilon / r * (2 * frac ** 12 - frac ** 6)  # Modulus.
                    
                    Fx[i] += F * (x[i] - x[j]) / r
                    Fy[i] += F * (y[i] - y[j]) / r
    return Fx, Fy




import time
from scipy.constants import Boltzmann as kB 
from tkinter import *

window_size = 600

tk = Tk()
tk.geometry(f'{window_size + 20}x{window_size + 20}')
tk.configure(background='#000000')

canvas = Canvas(tk, background='#ECECEC')  # Generate animation window 
tk.attributes('-topmost', 0)
canvas.place(x=10, y=10, height=window_size, width=window_size)


x[0] = 0
y[0] = 0
vx[0] = 0
vy[0] = 0

disk = canvas.create_oval(
    ( x[0]- r_disk) / L * window_size + window_size / 2, 
    ( y[0]- r_disk) / L * window_size + window_size / 2,
    ( x[0]+r_disk) / L * window_size + window_size / 2, 
    ( y[0]+r_disk) / L * window_size + window_size / 2,
    outline='#000000', 
    fill='#000000',
)

particles = []
for j in range(1, N_particles):
    particles.append(
        canvas.create_oval(
            (x[j] - sigma / 2) / L * window_size + window_size / 2, 
            (y[j] - sigma / 2) / L * window_size + window_size / 2,
            (x[j] + sigma / 2) / L * window_size + window_size / 2, 
            (y[j] + sigma / 2) / L * window_size + window_size / 2,
            outline='#00C0C0', 
            fill='#00C0C0',
        )
    )

step = 0
T_tot = 400
t = 0
trajectory = []

def stop_loop(event):
    global running
    running = False
tk.bind("<Escape>", stop_loop)  # Bind the Escape key to stop the loop.
running = True  # Flag to control the loop.
while t < T_tot and running:
    x_half = x + 0.5 * vx * dt      
    y_half = y + 0.5 * vy * dt      

    fx, fy = \
        total_force_cutoff(x_half, y_half, N_particles, sigma, eps, neighbours)
    
    nvx = vx + fx / m * dt
    nvy = vy + fy / m * dt
    nvx[0] = vx[0] + fx[0] / m_disk * dt
    nvy[0] = vy[0] + fy[0] / m_disk * dt
        
    nx = x_half + 0.5 * nvx * dt
    ny = y_half + 0.5 * nvy * dt       
    
    # Reflecting boundary conditions.
    for j in range(N_particles):
        if nx[j] < x_min:
            nx[j] = x_min + (x_min - nx[j])
            nvx[j] = - nvx[j]

        if nx[j] > x_max:
            nx[j] = x_max - (nx[j] - x_max)
            nvx[j] = - nvx[j]

        if ny[j] < y_min:
            ny[j] = y_min + (y_min - ny[j])
            nvy[j] = - nvy[j]
            
        if ny[j] > y_max:
            ny[j] = y_max - (ny[j] - y_max)
            nvy[j] = - nvy[j]
    
    nv = np.sqrt(nvx ** 2 + nvy ** 2)
    for i in range(N_particles):
        nphi[i] = math.atan2(nvy[i], nvx[i])
    
    # Update neighbour list.
    if step % 10 == 0:
        neighbours, neighbour_number = \
            list_neighbours(nx, ny, N_particles, cutoff_radius)

    # Update variables for next iteration.
    x = nx
    y = ny
    vx = nvx
    vy = nvy
    v = nv
    phi = nphi

    # Update animation frame.
    if step % 100 == 0:        
        canvas.coords(
            disk,
            ( nx[0] - r_disk) / L * window_size + window_size / 2,
            ( ny[0] - r_disk) / L * window_size + window_size / 2,
            ( nx[0] + r_disk) / L * window_size + window_size / 2,
            ( ny[0] + r_disk) / L * window_size + window_size / 2,
        )
        for j, particle in enumerate(particles):
            canvas.coords(
                particle,
                (nx[j + 1] - sigma / 2) / L * window_size + window_size / 2,
                (ny[j + 1] - sigma / 2) / L * window_size + window_size / 2,
                (nx[j + 1] + sigma / 2) / L * window_size + window_size / 2,
                (ny[j + 1] + sigma / 2) / L * window_size + window_size / 2,
            )
                    
        tk.title(f'Time {step * dt:.1f} - Iteration {step}')
        tk.update_idletasks()
        tk.update()
        time.sleep(.001)  # Increase to slow down the simulation.    

    step += 1

    t+=dt
    trajectory.append((x[0],y[0]))



tk.update_idletasks()
tk.update()
tk.mainloop()  # Release animation handle (close window to finish).


################## P1 ################################
x_positions = [pos[0] for pos in trajectory]
y_positions = [pos[1] for pos in trajectory]

# Plotting the trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_positions, y_positions, marker='o', linestyle='-', color='b', markersize=2, label="Disk Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Trajectory of the Disk in 2D Space")
plt.legend()
plt.grid(True)
plt.axis('equal')  # Keeps the aspect ratio equal for x and y axes
plt.show()


### P2 ###
N = len(x_positions)
msd = np.zeros(N)
for n in range(0,N):
    print(n)
    frac = 1/(N-n)
    sum = 0
    for i in range(0, N-n):
        sum += (x_positions[i+n]-x_positions[i]) ** 2 + (y_positions[i+n]-y_positions[i])**2
    tmp = frac * sum 
    msd[n] = tmp


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(N), msd, marker='o', linestyle='-', color='b', label='MSD')

# Add labels and title
plt.title('Mean Squared Displacement (MSD) vs. Time Lag', fontsize=16)
plt.xlabel('Time Lag (n)', fontsize=14)
plt.ylabel('MSD', fontsize=14)

# Add a grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Show the plot
plt.show()

 