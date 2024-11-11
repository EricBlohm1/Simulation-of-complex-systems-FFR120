import time
import numpy as np
import matplotlib.pyplot as plt

def neighbors_Moore(status):
    """
    Function to return the number of neighbors for each cell in status.
    
    Parameters
    ==========
    status : Current status.
    """

    # Initialize the neighbor count array
    n_nn = (
        np.roll(status, 1, axis=0) +  # Up.
        np.roll(status, -1, axis=0) +  # Down.
        np.roll(status, 1, axis=1) +  # Left.
        np.roll(status, -1, axis=1) +  # Right.
        np.roll(np.roll(status, 1, axis=0), 1, axis=1) +  # Up-Left.
        np.roll(np.roll(status, 1, axis=0), -1, axis=1) +  # Up-Right
        np.roll(np.roll(status, -1, axis=0), 1, axis=1) +  # Down-Left
        np.roll(np.roll(status, -1, axis=0), -1, axis=1)  # Down-Right
    )

    return n_nn


def apply_rule_2d(rule_2d, status):
    """
    Function to apply a 2-d rule on a status. Return the next status.
    
    Parameters
    ==========
    rule_2d : Array with size [2, 9]. Describe the CA rule.
    status : Current status.
    """
    
    Ni, Nj = status.shape  # Dimensions of 2-D lattice of the CA.
    next_status = np.zeros([Ni, Nj]) 
    
    # Find the number of neighbors.
    n_nn = neighbors_Moore(status) 
    for i in range(Ni):
        for j in range(Nj):
            next_status[i, j] = rule_2d[int(status[i, j]), int(n_nn[i, j])]
        
    return next_status

# Few commented structures for later use

'''
# Still life: the beehive.
gol[N // 2 - 3:N // 2 + 3, 
    N // 2 - 3:N // 2 + 3] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]

# Still life: the loaf.
gol[N // 2 - 3:N // 2 + 3, 
    N // 2 - 3:N // 2 + 3] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]

# Oscillator: the toad.
gol[N // 2 - 3:N // 2 + 3, 
    N // 2 - 3:N // 2 + 3] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]


# Oscillator: the beacon.
gol[N // 2 - 3:N // 2 + 3, 
    N // 2 - 3:N // 2 + 3] = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ]

# Glider.
gol[N // 2 - 3:N // 2 + 3, 
    N // 2 - 3:N // 2 + 3] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ]
'''



def n_state_change(gol_current, gol_previous):
    count = 0
    for i in range(0,len(gol_current)):
        for j in range(0,len(gol_current[0])):
            if gol_current[i][j] != gol_previous[i][j]:
                count +=1
    return count

# import time
from tkinter import *

N = 100

rule_2d = np.zeros([2, 9])

# Game of Life's rules.
rule_2d[0, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0]  # New born from empty cell.
rule_2d[1, :] = [0, 0, 1, 1, 0, 0, 0, 0, 0]  # Survival from living cell.
runs = 5
colors = ['blue', 'orange', 'green', 'red', 'purple']

steady_state = 200
step_max = 300
A = np.zeros((runs,step_max))
D = np.zeros(step_max)
C = np.zeros((runs,step_max-steady_state))
gol_check = np.zeros((step_max,N,N))




for run in range(0,runs):
    # Random initial state.
    gol = np.random.randint(2, size=[N, N])
    Ni, Nj = gol.shape  # Sets the variables describing the shape.
    N_skip = 299 # Visualize status every N_skip steps. 
    window_size = 600

    tk = Tk()
    tk.geometry(f'{window_size + 20}x{window_size + 20}')
    tk.configure(background='#000000')

    canvas = Canvas(tk, background='#ECECEC')  # Generate animation window.
    tk.attributes('-topmost', 0)
    canvas.place(x=10, y=10, height=window_size, width=window_size)

    step = 0

    def stop_loop(event):
        global running
        running = False
    tk.bind("<Escape>", stop_loop)  # Bind the Escape key to stop the loop.
    running = True  # Flag to control the loop.
    while step < step_max and running:

        gol = apply_rule_2d(rule_2d, gol)

        #keep track of gol's for all steps
        gol_check[step] = gol.copy()

        # Update animation frame.
        if step % N_skip == 0:        
            canvas.delete('all')
            gol_cells = []
            for i in range(Ni):
                for j in range(Nj):
                    gol_cell_color = '#FFFFFF' if gol[i, j] == 1 \
                    else '#000000' 
                    gol_cells.append(
                        canvas.create_rectangle(
                            j / Nj * window_size, 
                            i / Ni * window_size,
                            (j + 1) / Nj * window_size, 
                            (i + 1) / Ni * window_size,
                            outline='', 
                            fill=gol_cell_color,
                        )
                    )
            
            tk.title(f'Iteration {step}')
            tk.update_idletasks()
            tk.update()
            time.sleep(0.1)  # Increase to slow down the simulation.

        A[run][step] = np.sum(gol)
        D[step] += (1/N**2)*np.sum(gol)

        if step >= steady_state:
            count = n_state_change(gol_check[step],gol_check[step-1])
            C[run][step-steady_state] = count
        step += 1

## number of alive cells
plt.figure(figsize=(10, 6))
for run in range(0,runs):
    plt.plot(np.arange(step_max),A[run], color=colors[run], linestyle='-', label="A(t)")
    plt.xlabel("time step t")
    plt.ylabel("A(t)")
    plt.legend()
    plt.title("Number of alive cells over time")
    plt.grid(True)
plt.show()

## Density over 5 runs
D = D/runs
plt.figure(figsize=(10, 6))
plt.plot(np.arange(step_max),D, color='blue', linestyle='-', label="D(t)")
plt.xlabel("time step t")
plt.ylabel("D(t)")
plt.axvline(x=steady_state, color='black', linestyle=':', linewidth=2, label="Empirical steady state")
plt.legend()
plt.title("Average density of alive cell per unit area over 5 runs")
plt.grid(True)
plt.show()


#Number of cells that changes state
plt.figure(figsize=(10, 6))
for run in range(0,runs):
    plt.plot(np.arange(steady_state,step_max),C[run], color=colors[run], linestyle='-', label="C(t)")
    plt.xlabel("time step t")
    plt.ylabel("C(t)")
    plt.legend()
    plt.title("Number of cells that changes state at time t")
    plt.grid(True) 
plt.show()




tk.update_idletasks()
tk.update()
tk.mainloop()  # Release animation handle (close window to finish).