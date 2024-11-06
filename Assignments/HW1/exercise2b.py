import numpy as np
import matplotlib.pyplot as plt

def neighboring_spins(i_list, j_list, sl):
    """
    Function returning the position of the neighbouring spins of a list of 
    spins identified by their positions in the spin lattice.
    
    Parameters
    ==========
    i_list : Spin position first indices.
    j_list : Spin position second indices.
    sl : Spin lattice.
    """

    Ni, Nj = sl.shape  # Shape of the spin lattice.
    
    # Position neighbors right.
    i_r = i_list  
    j_r = list(map(lambda x:(x + 1) % Nj, j_list))   

    # Position neighbors left.
    i_l = i_list  
    j_l = list(map(lambda x:(x - 1) % Nj, j_list))   

    # Position neighbors up.
    i_u = list(map(lambda x:(x - 1) % Ni, i_list))  
    j_u = j_list  

    # Position neighbors down.
    i_d = list(map(lambda x:(x + 1) % Ni, i_list)) 
    j_d = j_list   

    # Spin values.
    sl_u = sl[i_u, j_u]
    sl_d = sl[i_d, j_d]
    sl_l = sl[i_l, j_l]
    sl_r = sl[i_r, j_r]

    return sl_u, sl_d, sl_l, sl_r



def energies_spins(i_list, j_list, sl, H, J):
    """
    Function returning the energies of the states for the spins in given 
    positions in the spin lattice.
    
    Parameters
    ==========
    i_list : Spin position first indices.
    j_list : Spin position second indices.
    sl : Spin lattice.
    """
    
    sl_u, sl_d, sl_l, sl_r = neighboring_spins(i_list, j_list, sl)
    
    sl_s = sl_u + sl_d + sl_l + sl_r 
    
    E_u = - H - J * sl_s
    E_d =   H + J * sl_s 
    
    return E_u, E_d



def probabilities_spins(i_list, j_list, sl, H, J, T):
    """
    Function returning the energies of the states for the spins in given 
    positions in the spin lattice.
    
    Parameters
    ==========
    i_list : Spin position first indices.
    j_list : Spin position second indices.
    sl : Spin lattice.
    """
    
    E_u, E_d = energies_spins(i_list, j_list, sl, H, J)
    
    Ei = np.array([E_u, E_d])
    
    Z = np.sum(np.exp(- Ei / T), axis=0)  # Partition function.
    pi = 1 / np.array([Z, Z]) * np.exp(- Ei / T)  # Probability.

    return pi, Z       


import random
import time
from tkinter import *

f = 0.05  # Number of randomly selected spins to flip-test.
N_skip = 4999 #10  Visualize status every N_skip steps. 

window_size = 600

N = 100  # Size of the splin lattice.
#Temperatura critica ~2.269.
T_list = np.array([0.1, 0.2, 0.5, 1, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 5])  # External field.
J = 1  # Spin-spin coupling.


steps_max = 5000
m = np.zeros(len(T_list))
N = 100
for index, T in enumerate(T_list):
    #### Re-initialize for each H ####
    H = 0.1
    step = 0
    print(f"T={T}")
    sl = 2 * np.random.randint(2, size=(N, N)) - 1
    N_up = np.sum(sl + 1) / 2
    N_down = N * N - N_up
    print(f"Spin lattice created:  N_up={N_up}  N_down={N_down}")

    tk = Tk()
    tk.geometry(f'{window_size + 20}x{window_size + 20}')
    tk.configure(background='#000000')

    canvas = Canvas(tk, background='#ECECEC')  # Generate animation window.
    tk.attributes('-topmost', 0)
    canvas.place(x=10, y=10, height=window_size, width=window_size)

    Nspins = np.size(sl)  # Total number of spins in the spin lattice.
    Ni, Nj = sl.shape

    S = int(np.ceil(Nspins * f))  # Number of randomly selected spins.

    def stop_loop(event):
        global running
        running = False
    tk.bind("<Escape>", stop_loop)  # Bind the Escape key to stop the loop.
    running = True  # Flag to control the loop.
    #############################################################

    while step < steps_max and running:
        if(step > 300):
            H = 0
        ns = random.sample(range(Nspins), S)

        i_list = list(map(lambda x: x % Ni, ns)) 
        j_list = list(map(lambda x: x // Ni, ns)) 

        pi, Z = probabilities_spins(i_list, j_list, sl, H, J, T)

        rn = np.random.rand(S)
        for i in range(S):
            if rn[i] > pi[0, i]:
                sl[i_list[i], j_list[i]] = -1
            else:
                sl[i_list[i], j_list[i]] = 1

        # Update animation frame.
        if step % N_skip == 0:        
            canvas.delete('all')
            spins = []
            for i in range(Ni):
                for j in range(Nj):
                    spin_color = '#FFFFFF' if sl[i,j] == 1 else '#000000'
                    spins.append(
                        canvas.create_rectangle(
                            j / Nj * window_size, 
                            i / Ni * window_size,
                            (j + 1) / Nj * window_size, 
                            (i + 1) / Ni * window_size,
                            outline='', 
                            fill=spin_color,
                        )
                    )
            
            tk.title(f'Iteration {step}')
            tk.update_idletasks()
            tk.update()
            time.sleep(0.1)  # Increase to slow down the simulation.

        m[index] += (1/N**2) * np.sum(sl)
        step += 1
    m[index] = m[index]/steps_max
## Plot m(H)
plt.plot(T_list, m, marker='o', linestyle='-', color='b', label='m(T)')
plt.xlabel('T')
plt.ylabel('m')
plt.title('Plot of m(T)')
plt.legend()
plt.grid(True)
plt.show()


tk.update_idletasks()
tk.update()
tk.mainloop()  # Release animation handle (close window to finish).