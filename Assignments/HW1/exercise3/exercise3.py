import numpy as np 
from matplotlib import pyplot as plt
    
def grow_trees(forest, p):
    """
    Function to pgrow new trees in the forest.
    
    Parameters
    ==========
    forest : 2-dimensional array.
    p : Probability for a tree to be generated in an empty cell.
    """
    
    Ni, Nj = forest.shape  # Dimensions of the forest.
    
    new_trees = np.random.rand(Ni, Nj)

    new_trees_indices = np.where(new_trees <= p)
    forest[new_trees_indices] = 1
    
    return forest


def propagate_fire(forest, i0, j0):
    """
    Function to propagate the fire on a populated forest.
    
    Parameters
    ==========
    forest : 2-dimensional array.
    i0 : First index of the cell where the fire occurs.
    j0 : Second index of the cell where the fire occurs.
    """
    
    Ni, Nj = forest.shape  # Dimensions of the forest.

    fs = 0  # Initialize fire size.

    if forest[i0, j0] == 1:
        active_i = [i0]  # Initialize the list.
        active_j = [j0]  # Tnitialize the list. 
        forest[i0, j0] = -1  # Sets the tree on fire.
        fs += 1  # Update fire size.
        
        while len(active_i) > 0:
            next_i = []
            next_j = []
            for n in np.arange(len(active_i)):
                # Coordinates of cell up.
                i = (active_i[n] + 1) % Ni
                j = active_j[n]
                # Check status
                if forest[i, j] == 1:
                    next_i.append(i)  # Add to list.
                    next_j.append(j)  # Add to list.
                    forest[i, j] = -1  # Sets the current tree on fire.
                    fs += 1  # Update fire size.

                # Coordinates of cell down.
                i = (active_i[n] - 1) % Ni
                j = active_j[n]
                # Check status
                if forest[i, j] == 1:
                    next_i.append(i)  # Add to list.
                    next_j.append(j)  # Add to list.
                    forest[i, j] = -1  # Sets the current tree on fire.
                    fs += 1  # Update fire size.

                # Coordinates of cell left.
                i = active_i[n]
                j = (active_j[n] - 1) % Nj
                # Check status
                if forest[i, j] == 1:
                    next_i.append(i)  # Add to list.
                    next_j.append(j)  # Add to list.
                    forest[i, j] = -1  # Sets the current tree on fire.
                    fs += 1  # Update fire size.

                # Coordinates of cell right.
                i = active_i[n]
                j = (active_j[n] + 1) % Nj
                # Check status
                if forest[i, j] == 1:
                    next_i.append(i)  # Add to list.
                    next_j.append(j)  # Add to list.
                    forest[i, j] = -1  # Sets the current tree on fire.
                    fs += 1  # Update fire size.

            active_i = next_i
            active_j = next_j        
            
    return fs, forest

def complementary_CDF(f, f_max):
    """
    Function to return the complementary cumulative distribution function.
    
    Parameters
    ==========
    f : Sequence of values (as they occur, non necessarily sorted).
    f_max : Integer. Maximum possible value for the values in f. 
    """
    
    num_events = len(f)
    s = np.sort(np.array(f)) / f_max  # Sort f in ascending order.
    c = np.array(np.arange(num_events, 0, -1)) / (num_events)  # Descending.
    
    c_CDF = c
    s_rel = s

    return c_CDF, s_rel

### Synthetic ###
def powerlaw_random(alpha, x_min, num_drawings):
    """
    Function that returns numbers drawn from a probability distribution
    P(x) ~ x ** (- alpha) starting from random numbers in [0, 1].
    
    Parameters
    ==========
    alpha : Exponent of the probability distribution. Must be > 1.
    x_min : Minimum value of the domain of P(x).
    num_drawings : Integer. Numbers of random numbers generated. 
    """
    
    if alpha <= 1:
        raise ValueError('alpha must be > 1')

    if x_min <= 0:
        raise ValueError('x_min must be > 0')

            
    r = np.random.rand(num_drawings)
    
    random_values = x_min * r ** (1 / (1 - alpha))

    return random_values


###### INIT SYSTEM AND LOOP######
N_list = np.array([16, 32, 64, 128, 256, 512, 1024])  # Size of the forrest
#_list = np.array([16, 32, 64, 128, 256])
iterations = 10
alphas = np.zeros((len(N_list),iterations))

for idx,N in enumerate(N_list):
    print(f"______\nN={N}")
    #if(N == 512):
        #iterations = 8
    #if(N== 1024):
        #iterations = 6

    #avgs = []

    for iter in range(0,iterations):
        p = 0.01  # Growth probability.
        f = 0.2  # Lightning strike probability.
        target_num_fires = 300  
        num_fires = 0
        forest = np.zeros([N, N])  # Empty forest.
        Ni, Nj = forest.shape  # Sets the variables describing the shape.
        fire_size = []  # Empty list of fire sizes.
        fire_history = []  # Empty list of fire history.
        while num_fires < target_num_fires:
            forest = grow_trees(forest, p)  # Grow new trees.
            
            p_lightning = np.random.rand()
            if p_lightning < f:  # Lightning occurs.
                i0 = np.random.randint(Ni)
                j0 = np.random.randint(Nj)
                
                fs, forest = propagate_fire(forest, i0, j0)
                if fs > 0:
                    fire_size.append(fs) 
                    num_fires += 1 
                    
                fire_history.append(fs)
                
            else:
                fire_history.append(0)

            forest[np.where(forest == -1)] = 0

        print(f'Target of {target_num_fires} fire events reached')


        c_CDF, s_rel = complementary_CDF(fire_size, forest.size)

        # Note loglog plot!
        """plt.loglog(s_rel, c_CDF, ".-", color='k', markersize=5, linewidth=0.5)

        plt.title('Empirical cCDF')

        plt.xlabel('relative size')
        plt.ylabel('c CDF')

        plt.show()"""

        #### cCDF and power law trends #####

        ### Empirical ###
        min_rel_size = 1e-3
        max_rel_size = 1e-1


        is_min = np.searchsorted(s_rel, min_rel_size)
        is_max = np.searchsorted(s_rel, max_rel_size)

        # Note!!! The linear dependence is between the logarithms
        p = np.polyfit(np.log(s_rel[is_min:is_max]),
                    np.log(c_CDF[is_min:is_max]), 1)

        beta = p[0]
        print(f'The empirical cCDF has an exponent beta = {beta:.4}')

        alpha = 1 - beta

        print(f'The empirical prob. distr. exponent: -alpha')
        print(f'with alpha = {alpha:.4}')

        alphas[idx][iter] = alpha
        #avgs.append(alpha)
        ############
    #alphas[idx] = np.sum(avgs) / iterations
    #stds[idx] = np.std(avgs)


print(alphas)
#print(stds)

"""plt.plot(1/N_list, alphas, 'o', label='Data points')
plt.xlabel('1/N')
plt.ylabel('Alpha')
plt.title('Plot of 1/N vs. Alpha')
plt.legend()
plt.show()


# Perform linear fit using polyfit on the last 5 values
coefficients = np.polyfit(1/N_list, alphas, 1)  # 1 for linear fit
slope, intercept = coefficients
line = slope * (1/N_list) + intercept

# Print the y-intercept
print(f"The y-intercept of the linear fit is: {intercept:.4f}")

plt.plot(1 / N_list, alphas, 'o', label='Data points')
plt.plot((1/N_list), line, 'r--', label=f'Linear fit (slope={slope:.2f},intercept={intercept:.2f})')

# Labels and title
plt.xlabel('1/N')
plt.ylabel('Alpha')
plt.title('Plot of 1/N vs. Alpha with Linear Fit')
plt.legend()
plt.show()"""

# Calculate the mean and standard deviation for each N
alpha_means = np.mean(alphas, axis=1)   # Mean across the iterations for each N


alpha_std_devs = np.std(alphas, axis=1, ddof=1)  # Standard deviation for error bars
print("means:",alpha_means,"stds: ", alpha_std_devs)
inv_N_values = [1 / N for N in N_list]

fit_coefficients = np.polyfit(inv_N_values, alpha_means, 1)
slope, intercept = fit_coefficients

# Generate linear fit line
fit_line = np.polyval(fit_coefficients, inv_N_values)

# Plot mean alpha points with error bars
plt.errorbar(inv_N_values, alpha_means, yerr=alpha_std_devs, fmt='o', color='blue', label='Mean alpha +- Std Dev')
# Plot the linear fit
plt.plot(inv_N_values, fit_line, 'k--', label=f'Linear Fit: alpha_inf = {intercept:.3f}')

# Add labels and title
plt.xlabel(r'$1/N$')
plt.ylabel(r'$\alpha$')
plt.title('Dependence of alpha on 1/N with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()



