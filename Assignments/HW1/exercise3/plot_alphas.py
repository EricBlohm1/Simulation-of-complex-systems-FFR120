import numpy as np
import matplotlib.pyplot as plt

# Provided data
alphas = np.array([1.30791642, 1.30735544, 1.26500305, 1.21988074, 1.17389297, 1.1470622, 1.12775834])
N_list = np.array([16, 32, 64, 128, 256, 512, 1024])

# Use only the last 5 values of alphas and N_list for the fit
alphas_last5 = alphas[-5:]
N_list_last5 = N_list[-5:]

# Calculate 1/N for the last 5 values
x_last5 = 1 / N_list_last5

# Perform linear fit using polyfit on the last 5 values
coefficients = np.polyfit(x_last5, alphas_last5, 1)  # 1 for linear fit
slope, intercept = coefficients

# Generate the fitted line for the last 5 points
fit_line_last5 = slope * x_last5 + intercept

# Print the y-intercept
print(f"The y-intercept of the linear fit for the last 5 points is: {intercept:.4f}")

# Plot the data points
plt.plot(1 / N_list, alphas, 'o', label='All Data points')
plt.plot(x_last5, alphas_last5, 'o', color='orange', label='Last 5 Data points')

# Plot the linear fit for the last 5 points as a dotted line
plt.plot(x_last5, fit_line_last5, 'r--', label=f'Linear fit (last 5 points, slope={slope:.2f})')

# Labels and title
plt.xlabel('1/N')
plt.ylabel('Alpha')
plt.title('Plot of 1/N vs. Alpha with Linear Fit (Last 5 Points)')
plt.legend()

# Show the plot
plt.show()
