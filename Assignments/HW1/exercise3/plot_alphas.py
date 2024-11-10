import numpy as np
import matplotlib.pyplot as plt


alphas = np.array([[1.31569453, 1.32402567, 1.31005177, 1.26665875, 1.31497457, 1.33124905,
  1.29578814, 1.3093294,  1.27141797, 1.28435779],
 [1.2835113,  1.29547278, 1.27710449, 1.30247106, 1.29839873, 1.2815834,
  1.31717347, 1.27610098, 1.3288826,  1.27264049],
 [1.25390093, 1.238716,   1.26877357, 1.24316284, 1.2389019,  1.28716023,
  1.2444757,  1.25674197, 1.27114427, 1.26494161],
 [1.20168756, 1.20822165, 1.20550901, 1.21761949, 1.22226098, 1.23089641,
  1.19427531, 1.22994184, 1.21968676, 1.23007718],
 [1.17364195, 1.17728415, 1.19129647, 1.1937755,  1.14922572, 1.17086706,
  1.1851752,  1.21244547, 1.1945893,  1.18623153],
 [1.15500203, 1.16826245, 1.14347467, 1.17837362, 1.18498173, 1.13170352,
  1.1600956,  1.13577061, 1.20062398, 1.11966529],
 [1.14035378, 1.12497595, 1.13829367, 1.14103446, 1.1404898,  1.10695644,
  1.1136765,  1.15666209, 1.11903796, 1.11171093]])

alpha_means = np.array([1.30235476, 1.29333393, 1.2567919,  1.21601762, 1.18345323, 1.15779535,
 1.12931916])
 
alpha_std_devs = np.array([0.02197956, 0.01876431, 0.01609277, 0.01298387, 0.01702144, 0.02569991,
 0.01630055])

N_list = np.array([16, 32, 64, 128, 256, 512, 1024])

#alpha_means = alpha_means[6:]
#alpha_std_devs = alpha_std_devs[-6:]
#N_list = N_list[-6:]

# alphas = [1.31513289 1.29298451 1.26087924 1.22198359 1.17718797 1.1361603 1.14262882]#
#alphas = np.array([1.31864883, 1.29383066, 1.25179234, 1.22168047, 1.18438247, 1.15611362, 1.13029531])
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