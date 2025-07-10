import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set parameters
T = 1        # Total time
n = 1000     # Number of time steps per sample path
dt = T / n   # Time step size
num_paths = 100000  # Number of sample paths

# Set seed for reproducibility
np.random.seed(123)

# Preallocate array to store max absolute values of Wiener process paths
max_values = np.zeros(num_paths)

# Simulate all paths
for i in range(num_paths):
    Z = np.random.normal(0, 1, n)           # Generate i.i.d N(0, 1)
    W = np.cumsum(Z) * np.sqrt(dt)          # Wiener process approximation
    W = np.insert(W, 0, 0)                  # Add W(0) = 0
    max_values[i] = np.max(np.abs(W))       # Max absolute value of path

# Sort values for further analysis
sorted_max_values = np.sort(max_values)

# Plot histogram and density
plt.figure(figsize=(10, 6))
sns.histplot(sorted_max_values, bins=50, stat='density', color='lightblue', edgecolor='black', kde=True)
plt.title("Histogram of Maximum Absolute Values of W(r)")
plt.xlabel("Max |W(r)|")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute critical value at 95th percentile (5% significance level)
critical_value = np.quantile(sorted_max_values, 0.95)
print(f"Critical value at 5% level (95th percentile): {critical_value:.5f}")
