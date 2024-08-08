import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


""" The standard PCA in scikit-learn expects a 2D data matrix where each row represents a sample 
and each column represents a feature. In the context of FPCA, the "features" are values of the 
functions at discretized points.
"""

# Generate a larger dataset with 100 observations and 24 time points per observation
np.random.seed(42)  # For reproducibility
num_observations = 10
num_time_points = 24

# Create synthetic data: a sine wave with some noise
time = np.linspace(0, 2 * np.pi, num_time_points)
data = np.array([np.sin(time + np.random.uniform(0, 2 * np.pi)) for _ in range(num_observations)])

# Standardize the data (important for PCA)
data_mean = np.mean(data, axis=0)
data_centered = data - data_mean

# Apply PCA
pca = PCA(n_components=3)  # Choose the number of principal components
pca.fit(data_centered.T)

# Transform the data to the principal component space
principal_components = pca.transform(data_centered.T)

# The principal components
print("Principal Components (first 5 observations):")
print(principal_components)

# The explained variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Plot the original data and the first two principal components
plt.figure(figsize=(14, 6))

# Original data
plt.subplot(1, 2, 1)
for i in range(num_observations):
    plt.plot(time, data[i], alpha=0.5)
plt.title('Original Data')
plt.xlabel('Time Points')
plt.ylabel('Values')

# First two principal components
plt.subplot(1, 2, 2)
plt.plot(principal_components[:, 0], alpha=0.5)
plt.plot(principal_components[:, 1], alpha=0.5)
plt.plot(principal_components[:, 2], alpha=0.5)
plt.title('First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
