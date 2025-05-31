'''
🧠 What is PCA?

PCA (Principal Component Analysis) is a technique used to:

    Reduce the number of features (dimensions) in a dataset

    While preserving as much variance (information) as possible

It transforms your data into a new coordinate system where:

    The first axis (principal component) captures the direction of maximum variance

    The second axis captures the next highest variance, orthogonal (at right angles) to the first

    And so on…
🔍 Why Use PCA?
Problem	How PCA Helps
Too many features (curse of dimensionality)	Reduces dimensions without losing much information
Features are correlated	Removes redundancy by creating uncorrelated components
Improves model performance	Helps with speed, generalization, and visualization

🧮 How PCA Works (Conceptually)
    Standardize the data  - Subtract the mean and scale to unit variance (important!)
    Compute the covariance matrix - This captures relationships between features

    Compute the eigenvectors and eigenvalues
        Eigenvectors → directions (principal components)
        Eigenvalues → amount of variance in each direction

    Select top k components
        Based on the eigenvalues

    Project data onto new axes
        You now have reduced-dimension data
    
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# fake dataset: 100 samples, 5 features
np.random.seed(0)
X = np.random.rand(100,5)

# standardize features
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)

# Plot the 2D projection
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")
plt.grid(True)
plt.show()