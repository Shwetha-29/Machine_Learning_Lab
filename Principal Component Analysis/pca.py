# ======================
# PCA on Wine Quality Dataset
# ======================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load Dataset
df = pd.read_csv("Wine_Quality.csv")
print("Original Dataset:\n", df.head())

# Step 2: Standardize Data (exclude target 'quality')
features = df.columns[:-1]  # all columns except 'quality'
X = df[features]

mean = X.mean()
std = X.std()
Z = (X - mean) / std
print("\nStandardized Data (Manual):\n", Z.head())

# Step 3: Covariance Matrix
cov_matrix = np.cov(Z.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 4: Eigenvalues & Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 5: Sort Eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("\nSorted Eigenvalues:\n", eigenvalues)

# Step 6: Explained Variance Ratio
explained_variance = eigenvalues / np.sum(eigenvalues)
print("\nExplained Variance Ratio (Manual):\n", explained_variance)

# Step 7: Project onto first 2 PCs
PCs = eigenvectors[:, :2]
pca_manual = np.dot(Z, PCs)
print("\nPCA Result (Manual):\n", pca_manual[:5])  # first 5 rows

# Step 8: PCA using sklearn
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(Z_scaled)

print("\nExplained Variance Ratio (sklearn):", pca.explained_variance_ratio_)
print("\nPCA Result (sklearn):\n", pca_result[:5])  # first 5 rows