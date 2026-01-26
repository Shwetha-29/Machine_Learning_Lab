import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = {
    "Attendance": [85, 90, 78, 92, 88],
    "Internal_Marks": [70, 85, 65, 90, 80],
    "Assignment_Marks": [75, 88, 70, 92, 85],
    "Final_Exam": [72, 89, 68, 94, 83]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

mean = df.mean()
std = df.std()

Z = (df - mean) / std
print("\nStandardized Data (Manual):\n", Z)

cov_matrix = np.cov(Z.T)
print("\nCovariance Matrix:\n", cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nSorted Eigenvalues:\n", eigenvalues)

explained_variance = eigenvalues / np.sum(eigenvalues)
print("\nExplained Variance Ratio (Manual):\n", explained_variance)

PCs = eigenvectors[:, :2]

pca_manual = np.dot(Z, PCs)
print("\nPCA Result (Manual):\n", pca_manual)

scaler = StandardScaler()
Z_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(Z_scaled)

print("\nExplained Variance Ratio (sklearn):", pca.explained_variance_ratio_)
print("\nPCA Result (sklearn):\n", pca_result)
