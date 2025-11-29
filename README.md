# Principal Component Analysis (PCA): Algorithms & Implementation

## Objective: Implement Principal Component Analysis (PCA) using SVD/eigenvalue methods and visualize high-dimensional data.

### Background: PCA finds orthogonal directions of maximum variance by diagonalizing the covariance matrix. Numerically, it is obtained via SVD of the centered data matrix.

#### Tasks:

- Take a dataset (Iris, MNIST, etc.).
- Standardize and center it.
- Compute covariance and perform SVD/eigen-decomposition.
- Project data onto top k components.
- Visualize variance explained and reduced-dimension scatter plot.

#### Deliverables:

- Notebook with scree plot, scatter plots, and report on conditioning and interpretation.
- Extensions: Compare SVD-based PCA with iterative PCA; apply to noise reduction.

#### References:

- I. T. Jolliffe, Principal Component Analysis, Springer, 2002.
- C. Bishop, Pattern Recognition and Machine Learning, Springer, 2006.

### 1. Covariance & Eigen-decomposition

- **Logic**: PCA is derived by computing the covariance matrix of standardized data, $C = \frac{1}{n-1} X^T X$.
- **Decomposition**: The eigenvectors of $C$ represent the principal components (directions of maximum variance), and the eigenvalues represent the magnitude of that variance.
- **Implementation**: We manually compute the covariance matrix and use `numpy.linalg.eigh` for decomposition.

### 2. Singular Value Decomposition (SVD)

- **Logic**: SVD operates directly on the data matrix $X = U \Sigma V^T$.
- **Equivalence**: We demonstrate that the right singular vectors $V$ are equivalent to the eigenvectors of the covariance matrix, and the singular values $s$ are related to eigenvalues by $\lambda_i = \frac{s_i^2}{n-1}$.
- **Advantage**: SVD is generally more numerically stable as it avoids computing $X^T X$ explicitly.

### 3. Power Iteration

- **Logic**: An iterative algorithm to find the dominant eigenvalue and eigenvector without performing a full decomposition.
- **Implementation**: We implement this from scratch to demonstrate how the first Principal Component can be approximated through repeated matrix multiplication, verifying convergence against the analytical solution.

## Features

- **Data Independence**: While the Iris dataset is used for visualization, the implementation is designed to be generic for any numerical dataset.
- **Standardization Logic**: Implements Z-score normalization ($z = \frac{x - \mu}{\sigma}$) to ensure PCA is not biased by feature magnitude.
- **Dimensionality Reduction**: Projects $d$-dimensional data into a $k$-dimensional subspace ($k < d$) defined by the top $k$ eigenvectors.
- **Noise Reduction**: Demonstrates the algorithmic capacity of PCA to filter out Gaussian noise by reconstructing data from a reduced set of components.

## Mathematical Verification

- **Method Equivalence**: The implementation proves that Eigen-decomposition and SVD yield identical Principal Components (within floating-point tolerance $\approx 10^{-15}$).
- **Convergence**: The Power Iteration method is shown to converge to the exact same vector as the dominant eigenvector found by standard libraries.
