#
#      0=============================0
#      |    TP3 Point Descriptors    |
#      0=============================0
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#

# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from ply import write_ply, read_ply
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#

def PCA(points):
    """
    PCA on a set of 3D points using covariance matrix.

    Input:
        points: (N,3) numpy array

    Returns:
        eigenvalues: (3,) in ASCENDING order [lambda_3, lambda_2, lambda_1]
        eigenvectors: (3,3) columns are the eigenvectors associated to eigenvalues
            eigenvectors[:, 0] -> eigenvector for lambda_3 (smallest variance direction)
            eigenvectors[:, 2] -> eigenvector for lambda_1 (largest variance direction)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PCA expects an array of shape (N,3)")

    n = points.shape[0]
    if n == 0:
        return np.zeros(3), np.eye(3)

    # centroid
    pm = np.mean(points, axis=0, keepdims=True)  # (1,3)
    Q = points - pm                               # (N,3)

    # covariance (3x3)
    C = (Q.T @ Q) / float(n)

    # symmetric eigendecomposition (ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    return eigenvalues, eigenvectors


def compute_local_PCA(query_points, cloud_points, radius):
    """
    Compute PCA on spherical neighborhoods (radius) for each query point.

    Inputs:
        query_points: (Nq,3)
        cloud_points: (Ns,3)
        radius: float

    Returns:
        all_eigenvalues: (Nq,3) in ASCENDING order [lambda_3, lambda_2, lambda_1]
        all_eigenvectors: (Nq,3,3) columns are eigenvectors (same convention as PCA)
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    Nq = query_points.shape[0]
    all_eigenvalues = np.zeros((Nq, 3), dtype=np.float64)
    all_eigenvectors = np.zeros((Nq, 3, 3), dtype=np.float64)

    # KDTree for neighbor search
    tree = KDTree(cloud_points)

    # Query all neighborhoods once (faster)
    neigh_lists = tree.query_radius(query_points, r=radius, return_distance=False)

    for i in range(Nq):
        inds = neigh_lists[i]

        # Need at least 3 points to fit a plane in 3D
        if inds is None or len(inds) < 3:
            all_eigenvalues[i, :] = 0.0
            all_eigenvectors[i, :, :] = np.eye(3)
            continue

        neighbors = cloud_points[inds, :]
        vals, vecs = PCA(neighbors)

        all_eigenvalues[i, :] = vals
        all_eigenvectors[i, :, :] = vecs

    return all_eigenvalues, all_eigenvectors


def compute_local_PCA_knn(query_points, cloud_points, k):
    """
    Compute PCA on k-nearest-neighbors neighborhoods for each query point.

    Inputs:
        query_points: (Nq,3)
        cloud_points: (Ns,3)
        k: int >= 3

    Returns:
        all_eigenvalues: (Nq,3) in ASCENDING order [lambda_3, lambda_2, lambda_1]
        all_eigenvectors: (Nq,3,3)
    """
    if k < 3:
        raise ValueError("k must be >= 3")

    Nq = query_points.shape[0]
    all_eigenvalues = np.zeros((Nq, 3), dtype=np.float64)
    all_eigenvectors = np.zeros((Nq, 3, 3), dtype=np.float64)

    tree = KDTree(cloud_points)
    # inds: (Nq, k)
    _, inds = tree.query(query_points, k=k, return_distance=True)

    for i in range(Nq):
        neighbors = cloud_points[inds[i], :]
        vals, vecs = PCA(neighbors)
        all_eigenvalues[i, :] = vals
        all_eigenvectors[i, :, :] = vecs

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    """
    Compute 4 features based on local PCA:
      - verticality
      - linearity
      - planarity
      - sphericity

    Returns:
        verticality, linearity, planarity, sphericity: (Nq,) each
    """
    eps = 1e-12

    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)

    # eigenvalues are ASCENDING: [lambda_3, lambda_2, lambda_1]
    lam3 = all_eigenvalues[:, 0]
    lam2 = all_eigenvalues[:, 1]
    lam1 = all_eigenvalues[:, 2]
    denom = lam1 + eps

    # normal = eigenvector of smallest eigenvalue (column 0)
    normals = all_eigenvectors[:, :, 0]  # (Nq,3)

    # verticality in [0,1]: 2*arcsin(|<n, ez>|)/pi ; ez=(0,0,1) so <n,ez>=n_z
    nz = np.clip(np.abs(normals[:, 2]), 0.0, 1.0)
    verticality = 2.0 * np.arcsin(nz) / np.pi

    # eigenvalue-based descriptors (using lambda1 >= lambda2 >= lambda3 convention)
    linearity = 1.0 - (lam2 / denom)
    planarity = (lam2 - lam3) / denom
    sphericity = lam3 / denom

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#

if __name__ == '__main__':

    # Load cloud as a [N x 3] matrix
    cloud_path = '../data/Lille_street_small.ply'
    cloud_ply = read_ply(cloud_path)
    cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

    # PCA verification
    # ****************
    if True:

        eigenvalues, eigenvectors = PCA(cloud)
        print("Eigenvalues returned by PCA (ascending: [lambda_3, lambda_2, lambda_1]):")
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)

    # Normal computation (radius)
    # ***************************
    if True:

        radius = 0.50
        t0 = time.time()
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, radius)
        t1 = time.time()
        print(f"Local PCA (radius={radius}) done in {t1 - t0:.2f}s")

        normals = all_eigenvectors[:, :, 0]  # normal = e3 (smallest variance direction)

        write_ply('../Lille_street_small_normals_r0p50.ply',
                  (cloud, normals),
                  ['x', 'y', 'z', 'nx', 'ny', 'nz'])

    # Normal computation (kNN) - for Question 4
    # *****************************************
    if True:

        k = 30
        t0 = time.time()
        all_eigenvalues_knn, all_eigenvectors_knn = compute_local_PCA_knn(cloud, cloud, k)
        t1 = time.time()
        print(f"Local PCA (kNN, k={k}) done in {t1 - t0:.2f}s")

        normals_knn = all_eigenvectors_knn[:, :, 0]

        write_ply('../Lille_street_small_normals_k30.ply',
                  (cloud, normals_knn),
                  ['x', 'y', 'z', 'nx', 'ny', 'nz'])

    # Feature computation (BONUS)
    # ***************************
    if True:

        radius_feat = 0.50
        t0 = time.time()
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, radius_feat)
        t1 = time.time()
        print(f"Features (radius={radius_feat}) done in {t1 - t0:.2f}s")

        write_ply('../Lille_street_small_features_r0p50.ply',
                  (cloud, verticality, linearity, planarity, sphericity),
                  ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
