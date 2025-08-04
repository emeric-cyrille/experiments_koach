import numpy as np
import scipy.linalg as la
from sklearn.neighbors import NearestNeighbors
import ot



class SpectralWassersteinDistance:
    """
    Implementation of Spectral Wasserstein Distance for measuring
    distance between embedding spaces using spectral perturbation theory
    """

    def __init__(self, k=3, method='r1'):
        """
        Initialize the Spectral Wasserstein Distance calculator

        Args:
            k (int): Number of eigenvectors to use
            method (str): 'r1' for r=1 case, 'r_multi' for r>1 case
        """
        self.k = k
        self.method = method

    def load_vec_file(self, filepath):
        """
        Load embeddings from .vec file format

        Args:
            filepath (str): Path to .vec file

        Returns:
            np.ndarray: Embedding matrix of shape (n_tokens, embedding_dim)
        """
        embeddings = []
        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip first line if it contains vocabulary size and dimension
            first_line = f.readline().strip()
            if len(first_line.split()) == 2:
                # First line contains vocab_size and dimension
                pass
            else:
                # First line is an embedding, reset file pointer
                f.seek(0)

            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    # Skip the word and take only the embedding values
                    embedding = [float(x) for x in parts[1:]]
                    embeddings.append(embedding)

        return np.array(embeddings)

    def compute_covariance_matrix(self, embeddings):
        """
        Compute covariance matrix from embeddings

        Args:
            embeddings (np.ndarray): Embedding matrix (n_tokens, d)

        Returns:
            np.ndarray: Covariance matrix (d, d)
        """
        n = embeddings.shape[0]
        return (1 / n) * embeddings.T @ embeddings

    def compute_acp(self, cov_matrix):
        """
        Compute eigendecomposition (ACP - Analyse en Composantes Principales)

        Args:
            cov_matrix (np.ndarray): Covariance matrix

        Returns:
            tuple: (eigenvectors, eigenvalues) sorted by decreasing eigenvalues
        """
        eigenvalues, eigenvectors = la.eigh(cov_matrix)

        # Sort by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvectors, eigenvalues

    def normalize_eigenvalues(self, eigenvalues):
        """
        Normalize eigenvalues to create probability distributions

        Args:
            eigenvalues (np.ndarray): Eigenvalues

        Returns:
            np.ndarray: Normalized eigenvalues (probability weights)
        """
        return eigenvalues / np.sum(eigenvalues)

    def compute_cost_matrix_r1(self, V_A, V_B):
        """
        Compute cost matrix for r=1 case using subspace distances

        Args:
            V_A (np.ndarray): Eigenvectors from space A (d, k)
            V_B (np.ndarray): Eigenvectors from space B (d, k)

        Returns:
            np.ndarray: Cost matrix (k, k)
        """
        k = V_A.shape[1]
        C = np.zeros((k, k))

        for i in range(k):
            for j in range(k):
                # Distance between subspaces spanned by single vectors
                dot_product = np.abs(V_A[:, i].T @ V_B[:, j])
                C[i, j] = np.sqrt(1 - dot_product ** 2)

        return C

    def compute_cost_matrix_r_multi(self, V_A, V_B, r=3):
        """
        Compute cost matrix for r>1 case using KNN-based local subspaces

        Args:
            V_A (np.ndarray): Eigenvectors from space A (d, k)
            V_B (np.ndarray): Eigenvectors from space B (d, k)
            r (int): Number of neighbors for KNN

        Returns:
            np.ndarray: Cost matrix (k, k)
        """
        k = V_A.shape[1]
        C = np.zeros((k, k))

        # Transpose for easier indexing
        V_A_T = V_A.T  # (k, d)
        V_B_T = V_B.T  # (k, d)

        # Build KNN for both sets of eigenvectors
        nbrs_A = NearestNeighbors(n_neighbors=min(r, k), metric='cosine')
        nbrs_A.fit(V_A_T)


        nbrs_B = NearestNeighbors(n_neighbors=min(r, k), metric='cosine')
        nbrs_B.fit(V_B_T)

        for i in range(k):
            # Find r nearest neighbors for V_A[i]
            _, indices_A = nbrs_A.kneighbors([V_A_T[i]])
            E_i = V_A[:, indices_A[0]]  # (d, r)

            for j in range(k):
                # Find r nearest neighbors for V_B[j]
                _, indices_B = nbrs_B.kneighbors([V_B_T[j]])
                F_j = V_B[:, indices_B[0]]  # (d, r)

                # Compute projection matrices
                Pi_E = E_i @ E_i.T
                Pi_F = F_j @ F_j.T

                # Compute distance between subspaces
                C[i, j] = (1 / np.sqrt(2)) * la.norm(Pi_E - Pi_F, 'fro')

        return C

    def solve_optimal_transport(self, r, c, C):
        """
        Solve optimal transport problem using POT library

        Args:
            r (np.ndarray): Source distribution
            c (np.ndarray): Target distribution
            C (np.ndarray): Cost matrix

        Returns:
            float: Wasserstein distance
        """
        # Use POT library to compute exact Wasserstein distance
        wasserstein_distance = ot.emd2(r, c, C)

        return wasserstein_distance

    def compute_distance(self, embeddings_A, embeddings_B):
        """
        Compute Spectral Wasserstein Distance between two embedding spaces

        Args:
            embeddings_A (np.ndarray): Embeddings from space A
            embeddings_B (np.ndarray): Embeddings from space B

        Returns:
            float: Spectral Wasserstein Distance
        """
        # Step 1: Compute covariance matrices
        Sigma_A = self.compute_covariance_matrix(embeddings_A)
        Sigma_B = self.compute_covariance_matrix(embeddings_B)

        # Step 2: Compute eigendecompositions
        V_A, lambda_A = self.compute_acp(Sigma_A)
        V_B, lambda_B = self.compute_acp(Sigma_B)

        # Step 3: Take top k eigenvectors and eigenvalues
        V_A = V_A[:, :self.k]
        V_B = V_B[:, :self.k]
        lambda_A = lambda_A[:self.k]
        lambda_B = lambda_B[:self.k]

        # Step 4: Normalize eigenvalues to create distributions
        r = self.normalize_eigenvalues(lambda_A)
        c = self.normalize_eigenvalues(lambda_B)

        # Step 5: Compute cost matrix
        if self.method == 'r1':
            C = self.compute_cost_matrix_r1(V_A, V_B)
        else:
            C = self.compute_cost_matrix_r_multi(V_A, V_B)

        # Step 6: Solve optimal transport
        distance = self.solve_optimal_transport(r, c, C)

        return distance

    def compute_distance_from_files(self, filepath_A, filepath_B):
        """
        Compute distance between two .vec files

        Args:
            filepath_A (str): Path to first .vec file
            filepath_B (str): Path to second .vec file

        Returns:
            float: Spectral Wasserstein Distance
        """
        embeddings_A = self.load_vec_file(filepath_A)
        embeddings_B = self.load_vec_file(filepath_B)

        print(f"Loaded embeddings A: {embeddings_A.shape}")
        print(f"Loaded embeddings B: {embeddings_B.shape}")

        return self.compute_distance(embeddings_A, embeddings_B)




def main():
    """
    Example usage of the Spectral Wasserstein Distance
    """

    # Initialize the distance calculator
    swd_r1 = SpectralWassersteinDistance(k=100, method='r1')
    swd_r_multi = SpectralWassersteinDistance(k=100, method='r_multi')

    # Compute distances
    print("Computing Spectral Wasserstein Distance (r=1 method)...")
    distance_r1 = swd_r1.compute_distance_from_files("embeddings_pretrained/ngh_xlmr.vec", "embeddings_pretrained/ngh_mbert.vec")
    print(f"Distance (r=1): {distance_r1:.6f}")

    print("\nComputing Spectral Wasserstein Distance (r>1 method)...")
    distance_r_multi = swd_r_multi.compute_distance_from_files("embeddings_pretrained/ngh_xlmr.vec", "embeddings_pretrained/ngh_mbert.vec")
    print(f"Distance (r>1): {distance_r_multi:.6f}")

    # Example with custom file paths
    print("\n" + "=" * 50)



if __name__ == "__main__":
    main()