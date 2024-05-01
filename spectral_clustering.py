import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.spatial.distance import cdist

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - cluster_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    def custom_k_means(data, k, max_iter=100, tol=1e-4):
        n_samples, n_features = data.shape
        # Initialize centroids by randomly selecting k points from the data
        centroids = data[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iter):
            # Calculate distances from each data point to each centroid
            dists = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            # Assign clusters based on closest centroid
            clusters = np.argmin(dists, axis=1)

            new_centroids = np.zeros((k, n_features))
            for i in range(k):
                # Extract points assigned to the current cluster
                cluster_points = data[clusters == i]
                if cluster_points.size > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    # Reinitialize centroid to a random point from the data if the cluster is empty
                    new_centroids[i] = data[np.random.choice(n_samples, 1, replace=False)].flatten()

            # Check for convergence: if centroids do not change significantly, exit loop
            if np.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids

        return clusters, centroids


    def calculate_sse(data, labels, centroids):
        """
        Calculate the Sum of Squared Errors (SSE) for given data and centroids.
        
        Parameters:
            data (np.ndarray): The dataset.
            labels (np.ndarray): Array of cluster labels for each data point.
            centroids (np.ndarray): Array of centroids, one for each cluster.
        
        Returns:
            float: The calculated SSE.
        """
        k = len(centroids)
        SSE = 0
        for i in range(k):
            cluster_data = data[labels == i]
            if cluster_data.size > 0:
                SSE += np.sum((cluster_data - centroids[i])**2)
        return SSE


    def adjusted_rand_index(true_labels, pred_labels):
        from scipy.special import comb
        n = len(true_labels)
        categories = np.unique(true_labels)
        clusters = np.unique(pred_labels)

        # Create contingency table
        contingency = np.array([[np.sum((true_labels == cat) & (pred_labels == clus)) for clus in clusters] for cat in categories])
        sum_comb_c = np.sum([comb(n_c, 2) for n_c in np.sum(contingency, axis=1)])
        sum_comb_k = np.sum([comb(n_k, 2) for n_k in np.sum(contingency, axis=0)])
        sum_comb = np.sum([comb(n_ij, 2) for n_ij in contingency.flatten()])
        total_comb = comb(n, 2)
        expected_comb = sum_comb_c * sum_comb_k / total_comb
        max_comb = (sum_comb_c + sum_comb_k) / 2
        
        if total_comb == expected_comb:  # Prevent division by zero
            return 0.0
        else:
            ARI = (sum_comb - expected_comb) / (max_comb - expected_comb)
            return ARI


    sigma = params_dict['sigma']
    k = params_dict['k']

    # Create the similarity matrix using the Gaussian kernel
    dists = cdist(data, data, 'sqeuclidean')
    W = np.exp(-dists / (2 * sigma**2))

    # Create the diagonal matrix for the degrees of the nodes
    D = np.diag(W.sum(axis=1))

    # Create the Laplacian matrix
    L = D - W

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Use the first k eigenvectors for clustering
    V = eigenvectors[:, :k]
    
    # k-means on the rows of V
    cluster_labels, centroids = custom_k_means(V, k)

   # Compute SSE, ensuring data is correctly sliced per cluster and centroids have correct dimension
    SSE = calculate_sse(V, cluster_labels, centroids)

    # Compute ARI
    ARI = adjusted_rand_index(labels, cluster_labels)

    return cluster_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').
    sse = []
    ari = []
    max_ari = 0
    min_sse = 0
    sigma_for_max_ari = 0
    sigma_for_min_sse = 0
    sigma = np.linspace(0.1, 10, 10)
    k = 5
    data = np.load('question1_cluster_data.npy')[:5000]
    labels = np.load('question1_cluster_labels.npy')[:5000]
    params_dict = {}
    for i in range(len(sigma)):
        params_dict['sigma'] = sigma[i]
        params_dict['k'] = k
        cluster_labels, SSE, ARI, eigenvalues = spectral(data[:1000], labels[:1000], params_dict)
        if i==0:
            min_sse = SSE
            sigma_for_min_sse = sigma[i]
        elif SSE < min_sse:
            min_sse = SSE
            sigma_for_min_sse = sigma[i]
        if ARI > max_ari:
            max_ari = ARI
            sigma_for_max_ari = sigma[i]
        ari.append(ARI)
        sse.append(SSE)


    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    
    max_ari_data = None
    max_ari_labels = None
    min_sse_data = None
    min_sse_labels = None

    final_sigma = sigma_for_max_ari
    eigenvalues = np.array([])
    
    for i in range(5):
        params_dict['sigma'] = final_sigma
        params_dict['k'] = k
        index_start = 1000 * i
        index_end = 1000 * (i + 1)
        cluster_labels, SSE, ARI, eigenvalue = spectral(data[index_start:index_end], labels[index_start:index_end], params_dict)
        groups[i] = {"sigma": float(final_sigma), "ARI": float(ARI), "SSE": float(SSE)}
        eigenvalues = np.append(eigenvalues, eigenvalue, axis=0)
        
        if i==0:
            min_sse = SSE
            min_sse_data = data[index_start:index_end]
            min_sse_labels = cluster_labels
            max_ari = ARI
            max_ari_data = data[index_start:index_end]
            max_ari_labels = cluster_labels
        
        if ARI > max_ari:
            max_ari = ARI
            max_ari_data = data[index_start:index_end]
            max_ari_labels = cluster_labels

        if SSE < min_sse:
            min_sse = SSE
            min_sse_data = data[index_start:index_end]
            min_sse_labels = cluster_labels
    
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"]

    plt.plot(sigma, ari)
    plt.title('ARI vs Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('ARI')
    plt.savefig('spectral_ARI_vs_Sigma.png')
    plt.show()
    plt.close()

    plt.plot(sigma, sse)
    plt.title('SSE vs Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('SSE')
    plt.savefig('spectral_SSE_vs_Sigma.png')
    plt.show()
    plt.close()

    sigmas = np.array([group["sigma"] for group in groups.values()])
    ARIs = np.array([group["ARI"] for group in groups.values()])
    SSEs = np.array([group["SSE"] for group in groups.values()])

    plot_ARI = plt.scatter(max_ari_data[:, 0], max_ari_data[:, 1], c=max_ari_labels, cmap='viridis', s=25)
    plt.title('Clusters with Largest ARI')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('spectral_cluster scatterplot with largest ARI.png')
    plt.close()

    plot_SSE = plt.scatter(min_sse_data[:, 0], min_sse_data[:, 1], c=min_sse_labels, cmap='viridis', s=25)
    plt.title('Clusters with Smallest SSE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('spectral_cluster scatterplot with smallest SSE.png')
    plt.close()

    plot_eig = plt.plot(np.sort(eigenvalues), linestyle='-')
    plt.title("Sorted Eigenvalues Plot")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.savefig('spectral_sorted_eigenvalues_plot.png')
    plt.close()

    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE
    answers["eigenvalue plot"] = plot_eig

    answers["mean_ARIs"] = float(np.mean(ARIs))
    answers["std_ARIs"] = float(np.std(ARIs))
    answers["mean_SSEs"] = float(np.mean(SSEs))
    answers["std_SSEs"] = float(np.std(SSEs))

    return answers


if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
