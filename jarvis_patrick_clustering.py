import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    def calculate_sse(data: np.ndarray, labels: np.ndarray) -> float:
        unique_labels = np.unique(labels[labels != -1])
        sse = sum(np.sum((data[labels == k] - np.mean(data[labels == k], axis=0))**2) for k in unique_labels)
        return sse
    def adjusted_rand_index(true_labels, pred_labels):
        """Calculate the Adjusted Rand Index (ARI) using a contingency table."""
        from scipy.special import comb
        contingency = np.array([[np.sum((true_labels == k) & (pred_labels == j)) for j in np.unique(pred_labels)] for k in np.unique(true_labels)])
        sum_comb_c = np.sum([comb(n_c, 2) for n_c in np.sum(contingency, axis=1)])
        sum_comb_k = np.sum([comb(n_k, 2) for n_k in np.sum(contingency, axis=0)])
        sum_comb = np.sum([comb(n_ij, 2) for n_ij in contingency.flatten()])
        total_comb = comb(len(true_labels), 2)
        expected_comb = sum_comb_c * sum_comb_k / total_comb
        max_comb = (sum_comb_c + sum_comb_k) / 2
        ARI = (sum_comb - expected_comb) / (max_comb - expected_comb) if max_comb != expected_comb else 0
        return ARI  


    k = params_dict['k']
    smin = params_dict['smin']
    n = data.shape[0]

    # Calculate the k-nearest neighbors
    dist_matrix = distance.cdist(data, data)
    indices = np.argsort(dist_matrix, axis=1)[:, 1:k+1]  # Excluding the point itself

    # Build the shared nearest neighbor graph
    snn_graph = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            shared_neighbors = np.intersect1d(indices[i], indices[j], assume_unique=True)
            if len(shared_neighbors) >= smin:
                snn_graph[i, j] = snn_graph[j, i] = True

    # Convert boolean graph to a sparse matrix and find connected components
    graph = csr_matrix(snn_graph)
    n_components, computed_labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # Calculate SSE and ARI
    SSE = calculate_sse(data, computed_labels)
    ARI = adjusted_rand_index(labels, computed_labels)

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    answers["jarvis_patrick_function"] = jarvis_patrick

    max_ari = -np.inf  # Smallest possible float
    min_sse = np.inf   # Largest possible float

    # Initialize variables for recording the parameters corresponding to max ARI and min SSE
    k_for_max_ari, k_for_min_sse = 0, 0
    smin_for_max_ari, smin_for_min_sse = 0, 0

    # Define the range of parameters for the parameter study
    k_values = np.linspace(3, 8, 5, dtype=int)
    smin_values = np.linspace(4, 10, 6, dtype=int)
    ari_list = []
    sse_list = []
    # Load data
    data = np.load('question1_cluster_data.npy')[:5000]
    labels = np.load('question1_cluster_labels.npy')[:5000]
    params_dict = {}
    best_ari = -np.inf
    best_sse = np.inf
    best_params_ari = {}
    best_params_sse = {}

    # Dictionary to store all results for visualization
    results = []

    # Loop over all combinations of k and smin
    for k in k_values:
        for smin in smin_values:
            params_dict = {'k': k, 'smin': smin}
            computed_labels, SSE, ARI = jarvis_patrick(data[:1000], labels[:1000], params_dict)

            # Store results
            results.append((k, smin, SSE, ARI))

            # Check if this combination gives a higher ARI or lower SSE
            if ARI > best_ari:
                best_ari = ARI
                best_params_ari = {'k': k, 'smin': smin, 'ARI': ARI}
            if SSE < best_sse:
                best_sse = SSE
                best_params_sse = {'k': k, 'smin': smin, 'SSE': SSE}

    # Organize results into a dictionary
    best_results = {
        "results": results,
        "best_params_ari": best_params_ari,
        "best_params_sse": best_params_sse
    }

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').



    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    final_k = best_params_ari['k']
    final_smin = best_params_ari['smin']
    max_ari_data = None
    max_ari_labels = None
    min_sse_data = None
    min_sse_labels = None
    min_sse = np.inf
    max_ari = -np.inf
    for i in range(5):
        params_dict['k'] = final_k
        params_dict['smin'] = final_smin
        index_start = 1000 * i
        index_end = 1000 * (i + 1) 
        computed_labels, SSE, ARI = jarvis_patrick(data[index_start:index_end], labels[index_start:index_end], params_dict)
        groups[i] = {"k": final_k, "smin": final_smin, "ARI": ARI, "SSE": SSE}

        if SSE < min_sse:
            min_sse = SSE
            min_sse_data = data[index_start:index_end]
            min_sse_labels = computed_labels
        
        if ARI > max_ari:
            max_ari = ARI
            max_ari_data = data[index_start:index_end]
            max_ari_labels = computed_labels



    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"]

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    k_values = [item[0] for item in best_results['results']]
    smin_values = [item[1] for item in best_results['results']]
    aris = [item[2] for item in best_results['results']]
    sses = [item[3] for item in best_results['results']]

    plt.scatter(k_values, smin_values, c=aris, cmap='viridis', s=25)
    plt.title('Clusters with Largest ARI')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('jarvis_cluster_scatterplot_with_largest_ARI.png')
    plt.show()
    plt.close()

    plt.scatter(k_values, smin_values, c=sses, cmap='viridis', s=25)
    plt.title('Clusters with Smallest SSE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('jarvis_cluster_scatterplot_with_smallest_SSE.png')
    plt.show()
    plt.close()

    ks = np.array([group["k"] for group in groups.values()])
    smins = np.array([group["smin"] for group in groups.values()])
    ARIs = np.array([group["ARI"] for group in groups.values()])
    SSEs = np.array([group["SSE"] for group in groups.values()])
    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter(max_ari_data[:, 0], max_ari_data[:, 1], c=max_ari_labels, cmap='viridis', s=25)
    plt.title('Clusters with Largest ARI')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('jarvis_cluster_scatterplot_with_largest_ARI.png')
    plt.close()
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    
    plot_SSE = plt.scatter(min_sse_data[:, 0], min_sse_data[:, 1], c=min_sse_labels, cmap='viridis', s=25)
    plt.title('Clusters with Smallest SSE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('jarvis_cluster_scatterplot_with_smallest_SSE.png')
    plt.close()
    answers["cluster scatterplot with smallest SSE"] = plot_SSE



    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = float(np.mean(ARIs))

    # A single float
    answers["std_ARIs"] = float(np.std(ARIs))

    # A single float
    answers["mean_SSEs"] = float(np.mean(SSEs))

    # A single float
    answers["std_SSEs"] = float(np.std(SSEs))

    return answers



# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
