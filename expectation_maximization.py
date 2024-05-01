import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

# ----------------------------------------------------------------------


def compute_SSE(data, labels):
    """
    Calculate the sum of squared errors (SSE) for a clustering.

    Parameters:
    - data: numpy array of shape (n, 2) containing the data points
    - labels: numpy array of shape (n,) containing the cluster assignments

    Returns:
    - sse: the sum of squared errors
    """
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse


def compute_ARI(confusion_matrix: NDArray[np.int32]):
    """
    Compute the Adjusted Rand Index (ARI) metric for evaluating the performance of a clustering algorithm.

    Parameters:
    confusion_matrix (numpy.ndarray): The confusion matrix representing the clustering results.

    Returns:
    float: The computed ARI value.
    """
    tp = confusion_matrix[0, 0]
    tn = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    ari = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ari


def confusion_matrix(true_labels, predicted_labels):
    classes = np.unique(np.concatenate((true_labels, predicted_labels)))
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_index = {cls: idx for idx, cls in enumerate(classes)}
    for true, pred in zip(true_labels, predicted_labels):
        conf_matrix[class_index[true]][class_index[pred]] += 1
    return conf_matrix


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """
    Compute the adjusted Rand index.

    Parameters:
    - labels_true: The true labels of the data points.
    - labels_pred: The predicted labels of the data points.

    Returns:
    - ari: The adjusted Rand index value.
    """
    contingency_table = np.histogram2d(
        labels_true,
        labels_pred,
        bins=(np.unique(labels_true).size, np.unique(labels_pred).size),
    )[0]
    sum_combinations_rows = np.sum(
        [np.sum(nj) * (np.sum(nj) - 1) / 2 for nj in contingency_table]
    )
    sum_combinations_cols = np.sum(
        [np.sum(ni) * (np.sum(ni) - 1) / 2 for ni in contingency_table.T]
    )
    N = np.sum(contingency_table)
    sum_combinations_total = N * (N - 1) / 2
    ari = (
        np.sum([np.sum(n_ij) * (np.sum(n_ij) - 1) / 2 for n_ij in contingency_table])
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    ) / (
        (sum_combinations_rows + sum_combinations_cols) / 2
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    )
    return ari


def multivariate_pdf(data, mean, cov):
    """Calculate the probability density function of a multivariate normal distribution."""
    mean = np.asarray(mean).flatten()
    cov = np.asarray(cov)
    k = mean.size
    assert data.shape[1] == k, "Data and mean dimensions do not match"
    assert (
        mean.size == cov.shape[0] == cov.shape[1]
    ), "Mean and covariance dimensions do not match"
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** k * det_cov))
    diff = data - mean
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
    return norm_const * np.exp(exponent)


def extract_samples(
    data: NDArray[np.floating], labels: NDArray[np.int32], num_samples: int
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    data_samples = data[indices]
    label_samples = labels[indices]
    return data_samples, label_samples


def em_algorithm(data: NDArray[np.floating], max_iter: int = 100) -> tuple[
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
]:
    n, d = data.shape
    weights = np.random.rand(2)
    weights /= np.sum(weights)
    means = np.random.rand(2, d)
    covariances = np.array([np.eye(d)] * 2)
    log_likelihoods = []
    probabilities = np.zeros((n, 2))
    for _ in range(max_iter):
        probabilities = np.array(
            [
                multivariate_pdf(data, mean, cov)
                for mean, cov in zip(means, covariances)
            ]
        ).T
        responsibilities = probabilities * weights
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        weighted_sum = np.sum(probabilities * weights, axis=1)
        log_likelihood = np.sum(np.log(weighted_sum))
        log_likelihoods.append(log_likelihood)
        for i in range(2):
            weight = responsibilities[:, i].sum()
            weights[i] = weight / n
            means[i] = (
                np.sum(data * responsibilities[:, i][:, np.newaxis], axis=0) / weight
            )
            cov_diff = data - means[i]
            covariances[i] = (
                np.dot(cov_diff.T, cov_diff * responsibilities[:, i][:, np.newaxis])
                / weight
            )
    if means[0][0] < means[1][0]:
        weights = np.flip(weights)
        means = np.flip(means, axis=0)
        covariances = np.flip(covariances, axis=0)
        responsibilities = np.flip(responsibilities, axis=1)
    predicted_labels = np.argmax(responsibilities, axis=1)
    return (
        weights,
        means,
        covariances,
        np.array(log_likelihoods),
        predicted_labels,
    )


def gaussian_mixture():
    answers = {}
    data = np.load("question2_cluster_data.npy")
    labels = np.load("question2_cluster_labels.npy")

    def sample_estimate(data, labels, nb_samples=10000, max_iter=100):
        data_samples, label_samples = extract_samples(data, labels, nb_samples)
        weights, means, covariances, log_likelihoods, predicted_labels = em_algorithm(
            data_samples, max_iter
        )
        confusion_mat = confusion_matrix(label_samples, predicted_labels)
        ARI = adjusted_rand_index(label_samples, predicted_labels)
        SSE = compute_SSE(data_samples, predicted_labels)
        return weights, means, covariances, log_likelihoods, confusion_mat, ARI, SSE

    data_samples, label_samples = extract_samples(data, labels, 10000)

    plot = plt.scatter(data_samples[:, 0], data_samples[:, 1], c=label_samples, s=0.1)
    answers["plot_original_cluster"] = plot

    answers["em_algorithm_function"] = em_algorithm

    max_iter = 100
    weights, means, covariances, log_likelihoods, predicted_labels = em_algorithm(
        data, max_iter
    )

    answers["log_likelihood"] = log_likelihoods

    plot_likelihood = plt.plot(list(range(max_iter)), log_likelihoods)
    plt.title("Log Likelihood vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.grid(True)
    answers["plot_log_likelihood"] = plot_likelihood
    plt.savefig("plot_log_likelihood.pdf")

    nb_trials = 10
    nb_samples = 10000
    max_iter = max_iter

    means_lst = []
    weights_lst = []
    covariances_lst = []
    confusion_lst = []
    ARI_lst = []
    SSE_lst = []
    for _ in range(nb_trials):
        weights, means, covariances, log_likelihoods, confusion_mat, ARI, SSE = (
            sample_estimate(data, labels, nb_samples=nb_samples, max_iter=max_iter)
        )
        weights_lst.append(weights)
        means_lst.append(means)
        covariances_lst.append(covariances)
        confusion_lst.append(np.array(confusion_mat))
        ARI_lst.append(ARI)
        SSE_lst.append(SSE)

    avg_weights = np.mean(weights_lst, axis=0)
    avg_means = np.mean(means_lst, axis=0)
    avg_covariances = np.mean(covariances_lst, axis=0)
    std_weights = np.std(weights_lst, axis=0)
    std_means = np.std(means_lst, axis=0)
    std_covariances = np.std(covariances_lst, axis=0)
    avg_ARI = np.mean(ARI_lst)
    std_ARI = np.std(ARI_lst)
    avg_SSE = np.mean(SSE_lst)
    std_SSE = np.std(SSE_lst)

    answers["probability_1_mean"] = [avg_means[0].tolist(), std_means[0].tolist()]
    answers["probability_2_mean"] = [avg_means[0].tolist(), std_means[1].tolist()]
    answers["probability_1_covariance"] = [
        avg_covariances[0],
        std_covariances[0],
    ]
    answers["probability_2_covariance"] = [
        avg_covariances[1],
        std_covariances[1],
    ]
    answers["probability_1_amplitude"] = [
        avg_weights[0].tolist(),
        std_weights[0].tolist(),
    ]
    answers["probability_2_amplitude"] = [
        avg_weights[1].tolist(),
        std_weights[1].tolist(),
    ]
    answers["average_confusion_matrix"] = np.mean(confusion_lst, axis=0)
    answers["std_confusion_matrix"] = np.std(confusion_lst, axis=0)
    answers["ARI"] = ARI_lst
    answers["SSE"] = SSE_lst
    answers["avg_std_ARI"] = [avg_ARI, std_ARI]
    answers["avg_std_SSE"] = [avg_SSE, std_SSE]

    return answers


if __name__ == "__main__":
    all_answers = gaussian_mixture()
    with open("gaussian_mixture.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
