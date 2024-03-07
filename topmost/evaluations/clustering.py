import numpy as np
from collections import defaultdict
from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metric(labels, preds):
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
    ]

    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)

    return results


def evaluate_clustering(theta, labels):
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)


def hierarchical_clustering(test_theta, test_labels):
    num_layer = len(test_theta)
    results = defaultdict(list)

    for layer in range(num_layer):
        layer_results = evaluate_clustering(test_theta[layer], test_labels)

        for key in layer_results:
            results[key].append(layer_results[key])

    for key in results:
        results[key] = np.mean(results[key])

    return results
