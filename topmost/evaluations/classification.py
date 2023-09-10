import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict


def evaluate_classification(train_theta, test_theta, train_labels, test_labels, classifier='SVM'):
    if classifier == 'SVM':
        clf = SVC()
    else:
        raise NotImplementedError

    clf.fit(train_theta, train_labels)
    preds = clf.predict(test_theta)
    results = {
        'acc': accuracy_score(test_labels, preds),
        'macro-F1': f1_score(test_labels, preds, average='macro')
    }
    return results


def evaluate_crosslingual_classification(
    train_theta_en,
    train_theta_cn,
    test_theta_en,
    test_theta_cn,
    train_labels_en,
    train_labels_cn,
    test_labels_en,
    test_labels_cn,
    classifier="SVM"
):
    intra_en = evaluate_classification(train_theta_en, test_theta_en, train_labels_en, test_labels_en, classifier)
    intra_cn = evaluate_classification(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn, classifier)

    cross_en = evaluate_classification(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en, classifier)
    cross_cn = evaluate_classification(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn, classifier)

    return {
        'intra_en': intra_en,
        'intra_cn': intra_cn,
        'cross_en': cross_en,
        'cross_cn': cross_cn
    }


def evaluate_hierarchical_classification(train_theta, test_theta, train_labels, test_labels, classifier='SVM'):
    num_layer = len(train_theta)
    results = defaultdict(list)

    for layer in range(num_layer):
        layer_results = evaluate_classification(train_theta[layer], test_theta[layer], train_labels, test_labels, classifier)

        for key in layer_results:
            results[key].append(layer_results[key])

    for key in results:
        results[key] = np.mean(results[key])

    return results
