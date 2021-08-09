import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score



# based on  https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(*ind)]) * 1.0 / y_pred.size


def test_averaged_run(dkmeans_builder, kmeans_builder,
             dkmeans_fit, kmeans_fit,
             data, labels,
             number_tests, initial_seed):
    rnd = np.random.default_rng(initial_seed)
    seeds = rnd.integers(1e8,size=number_tests)

    dkmeans_acc, dkmeans_nmi = [], []
    kmeans_acc, kmeans_nmi = [], []

    for s in seeds:
        dkmeans = dkmeans_builder(s)
        kmeans = kmeans_builder(s)

        cls_dkm = dkmeans_fit(dkmeans, data)
        cls_km = kmeans_fit(kmeans, data)

        dkmeans_acc.append(cluster_acc(labels, cls_dkm))
        dkmeans_nmi.append(normalized_mutual_info_score(labels,cls_km))

        kmeans_acc.append(cluster_acc(labels, cls_km))
        kmeans_nmi.append(normalized_mutual_info_score(labels, cls_km))

    return {
        "dkmeans": {
            "acc": {
                "mean": np.mean(dkmeans_acc),
                "var" : np.std(dkmeans_acc),
            },
            "nmi": {
                "mean": np.mean(dkmeans_nmi),
                "var": np.std(dkmeans_nmi),
            }
        },
        "kmeans": {
            "acc": {
                "mean": np.mean(kmeans_acc),
                "var" : np.std(kmeans_acc),
            },
            "nmi": {
                "mean": np.mean(kmeans_nmi),
                "var": np.std(kmeans_nmi),
            }
        }
    }

def print_results(result_dict):
    print("K-means")
    print("\tACC: {:.2f}\u00B1{:.2f}".format(result_dict["kmeans"]["acc"]["mean"]*100,result_dict["kmeans"]["acc"]["var"]*100))
    print("\tNMI: {:.2f}\u00B1{:.2f}".format(result_dict["kmeans"]["nmi"]["mean"]*100, result_dict["kmeans"]["nmi"]["var"]*100))
    print("Deep K-means")
    print("\tACC: {:.2f}\u00B1{:.2f}".format(result_dict["dkmeans"]["acc"]["mean"]*100, result_dict["dkmeans"]["acc"]["var"]*100))
    print("\tNMI: {:.2f}\u00B1{:.2f}".format(result_dict["dkmeans"]["nmi"]["mean"]*100, result_dict["dkmeans"]["nmi"]["var"]*100))
