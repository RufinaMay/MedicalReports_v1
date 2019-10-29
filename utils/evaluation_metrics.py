import numpy as np

def precision_recall(true_tags, predicted_tags):
    n_tags = true_tags.shape[1]
    N_c, N_g, N_p = np.zeros(n_tags), np.zeros(n_tags), np.zeros(n_tags)
    for t, p in zip(true_tags, predicted_tags):
        print(t)
        print(p)
        for i in range(len(t)):
            N_g[t[i]] += 1
            N_p[p[i]] += 1
            if t[i] == p[i]:
                N_c[t[i]] += 1

    overall_recall = np.sum(N_c) / np.sum(N_g)
    overall_precision = np.sum(N_c) / np.sum(N_p)

    idx = np.where(N_g == 0)
    N_g[idx] = 1
    per_class_recall = np.sum(N_c / N_g) / n_tags

    idx = np.where(N_p == 0)
    N_p[idx] = 1
    per_class_precision = np.sum(N_c / N_p) / n_tags

    return per_class_recall, per_class_precision, overall_recall, overall_precision