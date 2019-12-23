import numpy as np
import pickle
from collections import Counter
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
import scipy.sparse as sp
from matplotlib import pyplot as plt


def sparse_to_nornal(X_data, y_data, idx_to_img, idx_to_tag):
    """
    convert from sparse format back to img-tag mapping format
    :param X_data:
    :param y_data:
    :param idx_to_img:
    :return:
    """
    data = {}
    X_data, y_data = np.ravel(X_data.todense()), y_data.todense()
    for x, y in zip(X_data, y_data):
        vect = [idx_to_tag[i] for i in np.where(np.ravel(y) == 1)[0]]
        data[idx_to_img[x]] = vect
    return data


def draw_distribution(train, valid, test):
    train_tags_stat, valid_tags_stat, test_tags_stat = Counter(), Counter(), Counter()
    for k in train:
        for tag in train[k]:
            train_tags_stat[tag] += 1
    for k in valid:
        for tag in valid[k]:
            valid_tags_stat[tag] += 1
    for k in test:
        for tag in test[k]:
            test_tags_stat[tag] += 1

    height1 = np.array([train_tags_stat[k] for k in train_tags_stat])
    left1 = np.array([k for k in train_tags_stat])
    idxs = np.argsort(left1)
    left1 = np.take_along_axis(left1, idxs, axis=0)
    height1 = np.take_along_axis(height1, idxs, axis=0)

    height2 = np.array([test_tags_stat[k] for k in test_tags_stat])
    left2 = np.array([k for k in test_tags_stat])
    idxs = np.argsort(left2)
    left2 = np.take_along_axis(left2, idxs, axis=0)
    height2 = np.take_along_axis(height2, idxs, axis=0)

    height3 = np.array([valid_tags_stat[k] for k in valid_tags_stat])
    left3 = np.array([k for k in valid_tags_stat])
    idxs = np.argsort(left2)
    left2 = np.take_along_axis(left2, idxs, axis=0)
    height2 = np.take_along_axis(height2, idxs, axis=0)

    plt.bar(left1, height1, width=1.2, color=['red', 'green'])
    plt.title('Train set distribution')
    plt.show()

    plt.bar(left2, height2, width=1.2, color=['red', 'green'])
    plt.title('Test set distribution')
    plt.show()

    plt.bar(left3, height3, width=1.2, color=['red', 'green'])
    plt.title('Valid set distribution')
    plt.show()


def run_split(img_tag_path='IMG_TAG.pickle', tag_to_idx_path='TAG_TO_INDEX.pickle', draw_distrib=True, save=True):
    with open(img_tag_path, 'rb') as f:
        IMG_TAG_MAPPING = pickle.load(f)
    with open(tag_to_idx_path, 'rb') as f:
        TAG_TO_INDEX = pickle.load(f)
    UNIQ_TAGS = len(TAG_TO_INDEX)
    print(f'unique tags {UNIQ_TAGS}')
    idx_to_tag = {TAG_TO_INDEX[t]: t for t in TAG_TO_INDEX}
    X, y = [], []
    img_to_idx = {img: i + 1 for i, img in enumerate(IMG_TAG_MAPPING)}
    idx_to_img = {i + 1: img for i, img in enumerate(IMG_TAG_MAPPING)}
    for img in IMG_TAG_MAPPING:
        X.append(img_to_idx[img])
        vect = np.zeros(UNIQ_TAGS)
        for tag in IMG_TAG_MAPPING[img]:
            vect[TAG_TO_INDEX[tag]] = 1
        y.append(vect)

    X = sp.csr_matrix(np.array(X).reshape(-1, 1))
    y = sp.csr_matrix(np.array(y))

    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.25)
    X_train, y_train, X_valid, y_valid = iterative_train_test_split(X_train, y_train, test_size=0.15)

    print(f'total samples {X.shape[0]}')
    print(f'train samples {X_train.shape[0]}')
    print(f'valid samples {X_valid.shape[0]}')
    print(f'test samples {X_test.shape[0]}')

    train, valid = sparse_to_nornal(X_train, y_train, idx_to_img, idx_to_tag), sparse_to_nornal(X_valid, y_valid,
                                                                                                idx_to_img, idx_to_tag)
    test = sparse_to_nornal(X_test, y_test, idx_to_img, idx_to_tag)

    if draw_distrib:
        draw_distribution(train, valid, test)

    if save:
        with open('train_set.pickle', 'wb') as f:
            pickle.dump(train, f)
        with open('valid_set.pickle', 'wb') as f:
            pickle.dump(valid, f)
        with open('test_set.pickle', 'wb') as f:
            pickle.dump(test, f)

# dataframe = pd.DataFrame({
#     'original': Counter(str(combination) for row in get_combination_wise_output_matrix(y.A, order=2) for combination in row),
#     'train': Counter(str(combination) for row in get_combination_wise_output_matrix(y_train.A, order=2) for combination in row),
#     'test' : Counter(str(combination) for row in get_combination_wise_output_matrix(y_test.A, order=2) for combination in row)
# }).T.fillna(0.0)

# print(dataframe)
