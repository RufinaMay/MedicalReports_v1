import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter
import torchvision.transforms as transforms
import torch
import os
import pickle

from utils.constants import BATCH_SIZE, IMG_DIR


def normalize(images):
    """
    Normalize image or list of images to the range from -1 to 1
    :param images:
    :return:
    """
    return (np.array(images) - 127.5) / 127.5


def batch_tags(images, one_hot_tags):
    """
    Creates batch from images and corresponding one hot encoded tags
    :param images: list ot numpy array of images with shape (None, 448, 448, 3)
    :param one_hot_tags: list or numpy array of one hot
    :return:
    """
    """
    IMAGES: mapping between image path and its tags
    one_hot_tags: one hot encoded tags
    """
    batch_IMGS, batch_TAGS = [], []
    b = 0
    for im, tag in zip(images, one_hot_tags):
        batch_IMGS.append(im), batch_TAGS.append(tag)
        b += 1

        if b > BATCH_SIZE:
            yield normalize(batch_IMGS), np.array(batch_TAGS)
            b = 0
            batch_IMGS, batch_TAGS = [], []
    yield normalize(batch_IMGS), np.array(batch_TAGS)


def train_test_split(img_tag_mapping, test_size=0.2):
    """
    custom train test split for Indiana Chest Xray dataset
    :param img_tag_mapping: dictionary of img-tags pairs
    :param test_size: size of test set
    :return: train and test sets
    """
    n = len(img_tag_mapping)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    train_size = np.ceil((1 - test_size) * n).astype('int32')
    train_idxs, test_idxs = idxs[:train_size], idxs[train_size:]
    train, test = {}, {}
    i = 0
    for k in img_tag_mapping:
        if i in train_idxs:
            train[k] = img_tag_mapping[k]
        else:
            test[k] = img_tag_mapping[k]
        i += 1
    return train, test


def apply_hierarchy(train_set, valid_set, test_set):
    number_of_tags = Counter()
    for img in train_set:
        for label in train_set[img]:
            number_of_tags[label] += 1

    new_train_set, new_valid_set, new_test_set = {}, {}, {}
    for img in train_set:
        out = []
        for tag in number_of_tags:
            if tag in train_set[img]:
                out.append(tag)
        new_train_set[img] = out

    for img in valid_set:
        out = []
        for tag in number_of_tags:
            if tag in valid_set[img]:
                out.append(tag)
        new_valid_set[img] = out

    for img in test_set:
        out = []
        for tag in number_of_tags:
            if tag in test_set[img]:
                out.append(tag)
        new_test_set[img] = out

    return new_train_set, new_valid_set, new_test_set


def process_predictions(train_pred, y_train, tag_to_index, UNIQUE_TAGS):
    """
    Process predictions from attention LSTM to one-hot encoding format
    :param train_pred:
    :param y_train:
    :return:
    """
    predicted_overall, true_overall, predicted_scores_overall = [], [], []
    for prediction in train_pred.cpu().data.numpy():
        predicted_tags = np.zeros(UNIQUE_TAGS - 3)
        predicted_scores = np.zeros(UNIQUE_TAGS - 3)
        predicted_idxs = np.argmax(prediction, axis=1)
        predicted_max = np.amax(prediction, axis=1)
        for i, idx in enumerate(predicted_idxs):
            if idx < tag_to_index['start']:
                predicted_tags[idx] = 1
                predicted_scores[idx] = predicted_max[i]
        predicted_overall.append(predicted_tags)
        predicted_scores_overall.append(predicted_scores)

    for true in y_train.cpu().data.numpy():
        true_tags = np.zeros(UNIQUE_TAGS - 3)
        for idx in true:
            if idx < tag_to_index['start']:
                true_tags[idx] = 1
        true_overall.append(true_tags)

    return predicted_overall, true_overall, predicted_scores_overall


def read_and_resize(filename):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imgbgr = cv2.imread(filename)
    imgbgr = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    imgbgr = torch.FloatTensor(imgbgr / 255.)
    return transform(imgbgr)


def eval(predicted_overall, true_overall):
    true_overall, predicted_overall = np.array(true_overall), np.array(predicted_overall)
    precision, recall = 0, 0
    precision_upper, recall_upper = 0, 0
    overall_precision, overall_recall = [0, 0], [0, 0]
    n = 0
    for j in range(true_overall.shape[1] - 3):
        if np.sum(true_overall[:, j]) > 0:
            n += 1
            recall += np.sum(true_overall[:, j] * predicted_overall[:, j]) / np.sum(true_overall[:, j])
            if np.sum(predicted_overall[:, j]) > 0:
                precision += np.sum(true_overall[:, j] * predicted_overall[:, j]) / np.sum(predicted_overall[:, j])
            overall_recall[0] = overall_recall[0] + np.sum(true_overall[:, j] * predicted_overall[:, j])
            overall_recall[1] = overall_recall[1] + np.sum(true_overall[:, j])
            overall_precision[0] = overall_precision[0] + np.sum(true_overall[:, j] * predicted_overall[:, j])
            overall_precision[1] = overall_precision[1] + np.sum(predicted_overall[:, j])

    overall_precision = overall_precision[0] / overall_precision[1]
    overall_recall = overall_recall[0] / overall_recall[1]

    return precision / n, recall / n, overall_precision, overall_recall


def f1_score(predicted_overall, true_overall):
    true_overall, predicted_overall = np.array(true_overall), np.array(predicted_overall)
    macroF1, microF1, instanceF1 = 0, 0, 0

    # macro
    n = 0
    for j in range(true_overall.shape[1] - 3):
        if np.sum(true_overall[:, j]) > 0:
            n += 1
            val = 2 * np.sum(predicted_overall[:, j] * true_overall[:, j])
            d = np.sum(predicted_overall[:, j]) + np.sum(true_overall[:, j])
            val /= d
            macroF1 += val

    # micro
    val1 = 2 * np.sum(predicted_overall * true_overall)
    val2 = np.sum(predicted_overall) + np.sum(true_overall)
    microF1 = val1 / val2

    # instance f1
    n = 0
    for i in range(true_overall.shape[0]):
        if np.sum(true_overall[i]) != 0:
            n += 1
            val = 2 * np.sum(true_overall[i] * predicted_overall[i])
            d = np.sum(true_overall[i]) + np.sum(predicted_overall[i])
            instanceF1 += val / d

    return macroF1 / n, microF1, instanceF1 / n


def save_models(ENCODER_NAME, include_negatives, encoder=None, decoder=None, model=None):
    # save models
    dir_name = f'{ENCODER_NAME}_results'
    if include_negatives:
        dir_name = 'NegativeSampling_' + dir_name
    else:
        dir_name = 'NoNegativeSampling_' + dir_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    decoder_name = f'{ENCODER_NAME}_decoder'
    encoder_name = f'{ENCODER_NAME}_encoder'
    model_name = f'{ENCODER_NAME}_cnn_model'
    if include_negatives:
        decoder_name = 'NegativeSampling_' + decoder_name
        encoder_name = 'NegativeSampling_' + encoder_name
        model_name = 'NegativeSampling_' + model_name
    else:
        decoder_name = 'NoNegativeSampling_' + decoder_name
        encoder_name = 'NoNegativeSampling_' + encoder_name
        model_name = 'NoNegativeSampling_' + model_name

    if decoder is not None:
        torch.save(decoder, os.path.join(dir_name, decoder_name))
    if encoder is not None:
        torch.save(encoder, os.path.join(dir_name, encoder_name))
    if model is not None:
        torch.save(model, os.path.join(dir_name, model_name))


def save_metrics(ENCODER_NAME, include_negatives, train_metrics=None, valid_metrics=None, test_metrics=None):
    dir_name = f'{ENCODER_NAME}_results'
    if include_negatives:
        dir_name = 'NegativeSampling_' + dir_name
    else:
        dir_name = 'NoNegativeSampling_' + dir_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if train_metrics is not None:
        file_name = 'train_metrics.pickle'
        with open(os.path.join(dir_name, file_name), 'wb') as f:
            pickle.dump(train_metrics, f)

    if valid_metrics is not None:
        file_name = 'valid_metrics.pickle'
        with open(os.path.join(dir_name, file_name), 'wb') as f:
            pickle.dump(valid_metrics, f)

    if test_metrics is not None:
        file_name = 'test_metrics.pickle'
        with open(os.path.join(dir_name, file_name), 'wb') as f:
            pickle.dump(test_metrics, f)


def analyze_mistakes(true, predicted, predicted_scores, train_set, tag_to_index, make_plots=True):
    number_of_tags = Counter()
    for img in train_set:
        for label in train_set[img]:
            number_of_tags[label] += 1

    scores, occurences = [], []
    index_to_tag = {tag_to_index[k]: k for k in tag_to_index}
    tags_scores_occurences = []
    for j in range(predicted.shape[1]):
        score = roc_auc_score(true[:, j], predicted_scores[:, j])
        tag_id = j
        tag_name = index_to_tag[j]
        occurrence = number_of_tags[index_to_tag[j]]
        tags_scores_occurences.append((tag_name, tag_id, score, occurrence))
        scores.append(score)
        occurences.append(occurrence)

    tags_scores_occurences = sorted(tags_scores_occurences, key=lambda x: x[2])

    if make_plots:
        # 10 best plots
        print('10 best predictions of the model')
        for tag_stats in tags_scores_occurences[-11:]:
            tag_name, tag_id, auc = tag_stats[0], tag_stats[1], tag_stats[2]
            fpr, tpr, thresholds = roc_curve(true[:, tag_id], predicted_scores[:, tag_id])
            plt.plot(fpr, tpr, label=f'{tag_name}: {auc}')
        plt.title('10 best predictions')
        plt.legend()
        plt.show()
        # 10 worst plots
        print('10 worst predictions of the model')
        for tag_stats in tags_scores_occurences[:11]:
            tag_name, tag_id, auc = tag_stats[0], tag_stats[1], tag_stats[2]
            fpr, tpr, thresholds = roc_curve(true[:, tag_id], predicted_scores[:, tag_id])
            plt.plot(fpr, tpr, label=f'{tag_name}: {auc}')
        plt.title('10 worst predictions')
        plt.legend()
        plt.show()

        occurences = np.array(occurences)
        scores = np.array(scores)
        idx = np.argsort(occurences)
        occurences = np.take_along_axis(occurences, idx, axis=0)
        scores = np.take_along_axis(scores, idx, axis=0)
        print('Score vs Number of samples')
        plt.plot(occurences, scores)
        plt.ylabel('AUC score')
        plt.xlabel('Number of samples')
        plt.title('Score vs Number of samples')
        plt.show()

    return tags_scores_occurences
