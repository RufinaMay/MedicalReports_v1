import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import Counter

from utils.constants import BATCH_SIZE


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


def batch_nolabels_from_dir(images_dir, image_names):
    """
    Creates batch loader to train autoencoder. Each batch is loaded from the disk during each epoch.
    :param images_dir: path to directory where images are stored, e.g. data/chest_images
    :param image_names: name of image files in the images_dir
    :return: (image, image) pairs normalize to -1 1
    """
    batch_imgs = []
    b = 0
    for im_path in image_names:
        im = read_and_resize(f'{images_dir}/{im_path}')
        batch_imgs.append(im)
        b += 1
        if b >= BATCH_SIZE:
            out = normalize(batch_imgs)
            yield (out, out)
            b = 0
            batch_imgs = []
    if len(batch_imgs) > 0:
        out = normalize(batch_imgs)
        yield (out, out)


def read_and_resize(filename, img_shape):
    """
    Load an image in memory
    :param filename: path to image
    :param img_shape: tuple of (width, height)
    :return: RGB image with shape (448,448,3)
    """
    imgbgr = cv2.imread(filename)
    img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    img_result = cv2.resize(img_result, dsize=img_shape)
    return img_result


def showInRow(list_of_images):
    """
    Show images in a row
    :param list_of_images: list of images to be visualized
    :return: plots images in one row
    """
    count = len(list_of_images)
    for idx in range(count):
        subplot = plt.subplot(1, count, idx + 1)
        img = list_of_images[idx]
        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(img, cmap=cmap)
    plt.show()


def get_TSNE(dim_reducer, images, labels=None):
    """
    :param dim_reducer: model that reduces dimension
    :param images: np.array of images to be visualized with shape (None, 448,448,3)
    :param labels: labels of data points if needed
    :return: Plots reduced dimension using t_SNE to reduce dims
    """
    n = images.shape[0]
    reduced = dim_reducer.predict(images)
    embedded = TSNE(n_components=2).fit_transform(reduced.reshape(n, -1))
    if labels:
        plt.scatter(embedded[:, 0], embedded[:, 1], c=labels)
    else:
        plt.scatter(embedded[:, 0], embedded[:, 1])


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


def prepare_data(img_tag_mapping):
    """
    Custom split data on train, validation and test sets
    :param img_tag_mapping: dictionary of img-tags pairs
    :return: train, validation and test sets
    """



    train, test = train_test_split(img_tag_mapping, test_size=0.2)
    train, valid = train_test_split(train, test_size=0.2)

    return train, valid, test

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
