import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.constants import BATCH_SIZE, IMG_SHAPE
from utils.constants import TRAIN_IMAGES_PATHS, TEST_IMAGES_PATHS, VALID_IMAGES_PATHS

"""
TO ITERATE OVER BATCHES DO:
for X, y in batch(IMAGES, one_hot_tags):
  ...
"""


def batch(IMAGES, one_hot_tags):
    """
    IMAGES: mapping between image path and its tags
    one_hot_tags: one hot encoded tags
    """
    batch_IMGS, batch_TAGS = [], []
    b = 0
    for im, tag in zip(IMAGES, one_hot_tags):
        batch_IMGS.append(im), batch_TAGS.append(tag)
        b += 1

        if b > BATCH_SIZE:
            yield np.array(batch_IMGS), np.array(batch_TAGS)
            b = 0
            batch_IMGS, batch_TAGS = [], []
    yield np.array(batch_IMGS), np.array(batch_TAGS)


def batch_no_labels(IMAGES):
    """
    IMAGES: mapping between image path and its tags
    """
    batch_IMGS = []
    b = 0
    for im in IMAGES:
        batch_IMGS.append(im)
        b += 1
        if b > BATCH_SIZE:
            yield np.array(batch_IMGS)
            b = 0
            batch_IMGS = []
    yield np.array(batch_IMGS)


def batch_from_dir(images_dir, images_paths):
    """
    images_dir: e.g. data/chest_images
    """
    batch_imgs= []
    b = 0
    for im_path in images_paths:
        im = read_and_resize(f'{images_dir}/{im_path}')
        batch_imgs.append(im)
        b += 1
        if b >= BATCH_SIZE:
            out = (np.array(batch_imgs) - 127.5) / 127.5
            yield (out, out)
            b = 0
            batch_imgs = []
    if len(batch_imgs) > 0:
        out = (np.array(batch_imgs) - 127.5) / 127.5
        yield (out, out)


def read_and_resize(filename):
    # print(filename)
    imgbgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    img_result = cv2.resize(img_result, dsize=IMG_SHAPE[:2], interpolation=cv2.INTER_AREA)
    # img_result = image_normalization_mapping(imgbgr, 0, 255, -1,1)
    return img_result


def showInRow(list_of_images, titles=None, disable_ticks=True):
    count = len(list_of_images)
    for idx in range(count):
        subplot = plt.subplot(1, count, idx + 1)
        if titles is not None:
            subplot.set_title(titles[idx])

        img = list_of_images[idx]
        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(img, cmap=cmap)
        if disable_ticks:
            plt.xticks([]), plt.yticks([])
    plt.show()

def get_TSNE(dim_reducer, images, labels):
    n = images.shape[0]
    reduced = dim_reducer.predict(images)
    embedded = TSNE(n_components=2).fit_transform(reduced.reshape(n,-1))

    plt.scatter(embedded[:,0], embedded[:,1], c = labels)
