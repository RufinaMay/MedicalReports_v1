import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from utils.constants import BATCH_SIZE, IMG_SHAPE
from preprocessing.preprocessing import image_normalization_mapping
import time

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

def batch_from_dir(images_dir):
    """
    images_dir: e.g. data/chest_images
    """
    batch_IMGS = []
    b = 0
    for im_path in os.listdir(images_dir):
        if 'png' not in im_path:
            continue
        im = read_and_resize(f'{images_dir}/{im_path}')
        batch_IMGS.append(im)
        b += 1
        if b > BATCH_SIZE:
            yield cv2.dnn.blobFromImages(batch_IMGS, 1 / 127.5, IMG_SHAPE[:2], mean=127.5, swapRB=True, crop=False) # np.array(batch_IMGS)
            b = 0
            batch_IMGS = []
    if len(batch_IMGS)>0:
        yield cv2.dnn.blobFromImages(batch_IMGS, 1 / 127.5, IMG_SHAPE[:2], mean=127.5, swapRB=True, crop=False) # np.array(batch_IMGS)

def read_and_resize(filename):
    imgbgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    # img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    # img_result = cv2.resize(img_result, dsize = IMG_SHAPE[:2], interpolation = cv2.INTER_AREA)
    # img_result = image_normalization_mapping(imgbgr, 0, 255, -1,1)
    return imgbgr


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