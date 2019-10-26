import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils.constants import BATCH_SIZE, IMG_SHAPE


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
    batch_imgs= []
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


def read_and_resize(filename):
    """
    Load an image in memory
    :param filename: path to image
    :return: RGB image with shape (448,448,3)
    """
    imgbgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    img_result = cv2.resize(img_result, dsize=IMG_SHAPE[:2])
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

def get_TSNE(dim_reducer, images, labels = None):
    """

    :param dim_reducer: model that reduces dimension
    :param images: np.array of images to be visualized with shape (None, 448,448,3)
    :param labels: labels of data points if needed
    :return: Plots reduced dimension using t_SNE to reduce dims
    """
    n = images.shape[0]
    reduced = dim_reducer.predict(images)
    embedded = TSNE(n_components=2).fit_transform(reduced.reshape(n,-1))
    if labels: plt.scatter(embedded[:,0], embedded[:,1], c = labels)
    else: plt.scatter(embedded[:,0], embedded[:,1])
