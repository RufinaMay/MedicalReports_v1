import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
from utils.constants import IMG_DIR, PATH_IMG_TAG_MAPPING
from utils.utils import read_and_resize
from models.mlc.mlc import MultilabelClassification

plt.rcParams["figure.figsize"] = (16, 10) # (w, h)

with open(PATH_IMG_TAG_MAPPING, 'rb') as f:
  IMG_TAG_MAPPING = pickle.load(f)

IMAGES = []
TAGS = []
for im_path in IMG_TAG_MAPPING:
  IMAGES.append(read_and_resize(f'{IMG_DIR}/{im_path}.png'))
  TAGS.append(IMG_TAG_MAPPING[im_path])


# MLC
MLC = MultilabelClassification()
MLC.train(IMAGES, TAGS)