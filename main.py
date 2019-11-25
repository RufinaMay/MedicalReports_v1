import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
from utils.constants import IMG_DIR, PATH_IMG_TAG_MAPPING
from utils.utils import read_and_resize
from models.mlc.CheXpert_mlc import MultilabelClassification
from preprocessing.chest_xray_extractor import process_all_reports

plt.rcParams["figure.figsize"] = (16, 10) # (w, h)

with open(PATH_IMG_TAG_MAPPING, 'rb') as f:
  IMG_TAG_MAPPING = pickle.load(f)

# MLC
MLC = MultilabelClassification()
MLC.train()

# IMG_REPORT, IMG_TAG, TAG_VOCAB, WORD_VOCAB = process_all_reports('data/chest_reports/ecgen-radiology')
# print(len(TAG_VOCAB))
# print(len(IMG_TAG))