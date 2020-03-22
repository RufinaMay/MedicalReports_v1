import pickle
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt

from preprocessing.chest_xray_extractor import process_all_reports
from preprocessing.train_test_split import run_split
from models.mlc.attention_LSTM import Encoder, DecoderWithAttention, Attention
from utils.utils import apply_hierarchy, batch, f1_score, eval

# get all data and save to disk
IMG_REPORT, IMG_TAG, TAG_VOCAB, WORD_VOCAB, TAG_IMAGESNUMB = process_all_reports('data/chest_reports/ecgen-radiology')
# split on train test and validation sets
train, valid, test = run_split(save=True, draw_distrib=False)

# а теперь модельки наши
