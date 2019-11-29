# import os
# import numpy as np
# from matplotlib import pyplot as plt
import pickle
# from utils.constants import IMG_DIR, PATH_IMG_TAG_MAPPING
# from utils.utils import read_and_resize
# from models.mlc.CheXpert_mlc import MultilabelClassification
from preprocessing.chest_xray_extractor import process_all_reports

# plt.rcParams["figure.figsize"] = (16, 10) # (w, h)
#
# with open(PATH_IMG_TAG_MAPPING, 'rb') as f:
#   IMG_TAG_MAPPING = pickle.load(f)
#
# # MLC
# MLC = MultilabelClassification()
# MLC.train()

IMG_REPORT, IMG_TAG, TAG_VOCAB, WORD_VOCAB, TAG_IMAGESNUMB = process_all_reports('data/chest_reports/ecgen-radiology')
print(len(TAG_IMAGESNUMB))
print(TAG_IMAGESNUMB.most_common(10))
print(TAG_IMAGESNUMB.most_common()[-20:])

new_tag_vocab = set()
total = 0
part = 0
for tag in TAG_IMAGESNUMB:
    total += TAG_IMAGESNUMB[tag]
    if TAG_IMAGESNUMB[tag] >= 100:
        new_tag_vocab.add(tag)
        part += TAG_IMAGESNUMB[tag]
print(part, ' ', total)
print(part/total)
print(len(new_tag_vocab))

# with open('tag_vocab_6.pickle', 'wb') as f:
#     pickle.dump(new_tag_vocab, f)