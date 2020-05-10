# from preprocessing.chest_xray_extractor import process_all_reports
# from preprocessing.train_test_split import run_split
#
# # get all data and save to disk
# IMG_REPORT, IMG_TAG, TAG_VOCAB, WORD_VOCAB, TAG_IMAGESNUMB = process_all_reports('data/chest_reports/ecgen-radiology')
# # split on train test and validation sets
# train, valid, test = run_split(save=True, draw_distrib=True)

import pickle

with open('train_set.pickle', 'rb') as f:
    train = pickle.load(f)
with open('test_set.pickle', 'rb') as f:
    test = pickle.load(f)
with open('valid_set.pickle', 'rb') as f:
    valid = pickle.load(f)

print(len(train))
print(len(test))
print(test)
print(len(valid))
