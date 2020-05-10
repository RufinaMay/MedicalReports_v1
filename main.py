from preprocessing.chest_xray_extractor import process_all_reports
from preprocessing.train_test_split import run_split

# get all data and save to disk
IMG_REPORT, IMG_TAG, TAG_VOCAB, WORD_VOCAB, TAG_IMAGESNUMB = process_all_reports('data/chest_reports/ecgen-radiology')
# split on train test and validation sets
train, valid, test = run_split(save=True, draw_distrib=True)

