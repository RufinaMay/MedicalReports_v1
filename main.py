import pickle

from preprocessing.chest_xray_extractor import process_all_reports
from preprocessing.train_test_split import run_split

IMG_REPORT, IMG_TAG, TAG_VOCAB, WORD_VOCAB, TAG_IMAGESNUMB = process_all_reports('data/chest_reports/ecgen-radiology')

new_tag_vocab = set()
for i, tag in enumerate(TAG_IMAGESNUMB):
    if TAG_IMAGESNUMB[tag] >= 1:
        new_tag_vocab.add(tag)
print('new tag vocab', len(new_tag_vocab))
with open('tag_vocab_all.pickle', 'wb') as f:
    pickle.dump(new_tag_vocab, f)

print('after IMG tag size ', len(IMG_TAG))

run_split()
