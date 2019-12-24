import pickle
from collections import Counter

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

# remove tags from train dataset that occur less than 20 times
with open('train_set.pickle', 'rb') as f:
    train = pickle.load(f)

tag_occurrences = Counter()
for img in train:
    for tag in train[img]:
        tag_occurrences[tag] += 1

tag_vocab = set()
for tag in tag_occurrences:
    if tag_occurrences[tag] >= 30:
        tag_vocab.add(tag)

print(f'total tags considered in train set {len(tag_vocab)}')

# change tag to index file according to the tags taken into account from training set
TAG_TO_INDEX = {tag: i for i, tag in enumerate(tag_vocab)}

with open('TAG_TO_INDEX.pickle', 'wb') as f:
    pickle.dump(TAG_TO_INDEX, f)
