import os
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 8
LR = 0.001
EPOCHS = 10
IMG_SHAPE = (224, 224, 3)
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
MIN_SAMPLES = 30 #  if less than MIN_SAMPLES examples in train set we do not include those tags
IMG_DIR = 'data/chest_images'

MLC_EPOCHS = 100
PATH_IMG_TAG_MAPPING = 'IMG_TAG.pickle'
PATH_TAG_TO_INDEX = 'TAG_TO_INDEX.pickle'
PATH_ING_REPORT = 'IMG_REPORT.pickle'

epochs = 120  # number of epochs to train for (if early stopping is not triggered)
encoder_lr = 2e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
emb_dim = 32  # dimension of word embeddings
attention_dim = 64  # dimension of attention linear layers
decoder_dim = 32  # dimension of decoder RNN
dropout = 0.5