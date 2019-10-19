import os
from sklearn.model_selection import train_test_split

BATCH_SIZE = 8
LR = 0.001
EPOCHS = 10
IMG_SHAPE = (448, 448, 3)
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
IMG_DIR = 'data/chest_images'

TRAIN_IMAGES_PATHS = os.listdir(IMG_DIR)[:-1]
TRAIN_IMAGES_PATHS, TEST_IMAGES_PATHS = train_test_split(TRAIN_IMAGES_PATHS, test_size=TEST_SIZE, shuffle=True)
TRAIN_IMAGES_PATHS, VALID_IMAGES_PATHS = train_test_split(TRAIN_IMAGES_PATHS, test_size=VALIDATION_SIZE, shuffle=True)