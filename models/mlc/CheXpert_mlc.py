import pickle
import numpy as np
import pandas as pd
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, LSTM, TimeDistributed
from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from utils.constants import LR, MLC_EPOCHS, BATCH_SIZE, IMG_SHAPE
from utils.utils import normalize, read_and_resize

class MultilabelClassification():
    def __init__(self):
        PATH_IMG_TAG_MAPPING = 'preprocessing/CHEXPERT_IMG_TAG.csv'
        self.IMG_TAG = pd.read_csv(PATH_IMG_TAG_MAPPING)

        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=IMG_SHAPE),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            Conv2D(filters=256, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=256, kernel_size=(1, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=512, kernel_size=(1, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(12, activation='sigmoid')
         ])

        model.compile(optimizer=Adam(lr=LR), loss=binary_crossentropy)
        model.summary()
        return model

    def prepare_data(self):
        train, test = train_test_split(self.IMG_TAG, test_size=0.2)
        train, valid = train_test_split(train, test_size=0.2)

        return train, valid, test

    def batch_tags(self, img_tag_mapping):
        batch_IMGS, batch_TAGS = [], []
        b = 0
        for im_path, tag in img_tag_mapping.values:
            print(im_path)
            im = read_and_resize(f'../train_small/patient00001_study1_view1_frontal.jpg')# ???????????????????????????????????
            batch_IMGS.append(im)
            batch_TAGS.append(tag)
            b += 1
            if b >= BATCH_SIZE:
                yield normalize(batch_IMGS), np.array(batch_TAGS)
                b = 0
                batch_IMGS, batch_TAGS = [], []
        yield normalize(batch_IMGS), np.array(batch_TAGS)

    @staticmethod
    def batch_accuracy(true, predicted):
        guessed, total = 0, 0
        for t, p in zip(true, predicted):
            t_idxs = np.where(t != 0)[0]
            n = len(t_idxs)
            p_idxs = np.argsort(p)[:n]
            for p_id in p_idxs:
                if p_id in t_idxs:
                    guessed += 1
            total += n
        return guessed, total

    def eval(self):
        # calculate accuracy
        train_batch = self.batch_tags(self.train_set)
        valid_batch = self.batch_tags(self.valid_set)

        guessed, total = 0, 0
        for x_train, y_train in train_batch:
            train_pred = self.model.predict_on_batch(x_train)
            b_true, b_total = self.batch_accuracy(y_train, train_pred)
            guessed += b_true
            total += b_total

        train_acc = guessed / total


        guessed, total = 0, 0
        for x_valid, y_valid in valid_batch:
            valid_pred = self.model.predict_on_batch(x_valid)
            b_true, b_total = self.batch_accuracy(y_valid, valid_pred)
            guessed += b_true
            total += b_total

        valid_acc = guessed / total

        return train_acc, valid_acc

    def train(self):
        self.train_set, self.valid_set, self.test_set = self.prepare_data()
        steps_per_epoch = np.ceil(len(self.train_set) / BATCH_SIZE)-1
        validation_steps = np.ceil(len(self.valid_set) / BATCH_SIZE)-1

        print(len(self.valid_set))

        for e in range(MLC_EPOCHS):
            train_batch = self.batch_tags(self.train_set)
            valid_batch = self.batch_tags(self.valid_set)

            self.model.fit_generator(generator=train_batch,
                                     steps_per_epoch=steps_per_epoch,  # steps_per_epoch
                                     epochs=1,
                                     validation_data=valid_batch,
                                     validation_steps=validation_steps)

            # calculate evaluation metrics
            acc = self.eval()
            print(acc)