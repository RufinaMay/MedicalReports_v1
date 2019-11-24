import pickle
import numpy as np
from keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam

from utils.constants import PATH_DIM_REDUCER, UNIQUE_TAGS, LR, MLC_EPOCHS, BATCH_SIZE, IMG_DIR
from utils.utils import normalize, read_and_resize
from utils.evaluation_metrics import precision_recall

from utils.constants import IMG_SHAPE


class MultilabelClassification():
    def __init__(self):
        with open(PATH_DIM_REDUCER, 'rb') as f:
            self.dim_reducer = pickle.load(f)
        self.dim_reducer.trainable = False

        with open('TAG_TO_INDEX.pickle', 'rb') as f:
            self.tag_to_index = pickle.load(f)

        self.model = self.create_model()

    def create_model(self):
        # mlc_layers = Sequential([
        #     Flatten(input_shape=(7, 7, 128)),
        #     Dense(4096),
        #     Dense(UNIQUE_TAGS),
        #     Activation('sigmoid')
        # ], name='mlc_layers')
        #
        # model = Sequential([self.dim_reducer, mlc_layers], name='MLC')
        # model.compile(optimizer=Adam(lr=LR), loss=binary_crossentropy)
        # return model

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
            Conv2D(filters=256, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=256, kernel_size=(1, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=512, kernel_size=(1, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=512, kernel_size=(1, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(UNIQUE_TAGS, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(lr=LR), loss=binary_crossentropy)

        return model

    @staticmethod
    def train_test_split(img_tag_mapping, test_size=0.2):
        n = len(img_tag_mapping)
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        train_size = np.ceil((1 - test_size) * n).astype('int32')
        train_idxs, test_idxs = idxs[:train_size], idxs[train_size:]
        train, test = {}, {}
        i = 0
        for k in img_tag_mapping:
            if i in train_idxs:
                train[k] = img_tag_mapping[k]
            else:
                test[k] = img_tag_mapping[k]
            i += 1
        return train, test

    def prepare_data(self, img_tag_mapping):
        train, test = self.train_test_split(img_tag_mapping, test_size=0.2)
        train, valid = self.train_test_split(train, test_size=0.2)

        return train, valid, test

    def batch_tags(self, img_tag_mapping):
        batch_IMGS, batch_TAGS = [], []
        b = 0
        for im_path in img_tag_mapping:
            im = read_and_resize(f'{IMG_DIR}/{im_path}.png')
            batch_IMGS.append(im)
            one_hot_tags = np.zeros(UNIQUE_TAGS)
            for tag in img_tag_mapping[im_path]:
                one_hot_tags[self.tag_to_index[tag]] = 1
            # one_hot_tags= one_hot_tags / sum(one_hot_tags)

            batch_TAGS.append(one_hot_tags)
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

        # make prediction on train set
        # train_pre_rec = precision_recall(y_train, train_pred)

        # return train_pre_rec

    def train(self, img_tag_mapping):
        for e in range(MLC_EPOCHS):
            self.train_set, self.valid_set, self.test_set = self.prepare_data(img_tag_mapping)
            steps_per_epoch = np.ceil(len(self.train_set) / BATCH_SIZE)
            validation_steps = np.ceil(len(self.valid_set) / BATCH_SIZE)

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
