import pickle
import numpy as np
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from utils.constants import PATH_DIM_REDUCER, UNIQUE_TAGS, LR, MLC_EPOCHS, BATCH_SIZE
from utils.utils import normalize
from utils.evaluation_metrics import precision_recall


class MultilabelClassification():
    def __init__(self):
        with open(PATH_DIM_REDUCER, 'rb') as f:
            self.dim_reducer = pickle.load(f)
        self.dim_reducer.trainable = False
        self.model = self.create_model()

    def create_model(self):
        mlc_layers = Sequential([
            Flatten(input_shape=(7, 7, 128)),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(UNIQUE_TAGS),
            Activation('softmax')
        ], name='mlc_layers')

        model = Sequential([self.dim_reducer, mlc_layers], name='MLC')
        model.compile(optimizer=Adam(lr=LR), loss=categorical_crossentropy)
        return model

    @staticmethod
    def prepare_data(images, tags):
        images = normalize(images)
        one_hot = MultiLabelBinarizer()
        one_hot_tags = one_hot.fit_transform(tags).astype('float32')
        print("tags shape ", one_hot_tags.shape)
        print("images shape", images.shape)
        for i in range(one_hot_tags.shape[0]):
            one_hot_tags[i] = one_hot_tags[i] / sum(one_hot_tags[i])
        x_train, x_test, y_train, y_test = train_test_split(images, one_hot_tags, test_size=0.2, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def batch_tags(self, images, one_hot_tags):
        """
        Creates batch from images and corresponding one hot encoded tags
        :param images: list ot numpy array of images with shape (None, 448, 448, 3)
        :param one_hot_tags: list or numpy array of one hot
        :return:
        """
        """
        IMAGES: mapping between image path and its tags
        one_hot_tags: one hot encoded tags
        """
        batch_IMGS, batch_TAGS = [], []
        b = 0
        for im, tag in zip(images, one_hot_tags):
            batch_IMGS.append(im), batch_TAGS.append(tag)
            b += 1

            if b > BATCH_SIZE:
                yield normalize(batch_IMGS), np.array(batch_TAGS)
                b = 0
                batch_IMGS, batch_TAGS = [], []
        yield batch_IMGS, np.array(batch_TAGS)

    def eval(self):
        train_batch = self.batch_tags(self.x_train, self.y_train)
        valid_batch = self.batch_tags(self.x_valid, self.y_valid)
        # make prediction on train set
        for x_train, y_train in train_batch:
            train_pred = self.model.predict_on_batch(x_train)
            break
        for x_valid, y_valid in valid_batch:
            valid_pred = self.model.predict_on_batch(x_valid)
            break

        train_pre_rec = precision_recall(y_train, train_pred)
        valid_pre_rec = precision_recall(y_valid, valid_pred)

        return train_pre_rec, valid_pre_rec



    def train(self, images, tags):
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = self.prepare_data(images,
                                                                                                             tags)

        steps_per_epoch = np.ceil(self.x_train.shape[0] / BATCH_SIZE)
        validation_steps = np.ceil(self.x_valid.shape[0] / BATCH_SIZE)

        for e in range(MLC_EPOCHS):
            train_batch = self.batch_tags(self.x_train, self.y_train)
            valid_batch = self.batch_tags(self.x_valid, self.y_valid)

            self.model.fit_generator(generator=train_batch,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=1,
                                     validation_data=valid_batch,
                                     validation_steps=validation_steps)

            # calculate evaluation metrics
            self.eval()