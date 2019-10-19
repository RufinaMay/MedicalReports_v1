import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.optimizers import Adam
import pickle

from preprocessing.preprocessing import image_normalization_mapping
from utils.constants import LR, IMG_SHAPE, EPOCHS, IMG_DIR, TRAIN_IMAGES_PATHS, TEST_IMAGES_PATHS, VALID_IMAGES_PATHS
from utils.utils import batch_from_dir, showInRow

"""
TO TRAIN AUTOENCODER RUN:

"""


class ConvolutionalAutoencoder():
    def __init__(self, IMAGES):
        self.IMAGES = IMAGES
        self.autoencoder, self.dim_reducer = self.CreateAutoencoder(LR)
        self.train_batch = batch_from_dir(IMG_DIR, TRAIN_IMAGES_PATHS)
        self.valid_batch = batch_from_dir(IMG_DIR, VALID_IMAGES_PATHS)
        self.test_batch = batch_from_dir(IMG_DIR, TEST_IMAGES_PATHS)

    def CreateAutoencoder(self, lr=0.001):
        input_img = Input(shape=IMG_SHAPE)
        l1 = Conv2D(filters=512, kernel_size=2, strides=(2, 2),
                    input_shape=IMG_SHAPE, activation='tanh')(input_img)
        l2 = Conv2D(filters=256, kernel_size=2, strides=(2, 2), activation='tanh')(l1)
        l3 = Conv2D(filters=256, kernel_size=2, strides=(2, 2), activation='tanh')(l2)
        l4 = Conv2D(filters=128, kernel_size=2, strides=(2, 2), activation='tanh')(l3)

        encoding = Conv2D(filters=128, kernel_size=2, strides=(2, 2))(l4)

        l5 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2, 2), activation='tanh')(encoding)
        l6 = Conv2DTranspose(filters=256, kernel_size=2, strides=(2, 2), activation='tanh')(l5)
        l7 = Conv2DTranspose(filters=256, kernel_size=2, strides=(2, 2), activation='tanh')(l6)
        l8 = Conv2DTranspose(filters=512, kernel_size=2, strides=(2, 2), activation='tanh')(l7)

        decoded = Conv2DTranspose(filters=3, kernel_size=2, strides=(2, 2), activation='tanh')(l8)

        autoencoder = Model(inputs=input_img, outputs=decoded)
        optimizer = Adam(lr)
        autoencoder.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        dim_reducer = Model(inputs=input_img, outputs=encoding)

        autoencoder.summary()

        return autoencoder, dim_reducer

    def visualize(self):
        idx = np.random.randint(low=0, high=self.IMAGES.shape[0], size=5)
        images = self.IMAGES[idx]
        images_norm = (np.array(images) - 127.5) / 127.5

        decoded_imgs = self.autoencoder.predict(images_norm)
        img = (np.array(decoded_imgs) * 127.5) + 127.5

        showInRow(images)
        showInRow(img.astype('uint8'))
        showInRow(decoded_imgs.astype('uint8'))

    def train(self):
        for e in range(EPOCHS):
            self.autoencoder.fit_generator(generator=self.train_batch,
                                           epochs=2,
                                           verbose=1,
                                           validation_data=self.valid_batch)

            self.visualize()

    def test(self):
        pass
