import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.optimizers import Adam
import pickle

from utils.constants import LR, IMG_SHAPE, EPOCHS, IMG_DIR, TRAIN_IMAGES_PATHS, TEST_IMAGES_PATHS, VALID_IMAGES_PATHS, \
    STEPS_PER_EPOCH, VALIDATION_STEPS
from utils.utils import batch_nolabels_from_dir, showInRow

"""
TO TRAIN AUTOENCODER RUN:

cautoenc = ConvolutionalAutoencoder(IMAGES, None, None)
cautoenc.train()
cautoenc.test()
cautoenc.save()
"""


class ConvolutionalAutoencoder():
    def __init__(self, IMAGES, autoencoder=None, dim_reducer=None):
        """
        :param IMAGES: numpy array of Images for showing intermediate results of training with shape (?,448,448,3)
        :param autoencoder: if there is already trained autoencoder model it can be load, if it is None autoencoder
                model will be created
        :param dim_reducer: if there is already trained dim_reducer model it can be load, if it is None dim_reducer
                will be created
        The autoencoder model and dim_reducer will be created if not passed.
        Also batch loaders for train, validation and test sets are identified
        """
        self.IMAGES = IMAGES
        if autoencoder is not None and dim_reducer is not None:
            self.autoencoder = autoencoder
            self.dim_reducer = dim_reducer
        else:
            self.autoencoder, self.dim_reducer = self.CreateAutoencoder(LR)

    def CreateAutoencoder(self, lr=0.001):
        """
        Creates autoencoder and dim_reducer models
        :param lr: learning rate for training the model
        :return: assembled autoencoder, dim_reducer
        """
        input_img = Input(shape=IMG_SHAPE)
        l1 = Conv2D(filters=512, kernel_size=4, strides=(2, 2), input_shape=IMG_SHAPE, activation='tanh',
                    padding='same')(input_img)
        l2 = Conv2D(filters=512, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l1)
        l3 = Conv2D(filters=256, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l2)
        l4 = Conv2D(filters=128, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l3)
        l5 = Conv2D(filters=128, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l4)

        encoding = Conv2D(filters=128, kernel_size=2, strides=(2, 2))(l5)

        l6 = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(encoding)
        l7 = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l6)
        l8 = Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l7)
        l9 = Conv2DTranspose(filters=512, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l8)
        l10 = Conv2DTranspose(filters=512, kernel_size=4, strides=(2, 2), activation='tanh', padding='same')(l9)

        decoded = Conv2DTranspose(filters=3, kernel_size=2, strides=(2, 2), activation='tanh')(l10)

        autoencoder = Model(inputs=input_img, outputs=decoded)
        optimizer = Adam(lr)
        autoencoder.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        dim_reducer = Model(inputs=input_img, outputs=encoding)

        autoencoder.summary()

        return autoencoder, dim_reducer

    def visualize(self):
        """
        Visualize results of encoding decoding 5 random images from self.IMAGES
        In shows first original images, then decoded images
        :return:
        """
        idx = np.random.randint(low=0, high=self.IMAGES.shape[0], size=5)
        images = self.IMAGES[idx]
        images_norm = (np.array(images) - 127.5) / 127.5

        decoded_imgs = self.autoencoder.predict(images_norm)
        img = (np.array(decoded_imgs) * 127.5) + 127.5

        showInRow(images)
        showInRow(img.astype('uint8'))

    def train(self):
        """
        Perform training of autoencoder
        :return: trained autoencoder
        """
        for e in range(EPOCHS):
            train_batch = batch_nolabels_from_dir(IMG_DIR, TRAIN_IMAGES_PATHS)
            valid_batch = batch_nolabels_from_dir(IMG_DIR, VALID_IMAGES_PATHS)
            self.autoencoder.fit_generator(generator=train_batch,
                                           steps_per_epoch=STEPS_PER_EPOCH,
                                           epochs=1,
                                           verbose=1,
                                           validation_data=valid_batch,
                                           validation_steps=VALIDATION_STEPS)

            self.visualize()

    def test(self):
        """
        Assesing the autoencoder performance on test set
        :return: visualize the original images and decoded images
        """
        test_batch = batch_nolabels_from_dir(IMG_DIR, TEST_IMAGES_PATHS)
        for test_img, _ in test_batch:
            decoded = self.autoencoder.predict(test_img)
            # calculate mse between true and predicted
            test_imgss = test_img * 127.5 + 127.5
            decoded = decoded * 127.5 + 127.5
            showInRow(test_imgss.astype('uint8'))
            showInRow(decoded.astype('uint8'))

    def save(self):
        """
        Save autoencoder and dim_reducer to file
        :return:
        """
        with open('autoencoder.pickle', 'wb') as f:
            pickle.dump(self.autoencoder, f)

        with open('dim_reducer.pickle', 'wb') as f:
            pickle.dump(self.dim_reducer, f)
