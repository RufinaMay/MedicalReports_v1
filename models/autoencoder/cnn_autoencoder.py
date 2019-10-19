import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.optimizers import Adam
import pickle

from preprocessing.preprocessing import image_normalization_mapping
from utils.constants import LR, IMG_SHAPE, EPOCHS, IMG_DIR
from utils.utils import batch_from_dir, showInRow

"""
TO TRAIN AUTOENCODER RUN:

"""

class ConvolutionalAutoencoder():
    def __init__(self, IMAGES):
        self.X_train = IMAGES
        self.X_train_norm = image_normalization_mapping(IMAGES, 0,255,-1,1)
        self.autoencoder, self.dim_reducer = self.CreateAutoencoder(LR)

    def CreateAutoencoder(self, lr=0.001):
        input_img = Input(shape=IMG_SHAPE)
        l1 = Conv2D(filters=16, kernel_size=2, strides=(2, 2),
                    input_shape=IMG_SHAPE, activation='tanh')(input_img)
        l2 = Conv2D(filters=64, kernel_size=2, strides=(2, 2), activation='tanh')(l1)
        l3 = Conv2D(filters=128, kernel_size=2, strides=(2, 2), activation='tanh')(l2)
        l4 = Conv2D(filters=256, kernel_size=2, strides=(2, 2), activation='tanh')(l3)

        encoding = Conv2D(filters=128, kernel_size=2, strides=(2, 2))(l4)

        l5 = Conv2DTranspose(filters=512, kernel_size=2, strides=(2, 2), activation='tanh')(encoding)
        l6 = Conv2DTranspose(filters=256, kernel_size=2, strides=(2, 2), activation='tanh')(l5)
        l7 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2, 2), activation='tanh')(l6)
        l8 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2, 2), activation='tanh')(l7)

        decoded = Conv2DTranspose(filters=3, kernel_size=2, strides=(2, 2), activation='tanh')(l8)

        autoencoder = Model(inputs=input_img, outputs=decoded)
        optimizer = Adam(lr)
        autoencoder.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        dim_reducer = Model(inputs=input_img, outputs=encoding)

        return autoencoder, dim_reducer

    def visualize(self):
        idx = np.random.randint(low=0, high=self.X_train.shape[0], size = 5)
        images = self.X_train_norm[idx]
        decoded_imgs = self.autoencoder.predict(images)
        img = image_normalization_mapping(decoded_imgs, -1, 1, 0, 255).astype('uint8')
        showInRow(images)
        showInRow(img)

    def train(self):
        for epoch in range(EPOCHS):
            n = 0
            LOSS, ACC = 0., 0.
            for data in batch_from_dir(IMG_DIR):
                loss, accuracy = self.autoencoder.train_on_batch(data, data)
                n += 1
                LOSS += loss
                ACC += accuracy
            LOSS /= n
            ACC /= n
            if epoch % 10 == 0:
                self.visualize()
                print(f'epoch {epoch} loss: {LOSS} accuracy: {ACC}')

    # def save_models(self):
    #     with open(f'dim_reducer.pickle', 'wb') as f:
    #         pickle.dump(self.dim_reducer, f)
    #     with open(f'autoencoder_{self.NORMAL_CLASS}.pickle', 'wb') as f:
    #         pickle.dump(self.autoencoder, f)
    #
    # def reduce_dim_and_save(self):
    #     # REDUCE DIMENTION FOR EACH IMAGE
    #     reduced_train_normal = self.dim_reducer.predict(self.train_normal)
    #     reduced_test_normal = self.dim_reducer.predict(self.test_normal)
    #     test_mixed = np.concatenate((reduced_test_normal, reduced_anomal))
    #     labels = np.concatenate((np.ones(reduced_test_normal.shape[0]), np.zeros(reduced_anomal.shape[0])))
    #
    #     # SAVE IMAGES WITH REDUCED DIMENTIONS
    #     with open(f'train_normal_{self.NORMAL_CLASS}.pickle', 'wb') as f:
    #         pickle.dump(reduced_train_normal, f)
    #
    #     with open(f'test_mixed_{self.NORMAL_CLASS}.pickle', 'wb') as f:
    #         pickle.dump(test_mixed, f)
    #
    #     with open(f'labels_{self.NORMAL_CLASS}.pickle', 'wb') as f:
    #         pickle.dump(labels, f)