# file to run on colab
from utils.constants import IMG_DIR, TRAIN_IMAGES_PATHS, VALID_IMAGES_PATHS
from utils.utils import batch_from_dir

#
# # RUN AUTOENCODER
# cautoenc = ConvolutionalAutoencoder()
# cautoenc.train()
c = 0
for i, j in batch_from_dir(IMG_DIR, TRAIN_IMAGES_PATHS):
    c+=1


print(c)
