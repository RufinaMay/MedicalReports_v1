# file to run on colab
import os
import numpy as np
from models.autoencoder.cnn_autoencoder import ConvolutionalAutoencoder
from utils.utils import read_and_resize, batch_from_dir
from preprocessing.chest_xray_extractor import unzip
from utils.utils import batch_from_dir
#
# # download the data
# from IPython.display import clear_output
# !wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz -O images.tgz
# !wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz -O reports.tgz
# clear_output()
#
# unzip('images.tgz', 'data/chest_images')
# unzip('reports.tgz', 'data/chest_reports')
#
# # load images
# IMAGES = []
# for im_path in os.listdir('data/chest_images'):
#     IMAGES.append(read_and_resize(f'data/chest_images/{im_path}'))
# IMAGES = np.array(IMAGES)
#
# # RUN AUTOENCODER
# cautoenc = ConvolutionalAutoencoder(IMAGES)
# cautoenc.train()


for i in batch_from_dir('data\chest_images'):
    print(i.shape)
