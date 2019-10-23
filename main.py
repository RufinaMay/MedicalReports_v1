# file to run on colab
from utils.constants import IMG_DIR, TRAIN_IMAGES_PATHS, VALID_IMAGES_PATHS
from utils.utils import batch_from_dir
from preprocessing.chest_xray_extractor import process_all_reports

#
# # RUN AUTOENCODER
# cautoenc = ConvolutionalAutoencoder()
# cautoenc.train()
process_all_reports('data/chest_reports/ecgen-radiology')
