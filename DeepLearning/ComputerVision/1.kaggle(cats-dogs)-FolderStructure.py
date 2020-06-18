import os
import pandas as pd
from keras.models import Sequential

os.chdir('E:\\Data Science\\deeplearning\\Python scripts\\kaggle-cats vs dogs')
import utils

os.getcwd()

#Prepare small/full data set by calling Utils class method
train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples = \
                    utils.preapare_small_dataset_for_flow(
                            train_dir_original='E:\\Data Science\\Data\\CatsVsDogs\\train', 
                            test_dir_original='E:\\Data Science\\Data\\CatsVsDogs\\test',
                            target_base_dir='E:\\Data Science\\Data\\CatsVsDogs\\target base dir')

#Convert all images to standard width and height
img_width, img_height = 150, 150
