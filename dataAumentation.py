# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

def split_digits_in_img(img_array, x_list, y_list):
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(img_filename[i])

size = (32,120)     # 圖片resize的大小

img_filenames = os.listdir('training1')
for img_filename in img_filenames:
    if '.jpg' not in img_filename:
        continue
    img = load_img('training1/{0}'.format(img_filename), color_mode='grayscale',
                   target_size=size,interpolation='bilinear')
   # img = random_brightness(img, max_delta=32./255)
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)
    

train_gen = ImageDataGenerator(
featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
samplewise_std_normalization=False,rescale=1/255.,zca_whitening=False,zca_epsilon=1e-6,
rotation_range=0.,width_shift_range=0.,height_shift_range=0.,shear_range=0.,zoom_range=0.,
channel_shift_range=0.,fill_mode='nearest',cval=0.,horizontal_flip=False,vertical_flip=False,
preprocessing_function=None,data_format=K.image_data_format())


train_generator = train_gen.flow(X,y,batch_size=1,shuffle=True,seed=None,
                                  save_to_dir=None,save_prefix='',save_format='png')
'



