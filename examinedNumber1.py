# -*- coding: utf-8 -*-

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter
from tensorflow.keras.callbacks import EarlyStopping


epochs = 200      #訓練的次數
img_rows = None   #驗證碼影像檔的高
img_cols = None   #驗證碼影像檔的寬
digits_in_img = 6 #驗證碼影像檔中有幾位數
size = (32,120)     # 圖片resize的大小
num_classes = 11    # 分類數目 0-9 + '.' 共11個
callback = EarlyStopping(monitor='loss', patience=5) # for earlyStopping
x_list = list()   #存所有驗證碼數字影像檔的array
y_list = list()   #存所有的驗證碼數字影像檔array代表的正確數字
x_train = list()  #存訓練用驗證碼數字影像檔的array
y_train = list()  #存訓練用驗證碼數字影像檔array代表的正確數字
x_test = list()   #存測試用驗證碼數字影像檔的array
y_test = list()   #存測試用驗證碼數字影像檔array代表的正確數字

 
def split_digits_in_img(img_array, x_list, y_list):
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(str(img_filename[i])) 
        

img_filenames = os.listdir('training1')
for img_filename in img_filenames:
    if '.jpg' not in img_filename:
        continue
    img = load_img('training1/{0}'.format(img_filename), color_mode='grayscale',target_size=size,interpolation='bilinear')
    # img = random_brightness(img, max_delta=32./255)
    contrast_image = ImageEnhance.Contrast(img).enhance(3) # inhance the constrast
    img_array = img_to_array(contrast_image)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)
    

dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        '6': 6, '7': 7, '8': 8, '9': 9, '.':10}
i=0
for y in y_list:
    y_list[i] = dict[y]
    i += 1


y_list = keras.utils.to_categorical(y_list, num_classes=num_classes)
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.4)





if os.path.isfile('cnn_model_1.h5'):
    model = models.load_model('cnn_model_1.h5')
    print('Model loaded from file.')
else:
    model = models.Sequential()
    model.add(layers.Conv2D(32 , kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols // digits_in_img, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu',kernel_regularizer='l2')) # overfitting
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    print('New model created.')
 
model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), batch_size=digits_in_img, 
          epochs=epochs, verbose=1, callbacks=[callback], 
          validation_data=(np.array(x_test), np.array(y_test)))
 
loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
 
model.save('cnn_model.h5')


