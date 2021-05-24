# -*- coding: utf-8 -*-

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

epochs = 5       #訓練的次數
img_rows = None   #驗證碼影像檔的高
img_cols = None   #驗證碼影像檔的寬
digits_in_img = 6 #驗證碼影像檔中有幾位數
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
        y_list.append(img_filename[i])
        
img_filenames = os.listdir('training')
for img_filename in img_filenames:
    if '.png' not in img_filename:
        continue
    img = load_img('training/{0}'.format(img_filename), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)
    
    
y_list = keras.utils.to_categorical(y_list, num_classes=10)
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list)

if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
    print('Model loaded from file.')
else:
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols // digits_in_img, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(10, activation='softmax'))
    print('New model created.')
 
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), batch_size=digits_in_img, epochs=epochs, verbose=1, validation_data=(np.array(x_test), np.array(y_test)))
 
loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
 
model.save('cnn_model.h5')


