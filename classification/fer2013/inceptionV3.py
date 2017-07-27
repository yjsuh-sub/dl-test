#from keras.applications.inception_v3 import InceptionV3
from keras.utils import np_utils
from time import time
import numpy as np
import pandas as pd
from trial_record import savefigure
import keras
from keras import optimizers


path = 'C:/Anaconda2/bin/fer2013/fer2013'
df = pd.read_csv('%s/all.csv'%path)
df.drop('Unnamed: 0', axis=1, inplace=True)
df_train = df[df.Usage == 'Training']
X_train = np.array(df_train.iloc[:, 2:]).astype('float32') / 255.
X_train = X_train.reshape(28709, 48, 48, 1)
y_train = np.array(df_train.iloc[:, 0])
df_test = df[df.Usage == 'PrivateTest']
X_test = np.array(df_test.iloc[:, 2:]).astype('float32') / 255.
X_test = X_test.reshape(3589, 48, 48, 1)
y_test = np.array(df_test.iloc[:, 0])

X =  np.empty((28709, 48, 48, 3))
X[:, :, :, 0] = X_train[:, :, :, 0]
X[:, :, :, 1] = X_train[:, :, :, 0]
X[:, :, :, 2] = X_train[:, :, :, 0]

Y_train = np_utils.to_categorical(y_train, 7)
Y_test = np_utils.to_categorical(y_test, 7)
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.regularizers import l2
input_img = Input(shape=(48, 48, 1))
reg = [0.002, 0.002]
x = Conv2D(64, (7, 7), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(input_img)
#x = Dense(64, activation='relu')(input_img)
x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(192, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(input_img)
#x = Dense(64, activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
x = Dropout(0.5)(x)
# inception module with dimension reduction
# Inception 시작
tower_0 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(x)

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(x)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(x)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(reg[1]))(tower_3)

output = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=1)
# inception 끝
output = Flatten()(output)
output = Dropout(0.5)(output)
predictions = Dense(7, activation='softmax')(output)
model = Model(inputs=input_img, outputs=predictions)
opt = optimizers.Adadelta()
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


ct = time()
hist = model.fit(X_train, Y_train, epochs=30, batch_size=50, validation_data=(X_test, Y_test), verbose=2)
t = time() - ct

savefigure(hist, save_plot=False, trial=1, file_name='inception_')