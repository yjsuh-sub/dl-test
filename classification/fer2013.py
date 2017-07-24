#CNN

from keras.utils import np_utils
from time import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from keras import optimizers
from trial_record import trial_record
from trial_record import savefigure
from keras.preprocessing.image import ImageDataGenerator

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

Y_train = np_utils.to_categorical(y_train, 7)
Y_test = np_utils.to_categorical(y_test, 7)

#print(X_train.shape, X_train.dtype)
#print(y_train.shape, y_train.dtype, y_train[:3])
#print(X_test.shape, X_test.dtype)
#print(y_test.shape, y_test.dtype)

dic = {'file_name':'emotion2013.txt',
       'trial' : 4,
'layer_num' : 32,
'layer_size' : 5,
'dense_layer_num' : 64,
'input_shape' : X_train[0].shape,
'output_num' : 7,
'ratio_dropout' : [0.5, 0.5,  0.5, 0.5],
'reg' : [0.003, 0.004, 0.005],
'opt_name' : 'Adadelta',
'loss' : 'categorical_crossentropy',
'metric' : ['accuracy'],
'activation' : ['relu', 'softmax'],
'layer_name': ['Conv2D', 'Dense'],
'epoch' : 50,
'min_batch' : 50,
'init_data' : 'Emotion recognition',
'overall_layers': 12,
        }

for key in dic.keys():
    if type(dic[key]) == str:
        exec('%s="%s"' % (key, dic[key]))
    else:
        exec('%s=%s' % (key, dic[key]))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2

np.random.seed(0)

content = """

model = Sequential()

# strong overfitting >> regularizer 수 약간 늘리거나 dropout 비율 늘리기

model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], input_shape=input_shape, padding='same', kernel_regularizer=l2(reg[0])))
model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(ratio_dropout[0]))

#model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same', kernel_regularizer=l2(reg[0])))
#model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same', kernel_regularizer=l2(reg[0])))
#model.add(MaxPooling2D())
#model.add(Dropout(ratio_dropout[1]))

#model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same', kernel_regularizer=l2(reg[1])))
#model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same', kernel_regularizer=l2(reg[1])))
#model.add(MaxPooling2D())
#model.add(Dropout(ratio_dropout[1]))

model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same', kernel_regularizer=l2(reg[2])))
model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same', kernel_regularizer=l2(reg[2])))
model.add(MaxPooling2D())
model.add(Dropout(ratio_dropout[2]))

model.add(Flatten())
model.add(Dense(dense_layer_num, activation=activation[0], kernel_regularizer=l2(0.004)))
model.add(Dropout(ratio_dropout[3]))
model.add(Dense(output_num, activation=activation[1]))

opt = optimizers.Adadelta()
model.compile(loss=loss, optimizer=opt, metrics=metric)

#datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
#datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
#ct = time()
#hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=min_batch),
                    steps_per_epoch=len(X_train) / min_batch, epochs=epoch, verbose=2, validation_data=(X_test, Y_test))
#t = time() - ct


ct = time()
hist = model.fit(X_train, Y_train, epochs=epoch, batch_size=min_batch, validation_data=(X_test, Y_test), verbose=2)
t = time() - ct

model.save('emotion_%s.hdf5'%trial)
"""
temp = 'trial %s'%trial
exec(content)
print(temp + content)
with open('test.txt', 'w') as f:
    f.write(temp + content)

trial_record(hist, dic=dic, ttime=t)

savefigure(hist, save_plot=True, trial=trial)

#last: 0.86/0.62
#learning rate은 적당하고 overfitting만 증가시켜야 할 듯 인상