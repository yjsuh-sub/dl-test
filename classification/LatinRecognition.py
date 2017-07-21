from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.regularizers import l2
from trial_record import trial_record
from trial_record import savefigure
import numpy as np
from keras import optimizers
from sklearn.model_selection import train_test_split


path = 'C:/Users/boogi/image'
X = np.load('%s/LatinX.npy'%path)
Y = np.load('%s/LatinY.npy'%path)
X = X.T
X = X[:, :, :, np.newaxis]
Y = Y[0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

dic = {'trial' : 6,
'layer_num' : 64,
'layer_size' : 16,
'dense_layer_num' : 128,
'input_shape' : X[0].shape,
'output_num' : 26,
'ratio_dropout' : [0.3, 0.5, 0.4],
'reg' : [0.005, 0.003],
'opt_name' : 'Adadelta',
'loss' : 'categorical_crossentropy',
'metric' : ['accuracy'],
'activation' : ['relu', 'softmax'],
'layer_name': ['Conv2D', 'Dense'],
'epoch' : 30,
'min_batch' : 50,
'init_data' : 'Latin alphabet recognition',
'overall_layers': 12,
        }

for key in dic.keys():
    if type(dic[key]) == str:
        exec('%s="%s"' % (key, dic[key]))
    else:
        exec('%s=%s' % (key, dic[key]))

Y_train = np_utils.to_categorical(y_train, output_num)
Y_test = np_utils.to_categorical(y_test, output_num)

#print(X_train.shape, X_train.dtype) # 1170, 36, 36, 1
#print(y_train.shape, y_train.dtype, y_train[:3]) #1170,
#print(X_test.shape, X_test.dtype) # 390, 36, 36, 1
#print(y_test.shape, y_test.dtype) #360,

model = Sequential()
model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], input_shape=input_shape, padding='same', kernel_regularizer=l2(reg[0])))
model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(ratio_dropout[0]))

model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], input_shape=input_shape, padding='same', kernel_regularizer=l2(reg[1])))
model.add(Conv2D(layer_num, (layer_size, layer_size), activation=activation[0], padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(ratio_dropout[2]))

model.add(Flatten())
model.add(Dense(dense_layer_num, activation=activation[0], kernel_regularizer=l2(reg[1])))
model.add(Dropout(ratio_dropout[1]))
model.add(Dense(output_num, activation=activation[1]))

opt = optimizers.Adadelta()
model.compile(loss=loss, optimizer=opt, metrics=metric)

from time import time
ct = time()
hist = model.fit(X_train, Y_train, epochs=epoch, batch_size=min_batch, validation_data=(X_test, Y_test), verbose=2)
t = (time() - ct)

model.save('Latin_%s.hdf5'%trial)

trial_record(hist, dic=dic, ttime=t)

savefigure(hist, save_plot=True, trial=trial)

#trial_2: 0.986/0.946,
#trial_3: 0.979/0.944
#trial_5: 0.98/0.97