#MLP

from keras.utils import np_utils
from time import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from keras import optimizers
from trial_record import trial_record
from trial_record import savefigure
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

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

dic = {'file_name':'emotion2013_MLP.txt',
       'trial' : 1,
'layer_num' : 64,
'layer_size' : 5,
'dense_layer_num' : 128,
'input_shape' : X_train[0].shape,
'output_num' : 7,
'ratio_dropout' : [0.3, 0.4, 0.5],
'reg' : [0.003, 0.004],
'opt_name' : 'Adadelta',
'loss' : 'categorical_crossentropy',
'metric' : ['accuracy'],
'activation' : ['relu', 'softmax'],
'layer_name': ['Conv2D', 'Dense'],
'epoch' : 30,
'min_batch' : 200,
'init_data' : 'Emotion recognition via MLP',
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

model.add(Dense(layer_num, activation=activation[0], input_shape=input_shape))
model.add(Dropout(ratio_dropout[0]))
model.add(Dense(layer_num, activation=activation[0], kernel_regularizer=l2(reg[0])))
model.add(Dropout(ratio_dropout[1]))
model.add(Dense(layer_num, activation=activation[0], kernel_regularizer=l2(reg[1])))
model.add(Dropout(ratio_dropout[2]))
model.add(Dense(layer_num, activation=activation[0], kernel_regularizer=l2(reg[1])))
model.add(Dropout(ratio_dropout[2]))
model.add(Dense(layer_num, activation=activation[0], kernel_regularizer=l2(reg[1])))
model.add(Dropout(ratio_dropout[2]))
model.add(Dense(layer_num, activation=activation[0], kernel_regularizer=l2(reg[1])))
model.add(Dropout(ratio_dropout[2]))
model.add(Flatten())
model.add(Dense(output_num, activation=activation[1]))

opt = optimizers.Adadelta()
model.compile(loss=loss, optimizer=opt, metrics=metric)

#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
ct = time()
hist = model.fit(X_train, Y_train, epochs=epoch, batch_size=min_batch, validation_data=(X_test, Y_test), verbose=2)
t = time() - ct

model.save('MLP_%s.hdf5'%trial)
"""
dic['overall_layers'] = content.count('add')
temp = 'trial %s'%trial
exec(content)
print(temp + content)
with open('MLP.txt', 'w') as f:
    f.write(temp + content)

trial_record(hist, dic=dic, ttime=t)

savefigure(hist, save_plot=True, trial=trial, file_name='MLP_')

#last: 0.86/0.62
#learning rate은 적당하고 overfitting만 증가시켜야 할 듯 인상