from keras.models import load_model
import numpy as np
import pandas as pd
from keras.utils import np_utils
from trial_record import savefigure
from trial_record import trial_record
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras import optimizers
from keras.regularizers import l2
from time import time
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

from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 7)
Y_test = np_utils.to_categorical(y_test, 7)

def model_print(n):
    for i in range(1, n):
        model = load_model('fer2013_%s.hdf5'%i)
        print('='*50)
        print('trial_%s'%i)

        print(model.summary())

        a = model.get_config()
        d = {}
        d['overall_layers'] = len(a)
        temp_list_layer = []
        temp_list_filter = []
        temp_list_reg = []
        for i in model.get_config():

            if i['class_name'] not in temp_list_layer:
                temp_list_layer.append(i['class_name'])

            if (i['config']['filters']) & (i['config']['filters'] not in temp_list_filter):
                temp_list_filter.append(i['config']['filters'])

            if  (i['config']['kernel_regularizer']) & (i['config']['kernel_regularizer']):
                if not (i['config']['kernel_regularizer']['config']['l1']):
                    temp_list_reg.append(i['config']['kernel_regularizer']['config']['l1'])
                elif not (i['config']['kernel_regularizer']['config']['l2']):
                    temp_list_reg.append(i['config']['kernel_regularizer']['config']['l2'])

            print(i)

        d['layer_name'] = temp_list_layer
        d['filters'] = (temp_list_filter)
        df['reg'] = temp_list_reg

def run(i):
    model = load_model('fer2013_%s.hdf5' % i)
    ct = time()
    hist = model.fit(X_train, Y_train, epochs=30, batch_size=50, validation_data=(X_test, Y_test), verbose=2)
    t = time() - ct
    dic = {'trial' : 2,
                 'file_name':'fer2013.txt',
                 'path': '',
                'layer_num' : 96,
                'layer_size' : 5,
                'dense_layer_num' : 192,
                'input_shape' : (48, 48, 1),
                'output_num' : 7,
                'ratio_dropout' : [0.3, 0.5],
                'reg' : [0.001, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0001],
                'opt_name' : 'Adadelta',
                'loss' : 'categorical_crossentropy',
                'metric' : ['accuracy'],
                'activation' : ['relu', 'softmax'],
                'layer_name': ['Conv2D', 'Dense'],
                'epoch' : 30,
                'min_batch' : 50,
                'init_data' : 'Emotion recognition_fer2013',
                 'overall_layers' : 20
            }
    trial_record(hist, dic, ttime=t)
    savefigure(hist, save_plot=True, trial=1, file_name='fer2013_')

def vgg():
    from keras.applications.vgg16 import VGG16, decode_predictions
    from keras.utils import np_utils

    model = VGG16(weights='imagenet', include_top=False)

    X =  np.empty((28709, 48, 48, 3))
    X[:, :, :, 0] = X_train[:, :, :, 0]
    X[:, :, :, 1] = X_train[:, :, :, 0]
    X[:, :, :, 2] = X_train[:, :, :, 0]
    #X_train = np.expand_dims(X_train, axis=0)
    #X_train = preprocess_input(X_train[0])
    features = model.predict(X)
    #print(features, features.shape)
    features = features.reshape(28709, -1)
    fdf = pd.DataFrame(features)
    print(fdf.describe())
    #np.save('prob', features)
    #print(decode_predictions(features))


model_print(2)
