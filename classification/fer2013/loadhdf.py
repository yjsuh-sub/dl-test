from keras.models import load_model
import pandas as pd
import numpy as np

def model_print():
    for i in range(1, 11):
        model = load_model('fer2013_%s.hdf5'%i)
        print('='*50)
        print('trial_%s'%i)
        print(model.summary())
        print(model.get_config())

def vgg():
    from keras.applications.vgg16 import VGG16, decode_predictions
    from keras.utils import np_utils
    path = 'C:/Anaconda2/bin/fer2013/fer2013'
    df = pd.read_csv('%s/all.csv'%path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df_train = df[df.Usage == 'Training']
    X_train = np.array(df_train.iloc[:, 2:]).astype('float32')# / 255.
    X_train = X_train.reshape(28709, 48, 48, 1)
    y_train = np.array(df_train.iloc[:, 0])
    df_test = df[df.Usage == 'PrivateTest']
    X_test = np.array(df_test.iloc[:, 2:]).astype('float32')# / 255.
    X_test = X_test.reshape(3589, 48, 48, 1)
    y_test = np.array(df_test.iloc[:, 0])

    Y_train = np_utils.to_categorical(y_train, 7)
    Y_test = np_utils.to_categorical(y_test, 7)

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

model_print()