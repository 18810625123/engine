# coding=utf-8
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
import numpy as np
import _thread
_thread._local = True

class GoogleNet():
    seed = 7
    np.random.seed(seed)

    # 卷积层和标准层组成的block
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    # Inception Block
    def Inception(x, nb_filter):
        branch1x1 = GoogleNet.Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        branch3x3 = GoogleNet.Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch3x3 = GoogleNet.Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

        branch5x5 = GoogleNet.Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch5x5 = GoogleNet.Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branchpool = GoogleNet.Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

        return x

    def build():
        inpt = Input(shape=(224, 224, 3))
        # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = GoogleNet.Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = GoogleNet.Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = GoogleNet.Inception(x, 64)  # 256
        x = GoogleNet.Inception(x, 120)  # 480
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = GoogleNet.Inception(x, 128)  # 512
        x = GoogleNet.Inception(x, 128)
        x = GoogleNet.Inception(x, 128)
        x = GoogleNet.Inception(x, 132)  # 528
        x = GoogleNet.Inception(x, 208)  # 832
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = GoogleNet.Inception(x, 208)
        x = GoogleNet.Inception(x, 256)  # 1024
        x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
        x = Dropout(0.4)(x)
        x = Dense(1000, activation='relu')(x)
        x = Dense(1000, activation='softmax')(x)
        model = Model(inpt, x, name='inception')
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.summary()
        return model

