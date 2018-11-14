import numpy as np
from keras import models
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Convolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from autoencoder_layers import DependentDense, DePool2D
from helpers import show_representations


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    return (X_train, y_train), (X_test, y_test)


def build_model(nb_filters=32, nb_pool=2, nb_conv=3):
    model = models.Sequential()
    d = Dense(30)
    c = Convolution2D(nb_filters, (nb_conv, nb_conv), padding='same', input_shape=(28, 28, 1))
    mp =MaxPooling2D(pool_size=(nb_pool, nb_pool), padding='same')
    # =========      ENCODER     ========================
    model.add(c)
    model.add(Activation('tanh'))
    model.add(mp)
    model.add(Dropout(0.25))
    # =========      BOTTLENECK     ======================
    model.add(Flatten())
    model.add(d)
    model.add(Activation('tanh'))
    # =========      BOTTLENECK^-1   =====================
    model.add(DependentDense(nb_filters * 14 * 14, d))
    model.add(Activation('tanh'))
    model.add(Reshape((14, 14, nb_filters)))
    # =========      DECODER     =========================
    model.add(DePool2D(mp, size=(nb_pool, nb_pool)))
    model.add(Conv2DTranspose(1, (nb_conv, nb_conv), padding='same'))
    model.add(Activation('tanh'))

    return model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    model = build_model()
    if not False:
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.summary()
        model.fit(X_train, X_train, epochs=50, batch_size=512, validation_split=0.2,
                  callbacks=[EarlyStopping(patience=3)])
        model.save_weights('./conv.neuro', overwrite=True)
    else:
        model.load_weights('./conv.neuro')
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    show_representations(model, X_test)
