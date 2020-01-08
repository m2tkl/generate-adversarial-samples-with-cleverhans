import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
import numpy as np
import os, sys

sys.path.append( "../" )
from mnist_data import *

def build_model():
    model = Sequential()
    model.add( Dense( 200, activation='relu', input_shape=(784,) ) )
    model.add( Dense( 60, activation='relu' ) )
    model.add( Dense( 10, activation='softmax' ) )
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'],
    )
    return model

def train_model( x_train, y_train, x_val, y_val, batch_size=1, epochs=1 ):
    '''build and train model
    Args:
        x_train, y_trian: trainig data
        x_val, y_val: validation data
        batch_size (int): batch size
        epochs (int): how many epochs
    Returns:
        model: trained model
    '''
    model = build_model()
    model.fit(
        x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        validation_data = ( x_val, y_val )
    )
    return model

if __name__ == '__main__':
    # for Reproducibility
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed( 7 )
    # GPU configulations
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads = 1,
        inter_op_parallelism_threads = 1
    )
    tf.set_random_seed( 7 )
    session = tf.Session( graph = tf.get_default_graph(), config = None )
    K.set_session( session )
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    model = train_model(
        x_train[:54000], y_train[:54000],
        x_train[54000:], y_train[54000:],
        batch_size = 128,
        epochs = 10,
    )

    score = model.evaluate( x_test, y_test, verbose = 0 )
    print( "test accuracy: {}".format( score[1] ) )

    model.save( "./mnist_dense.hdf5" )
