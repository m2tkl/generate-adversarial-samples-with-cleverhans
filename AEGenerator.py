from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import CarliniWagnerL2

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import os,sys
import numpy as np

sys.path.append( '../' )
import mnist_data
import utils
from model import mnist_dense

class AEGenerator:
    def __init__(self, model):
        self.clever_model = KerasModelWrapper( model )

    def create_adversarial_examples( self, data, session, AE_type, AE_option ):
        x = tf.placeholder( tf.float32, shape=( None, 784 ) )
        y = tf.placeholder( tf.float32, shape=( None, 10 ) )
        adv_model = AE_type( self.clever_model, sess=session )
        adv_gen = adv_model.generate(x, **AE_option )
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            adv_data = sess.run( adv_gen, feed_dict = { x: data } )
        return adv_data

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist_data.load_mnist_data()

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

    directory_name = "./model/data_hdf5/"
    model_name = "mnist_dense.hdf5"
    file_name = directory_name + model_name

    if os.path.exists( file_name ):
        test_model = load_model( file_name )
        print( "load trained model" )
    else:
        test_model = mnist_dense.train_model(
            x_train[:54000], y_train[:54000],
            x_test, y_test,
            batch_size=128,
            epochs=10
        )
        test_model.save( file_name )

    AEGen = AEGenerator( test_model )

    FGSM_params = {
        'eps': 0.15,
        'clip_min': 0.0,
        'clip_max': 1.0,
    }
    data_fgsm = AEGen.create_adversarial_examples( x_train[:10], session, FastGradientMethod, FGSM_params )
    utils.display_img_mnist( data_fgsm )

    BIM_params = {
        'eps_iter': 0.1,
        'nb_iter': 1,
        'clip_min': 0.0,
        'clip_max': 1.0,
    }
    data_bim = AEGen.create_adversarial_examples( x_train[:10], session, BasicIterativeMethod, BIM_params )
    utils.display_img_mnist( data_bim )

    CW_params = {
        'max_iterations': 100,
        'learning_rate': 0.01,
        'batch_size': 1,
        'confidence': 0.01,
        'clip_min': 0.0,
        'clip_max': 1.0,
    }
    data_cw = AEGen.create_adversarial_examples( x_train[:10], session, CarliniWagnerL2, CW_params )
    utils.display_img_mnist( data_cw )

    DP_params = {
        'nb_candidate': 10,
        'clip_min': 0.0,
        'clip_max': 1.0,
    }
    data_dp = AEGen.create_adversarial_examples( x_train[:10], session, DeepFool, DP_params )
    utils.display_img_mnist( data_dp )
