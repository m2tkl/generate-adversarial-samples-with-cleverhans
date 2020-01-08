import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize
import PyQt5


def plt_show():
    plt.tight_layout()
    plt.show()

def display_img_mnist(x):
    '''
    display mnist data
    Args:
        x (ndarray): 28*28 image data
    Returns:
    '''
    num_img = x.shape[0]
    idx = 1 # for allocation setting
    fig = plt.figure( facecolor="w", figsize = ( 10, (num_img / 10 ) + 1 ) )
    for i in range( num_img ):
        ax = fig.add_subplot( ( num_img / 10 ) + 1 , 10, idx, xticks = [], yticks = [] )
        ax.imshow( x[i].reshape( 28, 28 ), cmap='gray' )
        idx += 1
    plt_show()

def display_img_mnist_digit( x, y, digit ):
    '''
    display mnist image with the specified digit
    Args:
        x (ndarray): 28 * 28 image data_dp
        y: label of x(one-hot-label)
        digit (int): label number(0 ~ 9)
    Returns:
    '''
    index = get_index_digit( y, digit )
    display_img_mnist( x[index] )

def get_index_digit( label, digit ):
    '''
    return the index of the label specified by digit
    Args:
        y (ndarray): label(one-hot-label)
    Returns:
        index (int list): list of the index of the specified label
    '''
    return ( np.where( label[:, digit] == 1 )[0]).tolist()