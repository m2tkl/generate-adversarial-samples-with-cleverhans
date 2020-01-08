from keras.datasets import mnist
import keras
from keras import backend as K

def load_mnist_data():
    '''
    return data after preprocessiong
    x_train.shape: (60,000, 784)
    y_train.shape: (60,000, 10)
    x_test.shape: (10,000, 784)
    y_test.shape: (10,000, 10)
    Args:
    Returns:
        x_train, y_train: train data(60,000)
        x_test, y_test: test data(10,000)
    '''
    num_classes = 10
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def load_mnist_data_conv():
    '''
    return data after preprocessiong
    x_train.shape: (60,000, 28, 28, 1)
    y_train.shape: (60,000, 10)
    x_test.shape: (10,000, 28, 28, 1)
    y_test.shape: (10,000, 10)
    Args:
    Returns:
        x_train, y_train: train data(60,000)
        x_test, y_test: test data(10,000)
    '''
    num_classes = 10
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print("x_train.shape: {}\ty_train.shape{}".format(x_train.shape, y_train.shape))
    print("x_test.shape: {}\ty_test.shape{}".format(x_test.shape, y_test.shape))

    (x_train_conv, y_train_conv), (x_test_conv, y_test_conv) = load_mnist_data_conv()
    print("x_train_conv.shape: {}\ty_train_conv.shape{}".format(x_train_conv.shape, y_train_conv.shape))
    print("x_test_conv.shape: {}\ty_test_conv.shape{}".format(x_test_conv.shape, y_test_conv.shape))
