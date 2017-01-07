from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

def generate_model():
    # Real-time data preprocessing and data augmentation
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Convolutional network building
    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 96, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.4)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model
