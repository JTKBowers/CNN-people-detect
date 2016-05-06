'''
Adapted from https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

Trains a convolutional neural network to detect people. The training set used is the INRIA person dataset (http://pascal.inrialpes.fr/data/human/).
'''

# import IN THIS ORDER - otherwise cv2 gets loaded after tensorflow,
# and tensorflow loads an incompatible internal version of libpng
# https://github.com/tensorflow/tensorflow/issues/1924
import cv2
import numpy as np
import tensorflow as tf

from Datasets.tud import loadTUD
from Datasets.INRIA import loadINRIA

import Model

if __name__ == '__main__':
    combined_dataset = loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-pedestrians') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-campus-sequence') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/TUD-Brussels') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/train-210') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/train-400') + \
          loadINRIA('/mnt/data/Datasets/pedestrians/INRIA/INRIAPerson')
    train_pos = combined_dataset.train.num_positive_examples
    train_neg = combined_dataset.train.num_negative_examples
    print(len(combined_dataset.train), 'training examples ({},{}).'.format(train_pos, train_neg))
    print(len(combined_dataset.test), 'testing examples ({},{}).'.format(combined_dataset.test.num_positive_examples, combined_dataset.test.num_negative_examples))

    combined_dataset.shuffle()
    combined_dataset.balance()

    nn_im_w = 320
    nn_im_h = 240
    with tf.Session() as sess:
        model = Model.BooleanModel()
        model.build_graph(nn_im_w, nn_im_h, sess=sess)

        print("Training...")
        model.train(combined_dataset.train)

        print("Testing...")
        test_accuracy = model.test(combined_dataset.test)

        print("test accuracy %g" % test_accuracy)

        # save model:
        # Weights
        model.save(sess, 'out/')
