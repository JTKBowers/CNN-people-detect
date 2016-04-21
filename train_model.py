'''
Adapted from https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

Trains a convolutional neural network to detect people. The training set used is the INRIA person dataset (http://pascal.inrialpes.fr/data/human/).
'''

# import IN THIS ORDER - otherwise cv2 gets loaded after tensorflow,
# and tensorflow loads an incompatible internal version of libpng
# https://github.com/tensorflow/tensorflow/issues/1924
import cv2
import tensorflow as tf

from Datasets.tud import loadTUD
from Datasets.INRIA import loadINRIA

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def build_layer_1(x, im_w, im_h):
    patch_size = 5
    input_channels = 3
    output_channels = 32 #features computed by first layer for each patch
    W_conv1 = weight_variable([patch_size, patch_size, input_channels, output_channels])
    b_conv1 = bias_variable([output_channels])

    num_colour_channels = 3
    x_image = tf.reshape(x, [-1,im_w,im_h,num_colour_channels])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    return h_pool1

def build_layer_2(h_pool1):
    patch_size = 5
    input_channels = 32
    output_channels = 64 #features computed by first layer for each patch
    W_conv2 = weight_variable([patch_size, patch_size, input_channels, output_channels])
    b_conv2 = bias_variable([output_channels])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    return h_pool2

def fully_connected_layer(h_pool2, im_w, im_h):
    num_neurons = 1024
    input_channels = 64
    W_fc1 = weight_variable([im_w * im_h * input_channels, num_neurons])
    b_fc1 = bias_variable([num_neurons])

    h_pool2_flat = tf.reshape(h_pool2, [-1, im_w * im_h * input_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return h_fc1

def dropout(h_fc1):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return keep_prob, h_fc1_drop

def build_readout_layer(h_fc1_drop):
    W_fc2 = weight_variable([1024, 100])
    b_fc2 = bias_variable([100])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv

if __name__ == '__main__':
    tud = loadTUD('/mnt/pedestrians/tud/tud-pedestrians') + \
          loadTUD('/mnt/pedestrians/tud/tud-campus-sequence') + \
          loadINRIA('/mnt/pedestrians/INRIA/INRIAPerson')
    print(len(tud.train), len(tud.test))

    nn_im_w = 100
    nn_im_h = 100
    nn_out_w = 10
    nn_out_h = 10
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            x = tf.placeholder(tf.float32, shape=[None, nn_im_w*nn_im_h*3])
            y_ = tf.placeholder(tf.float32, shape=[None, nn_out_w*nn_out_h])

            h_pool1 = build_layer_1(x, nn_im_w, nn_im_h)
            h_pool2 = build_layer_2(h_pool1)
            h_fc1 = fully_connected_layer(h_pool2, nn_im_w//4, nn_im_h//4) #/4 because of two 2x2 pooling layers
            keep_prob, h_fc1_drop = dropout(h_fc1)
            y_conv = build_readout_layer(h_fc1_drop) #vector of classes - length 10

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.initialize_all_variables())


        for batch_no, batch in enumerate(tud.train.iter_batches(nn_im_w, nn_im_h, nn_out_w,nn_out_h)):
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print(train_accuracy)
            print("step %d, training accuracy %g"%(batch_no, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        cum_accuracy = 0
        num_batches = 0
        for batch in tud.test.iter_batches(nn_im_w, nn_im_h,nn_out_w,nn_out_h):
            cum_accuracy += accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            num_batches += 1
        mean_accuracy = cum_accuracy/num_batches
        print("test accuracy %g"%mean_accuracy)