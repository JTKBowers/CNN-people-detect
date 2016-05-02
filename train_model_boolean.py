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
    return h_pool1, W_conv1, b_conv1

def build_layer_2(h_pool1):
    patch_size = 5
    input_channels = 32
    output_channels = 16 #features computed by first layer for each patch
    W_conv2 = weight_variable([patch_size, patch_size, input_channels, output_channels])
    b_conv2 = bias_variable([output_channels])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    return h_pool2, W_conv2, b_conv2

def build_layer_3(h_pool1):
    patch_size = 5
    input_channels = 16
    output_channels = 8 #features computed by first layer for each patch
    W_conv3 = weight_variable([patch_size, patch_size, input_channels, output_channels])
    b_conv3 = bias_variable([output_channels])

    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    return h_pool3, W_conv3, b_conv3

def fully_connected_layer(h_pool2, im_w, im_h):
    num_neurons = 1024
    input_channels = 8
    W_fc1 = weight_variable([im_w * im_h * input_channels, num_neurons])
    b_fc1 = bias_variable([num_neurons])

    h_pool2_flat = tf.reshape(h_pool2, [-1, im_w * im_h * input_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return h_fc1, W_fc1, b_fc1

def dropout(h_fc1):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return keep_prob, h_fc1_drop

def build_readout_layer(h_fc1_drop):
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    #y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    #y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    #y_conv=tf.nn.l2_normalize(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, dim=1)
    y_raw=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.nn.sigmoid(y_raw)
    #y_conv = (y_raw + 1)/2
    return y_conv, W_fc2, b_fc2


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

    # Move examples so the training set is 50% positive examples
    if train_pos > train_neg:
        combined_dataset.shuffle()
        num_to_remove = train_pos-train_neg
        print("Removing", num_to_remove, "positive examples")
        cutoff_index = [i for i,example in enumerate(combined_dataset.train.images) if example[3] != []][:num_to_remove][-1]
        combined_dataset.train.images[:] = [example for i, example in enumerate(combined_dataset.train.images) if (example[3] == [] and i <= cutoff_index) or i > cutoff_index]
        print(len(combined_dataset.train), combined_dataset.train.num_positive_examples, combined_dataset.train.num_negative_examples)

    if train_neg > train_pos:
        print("Removing", train_neg-train_pos, "negative examples")
    nn_im_w = 320
    nn_im_h = 240
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            x = tf.placeholder(tf.float32, shape=[None, nn_im_w*nn_im_h*3])
            y_ = tf.placeholder(tf.float32, shape=[None, 1])

            h_pool1, W_conv1, b_conv1 = build_layer_1(x, nn_im_w, nn_im_h)
            h_pool2, W_conv2, b_conv2 = build_layer_2(h_pool1)
            h_pool3, W_conv3, b_conv3 = build_layer_3(h_pool2)
            h_fc1, W_fc1, b_fc1 = fully_connected_layer(h_pool3, nn_im_w//8, nn_im_h//8) #/4 because of two 2x2 pooling layers
            keep_prob, h_fc1_drop = dropout(h_fc1)
            y_conv, W_fc2, b_fc2 = build_readout_layer(h_fc1_drop)

            mean_error =  tf.reduce_mean(tf.square(y_ - y_conv))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(mean_error)
            accuracy = (1.0 - tf.reduce_mean(tf.square(y_ - y_conv)))
        sess.run(tf.initialize_all_variables())

        print("Training...")
        combined_dataset.train.shuffle()
        batch_size = 50
        num_images = len(combined_dataset.train)
        for batch_no, batch in enumerate(combined_dataset.train.iter_batches(nn_im_w, nn_im_h, 1,1, batch_size=batch_size)):
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            if batch_no % 5 == 0:
                print("%.0f%%, training accuracy %g"%(100*batch_no*batch_size/num_images, train_accuracy))
                # r = y_conv.eval(feed_dict={x: batch[0], keep_prob: 1.0})
                # print('Guess: ',  np.round(r.flatten()))
                # print('Actual:', np.round(batch[1].flatten()))


            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("Testing...")
        cum_accuracy = 0
        num_batches = 0
        for batch in combined_dataset.test.iter_batches(nn_im_w, nn_im_h,1,1, batch_size=10):
            cum_accuracy += accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            num_batches += 1
            break
        mean_accuracy = cum_accuracy/num_batches

        print("test accuracy %g"%mean_accuracy)

        # display an image
        cv2.namedWindow('Input')
        im, y = next(combined_dataset.test.iter(nn_im_w,nn_im_h, 1, 1, normalize=False))
        y = y_conv.eval(feed_dict={x: im, keep_prob: 1.0})

        im = im.reshape((nn_im_h,nn_im_w, 3)).astype(np.uint8)
        y = (255*y).astype(np.uint8)
        cv2.imshow('Input',im)
        r = h_pool3.eval(feed_dict={x: batch[0], keep_prob: 1.0})
        r = r[0][:][:][0:3]
        print(r.shape, np.max(r), np.min(r), np.mean(r))

        print(y)
        cv2.waitKey()

        # save model:
        # Weights
        np.save('W1.npy', sess.run(W_conv1))
        np.save('W2.npy', sess.run(W_conv2))
        np.save('W3.npy', sess.run(W_conv3))
        np.save('W4.npy', sess.run(W_fc1))
        np.save('W5.npy', sess.run(W_fc2))

        # Biases
        np.save('b1.npy', sess.run(b_conv1))
        np.save('b2.npy', sess.run(b_conv2))
        np.save('b3.npy', sess.run(b_conv3))
        np.save('b4.npy', sess.run(b_fc1))
        np.save('b5.npy', sess.run(b_fc2))
