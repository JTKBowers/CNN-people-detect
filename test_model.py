import cv2

import tensorflow as tf
import numpy as np

import Model

from Datasets.tud import loadTUD
from Datasets.INRIA import loadINRIA

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

cover_people = False
nn_channel = 0
with tf.Session() as sess:
    model = Model.BooleanModel()
    w = 320
    h = 240
    model.load('0.8_0.8', w, h, sess=sess)
    #print('%0.2f' % model.test(combined_dataset.test + combined_dataset.train))

    for im, y in combined_dataset.test.iter(w,h, w//8, h//8, normalize=False):
        y_conv = model.eval(im)
        conv_output = model.layers[1][0].eval(feed_dict={model.x: im, model.keep_prob: 1.0})[0]
        print(255*int(np.max(y_conv)) == np.max(y))
        im = im.reshape((h,w, 3)).astype(np.uint8)
        y = y.reshape((h//8, w//8)).astype(np.uint8)

        if cover_people:
            im = cv2.bitwise_and(im, im, mask=255-cv2.resize(y, (w, h))) # hide annotated people
        cv2.imshow('Input',im)
        cv2.imshow('Output',cv2.resize(y, (w, h)))
        while(True):
            conv_im = 255*conv_output[:,:,nn_channel].astype(np.uint8)
            print(conv_im.shape)
            cv2.imshow('Conv Output', cv2.resize(conv_im, (w, h)))
            k = cv2.waitKey() & 0xFF
            if k == ord('q') or k == 27 or k == ord(' '):
                break
            if k == ord('c'):
                cover_people = not cover_people
            if k == ord('z'):
                nn_channel -= 1
            if k == ord('a'):
                nn_channel += 1
            nn_channel %= 8
        if k == ord('q') or k == 27:
            break
