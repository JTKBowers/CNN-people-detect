import cv2

import tensorflow as tf
import numpy as np

from Datasets.Dataset import batcher

from person_classification import PersonModel

from Datasets.tud import loadTUD
from Datasets.INRIA import loadINRIA
from Datasets.Zurich import loadZurich


combined_dataset = loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-pedestrians') + \
      loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-campus-sequence') + \
      loadTUD('/mnt/data/Datasets/pedestrians/tud/TUD-Brussels') + \
      loadTUD('/mnt/data/Datasets/pedestrians/tud/train-210') + \
      loadTUD('/mnt/data/Datasets/pedestrians/tud/train-400') + \
      loadINRIA('/mnt/data/Datasets/pedestrians/INRIA/INRIAPerson') + \
      loadZurich('/mnt/data/Datasets/pedestrians/zurich')

combined_dataset.train.generate_negative_examples()
combined_dataset.shuffle()
combined_dataset.balance()

train_pos = combined_dataset.train.num_positive_examples
train_neg = combined_dataset.train.num_negative_examples
print(len(combined_dataset.train), 'training examples ({},{}).'.format(train_pos, train_neg))
print(len(combined_dataset.test), 'testing examples ({},{}).'.format(combined_dataset.test.num_positive_examples, combined_dataset.test.num_negative_examples))

cover_people = False
nn_channel = 0
with tf.Session() as sess:
    model = PersonModel(sess)
    w = 64
    h = 160
    model.load('out', w, h)
    #print('%0.2f' % model.test(combined_dataset.test + combined_dataset.train))

    for im, y in batcher(combined_dataset.test.iter_people(), batch_size=1):
        y_conv = model.eval(im)
        conv_output = model.layers[2][0].eval(feed_dict={model.x: im, model.keep_prob: 1.0})[0]
        print(np.max(conv_output))
        #print(255*int(np.max(y_conv)) == np.max(y))
        print(np.max(y_conv), np.max(y))
        im = (255*im[0].reshape((h,w, 3))).astype(np.uint8)
        #y = y.reshape((h//8, w//8)).astype(np.uint8)

        if cover_people:
            im = cv2.bitwise_and(im, im, mask=255-cv2.resize(y, (w, h))) # hide annotated people
        cv2.imshow('Input',im)
        #cv2.imshow('Output',cv2.resize(y, (w, h)))
        while(True):
            conv_im = (255*conv_output[:,:,nn_channel]).astype(np.uint8)
            print("Convnet channel", nn_channel)
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
