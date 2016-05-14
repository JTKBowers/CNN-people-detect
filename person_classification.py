# import IN THIS ORDER - otherwise cv2 gets loaded after tensorflow,
# and tensorflow loads an incompatible internal version of libpng
# https://github.com/tensorflow/tensorflow/issues/1924
import cv2
import numpy as np
import tensorflow as tf

from Datasets.Dataset import batcher

from Datasets.tud import loadTUD
from Datasets.INRIA import loadINRIA
from Datasets.Zurich import loadZurich

import Model

class PersonModel(Model.BooleanModel):
    def train(self, person_iter):
        batch_size = 50
        for batch_no, batch in enumerate(batcher(person_iter, batch_size=100)):
            train_accuracy = self.accuracy.eval(feed_dict={
                self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
            if batch_no % 5 == 0:
                print("Step %i, training accuracy %g"%(batch_no, train_accuracy))
                # r = y_conv.eval(feed_dict={self.x: batch[0], keep_prob: 1.0})
                # print('Guess: ',  np.round(r.flatten()))
                # print('Actual:', np.round(batch[1].flatten()))
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

if __name__ == '__main__':
    combined_dataset = loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-pedestrians') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-campus-sequence') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-crossing-sequence') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/TUD-Brussels') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/train-210') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/train-400') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/TUD-MotionPairs/positive') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/TUD-MotionPairs/negative') + \
          loadINRIA('/mnt/data/Datasets/pedestrians/INRIA/INRIAPerson') + \
          loadZurich('/mnt/data/Datasets/pedestrians/zurich')

    combined_dataset.train.generate_negative_examples()
    combined_dataset.shuffle()
    combined_dataset.balance()

    train_pos = combined_dataset.train.num_positive_examples
    train_neg = combined_dataset.train.num_negative_examples
    print(len(combined_dataset.train), 'training examples ({},{}).'.format(train_pos, train_neg))
    print(len(combined_dataset.test), 'testing examples ({},{}).'.format(combined_dataset.test.num_positive_examples, combined_dataset.test.num_negative_examples))
    nn_im_w = 64
    nn_im_h = 160
    with tf.Session() as sess:
        model = PersonModel(sess)
        model.build_graph(nn_im_w, nn_im_h)

        print("Training...")
        model.train(combined_dataset.train.iter_people())

        print("Testing...")
        test_accuracy, confusion_matrix = model.test(combined_dataset.test.iter_people())

        print("test accuracy %g" % test_accuracy)
        print("Test confusion matrix:", confusion_matrix)

        # save model:
        # Weights
        model.save('out/')
