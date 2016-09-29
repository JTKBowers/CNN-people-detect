import cv2
import numpy as np

from Datasets.tud import load_tud
from Datasets.inria import load_inria
from Datasets.zurich import load_zurich

if __name__ == '__main__':
    combined_dataset = load_tud('/mnt/data/Datasets/pedestrians/tud/tud-pedestrians') + \
          load_tud('/mnt/data/Datasets/pedestrians/tud/tud-campus-sequence') + \
          load_tud('/mnt/data/Datasets/pedestrians/tud/tud-crossing-sequence') + \
          load_tud('/mnt/data/Datasets/pedestrians/tud/TUD-Brussels') + \
          load_tud('/mnt/data/Datasets/pedestrians/tud/train-210') + \
          load_tud('/mnt/data/Datasets/pedestrians/tud/train-400') + \
          load_tud('/mnt/data/Datasets/pedestrians/tud/TUD-MotionPairs/positive') + \
          load_tud('/mnt/data/Datasets/pedestrians/tud/TUD-MotionPairs/negative') + \
          load_inria('/mnt/data/Datasets/pedestrians/INRIA/INRIAPerson') + \
          load_zurich('/mnt/data/Datasets/pedestrians/zurich')

    # combined_dataset.train.generate_negative_examples()
    # combined_dataset.test.generate_negative_examples()
    # combined_dataset.shuffle()
    # combined_dataset.balance()

    num_train = len(combined_dataset.train)
    num_pos = combined_dataset.train.num_positive_examples
    num_neg = combined_dataset.train.num_negative_examples
    print('{} training examples ({}+, {}-)'.format(num_train, num_pos, num_neg))

    num_test = len(combined_dataset.test)
    num_pos = combined_dataset.test.num_positive_examples
    num_neg = combined_dataset.test.num_negative_examples
    print('{} test examples ({}+, {}-)'.format(num_test, num_pos, num_neg))

    # Iterate over people
    for im, class_ in combined_dataset.train.iter_people():
        cv2.imshow('Output',im)
        k = cv2.waitKey() & 0xFF
        if k == ord('q')or k == 27:
            break
