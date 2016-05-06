import cv2
import numpy as np

from Datasets.tud import loadTUD
from Datasets.INRIA import loadINRIA
from Datasets.Zurich import loadZurich

if __name__ == '__main__':
    combined_dataset = loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-pedestrians') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/tud-campus-sequence') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/TUD-Brussels') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/train-210') + \
          loadTUD('/mnt/data/Datasets/pedestrians/tud/train-400') + \
          loadINRIA('/mnt/data/Datasets/pedestrians/INRIA/INRIAPerson') + \
          loadZurich('/mnt/data/Datasets/pedestrians/Zurich')
    combined_dataset.train.generate_negative_examples()
    combined_dataset.shuffle()
    combined_dataset.balance()

    print(len(combined_dataset.train), 'examples')
    print(combined_dataset.train.num_positive_examples, 'positive examples')
    print(combined_dataset.train.num_negative_examples, 'negative examples')

    cover_people = True

    cv2.namedWindow('Input')
    cv2.namedWindow('Output')
    input_width, input_height = 512, 512
    output_width, output_height = 100, 100
    for im, y in combined_dataset.test.iter(input_width,input_height, output_width, output_height, normalize=False):
        im = im.reshape((input_width,input_height, 3)).astype(np.uint8)
        y = y.reshape((output_height,output_width)).astype(np.uint8)

        if cover_people:
            im = cv2.bitwise_and(im, im, mask=255-cv2.resize(y, (input_width, input_height))) # hide annotated people
        cv2.imshow('Input',im)
        cv2.imshow('Output',y)
        k = cv2.waitKey() & 0xFF
        if k == ord('q')or k == 27:
            break
