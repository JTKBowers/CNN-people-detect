import cv2
import numpy as np

from Datasets.tud import loadTUD
from Datasets.INRIA import loadINRIA

if __name__ == '__main__':
    grp = loadTUD('/mnt/pedestrians/tud/tud-pedestrians')

    cv2.namedWindow('Input')
    cv2.namedWindow('Output')
    input_width, input_height = 256, 256
    output_width, output_height = 200, 100
    for im, y in grp.test.iter(input_width,input_height, output_width, output_height):
        im = im.reshape((input_width,input_height, 3))
        y = y.reshape((output_height,output_width)).astype(np.uint8)
        cv2.imshow('Input',im)
        cv2.imshow('Output',y)
        k = cv2.waitKey() & 0xFF
        if k == ord('q')or k == 27:
            break
