'''
Demonstrate the system by filtering a HOG detector with a CNN.
'''

import cv2
import tensorflow as tf
import numpy as np
import json, os

from Datasets.tud import load_tud
from Datasets.inria import load_inria
from Datasets.zurich import load_zurich

import Model

def basic_dataset_iterator(dataset, output_width, output_height):
    for image_path, image_width, image_height, bboxes in dataset.images:
        im = cv2.imread(image_path)
        if im is None:
            raise Exception('Image did not load!' + image_path)
        if image_width == 0 or image_height == 0:
            image_height, image_width, _ = im.shape


        w_scale = output_width/image_width
        h_scale = output_height/image_height

        bboxes = list(map(lambda x: BoundingBox.from_corners(*x), bboxes))
        for bbox in bboxes:
            bbox.rescale(w_scale, h_scale)
        yield cv2.resize(im, (output_width, output_height)), bboxes

class BoundingBox:
    def __init__(self):
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

    def from_corners(x1, y1, x2, y2):
        b = BoundingBox()
        b.x1 = x1
        b.x2 = x2
        b.y1 = y1
        b.y2 = y2
        return b

    def from_point_wh(x, y, w, h):
        b = BoundingBox()
        b.x1 = x
        b.x2 = x+w
        b.y1 = y
        b.y2 = y+h
        return b

    def normalise(self, im_w, im_h):
        self.x1 /= im_w
        self.x2 /= im_w
        self.y1 /= im_h
        self.y2 /= im_h

    def rescale(self, scale_x, scale_y):
        self.x1 = int(self.x1*scale_x)
        self.x2 = int(self.x2*scale_x)
        self.y1 = int(self.y1*scale_y)
        self.y2 = int(self.y2*scale_y)

    @property
    def width(self):
        return abs(self.x1 - self.x2)

    @property
    def height(self):
        return abs(self.y1 - self.y2)

    @property
    def centreX(self):
        return (self.x1 + self.x2)/2

    @property
    def centreY(self):
        return (self.y1 + self.y2)/2

    @property
    def area(self):
        return abs(self.x1 - self.x2)/abs(self.y1 - self.y2)

    def draw(self, im, colour=(0, 255, 0), thickness=2):
        cv2.rectangle(im, (self.x1, self.y1), (self.x2, self.y2), colour, thickness)

    def overlaps(self, other):
        '''
        Detects whether this bounding box overlaps with other.
        '''
        return (abs(self.centreX - other.centreX) * 2 < (self.width + other.width)) and (abs(self.centreY - other.centreY) * 2 < (self.height + other.height))

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
def generate_bboxes(image):
    '''
    Generates optimistic bounding boxes from the images iterator.

    Arguments:
    - image: A numpy array representing the image to detect people in (Assumes a size of 320x240).
    Yields bounding box objects
    '''

    (bboxes, confidences) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for bbox, confidence in zip(bboxes, confidences):
        yield BoundingBox.from_point_wh(*bbox), confidence


def nn_eval_image(model, image, nn_im_w, nn_im_h):
    im_out = image.copy()

    min_x, max_x = min(test_bbox.x1, test_bbox.x2), max(test_bbox.x1, test_bbox.x2)
    min_y, max_y = min(test_bbox.y1, test_bbox.y2), max(test_bbox.y1, test_bbox.y2)
    nn_im = cv2.resize(im_out[min_y:max_y, min_x:max_x], (nn_im_w, nn_im_h))

    nn_result = model.eval(nn_im.reshape([1,nn_im_w*nn_im_h*3]))

    return nn_result[0][0]

def bbox_correct(bbox, example_bboxes):
    for output_bbox in example_bboxes:
        if test_bbox.overlaps(output_bbox):
            return True
    else:
        return False

if __name__ == '__main__':
    combined_dataset = load_inria('/mnt/data/Datasets/pedestrians/INRIA/INRIAPerson')
    nn_im_w = 64
    nn_im_h = 160
    with tf.Session() as sess:
        model = Model.BooleanModel(sess)
        model.load('out/', nn_im_w, nn_im_h)

        image_count = 0
        HOG_TP_count = 0
        HOG_FP_count = 0

        NN_TP_count = 0
        NN_FP_count = 0
        NN_FN_count = 0
        for image, example_bboxes in basic_dataset_iterator(combined_dataset.test, 320, 240):
            for test_bbox, confidence in generate_bboxes(image):
                nn_confidence = nn_eval_image(model, image, nn_im_w, nn_im_h)
                reject = (nn_confidence+confidence)/2 < 0.5

                if bbox_correct(test_bbox, example_bboxes):
                    HOG_TP_count += 1
                    #if the bbox is correct and we wrongly rejected it
                    if reject:
                        NN_FN_count += 1
                    else:
                        NN_TP_count += 1
                else:
                    HOG_FP_count += 1
                    # if it is incorrect and not rejected, it is a false positive
                    if not reject:
                        NN_FP_count += 1
                    else:
                        print("False positive correctly removed!")
                        im_out = image.copy()
                        clean = image.copy()
                        for test_bbox2, con2 in generate_bboxes(image):
                            # draw bboxes that were not removed by the NN
                            nn_confidence2 = nn_eval_image(model, image, nn_im_w, nn_im_h)
                            if (nn_confidence2+con2)/2 >= 0.5:
                                test_bbox2.draw(clean)
                            test_bbox2.draw(im_out)
                        for output_bbox in example_bboxes:
                            output_bbox.draw(im_out, colour=(255,0,0))
                            output_bbox.draw(clean, colour=(255,0,0))

                        cv2.imshow("Tag", im_out)
                        cv2.imshow("cleaned", clean)
                        k = cv2.waitKey() & 0xFF
                        if k == ord('s'):
                            cv2.imwrite('nn_removed.png', im_out)
                            cv2.imwrite('nn_removed_clean.png', clean)
                        if k == ord('q') or k == 27:
                            exit()

            image_count += 1
        print("HOG    FPPI: ", HOG_FP_count/image_count)# As low as possible
        print("HOG/NN FPPI: ", NN_FP_count/image_count) # As low as possible - this is things that were incorrectly classified by the HOG, but removed by the neural network
        print("HOG    TPPI: ", HOG_TP_count/image_count)# As high as possible - this is things that were correct from the HOG
        print("HOG/NN TPPI: ", NN_TP_count/image_count) # As high as possible - this is things that were correct from the HOG and not removed by the neural network
        print("HOG/NN FNPI: ", NN_FN_count/image_count) # As low as possible - this is things that were correctly classified by the HOG, but removed by the neural network

        # optimal TPPI
        images = 0
        tp = 0
        for _, _,_, bboxes in combined_dataset.test.images:
            tp += len(bboxes)
            images += 1
        print("Optimal TTPI: ", tp/images)

        print("Speed!")
        import time
        num_images = len(combined_dataset.test)
        start = time.clock()
        for image, example_bboxes in basic_dataset_iterator(combined_dataset.test, 320, 240):
            bboxes = generate_bboxes(image)
        end = time.clock()
        print("HOG time elapsed per image:", (end-start)/num_images)

        start = time.clock()
        for image, example_bboxes in basic_dataset_iterator(combined_dataset.test, 320, 240):
            for test_bbox, confidence in generate_bboxes(image):
                nn_confidence = nn_eval_image(model, image, nn_im_w, nn_im_h)
                reject = (nn_confidence+confidence)/2 < 0.5
        end = time.clock()
        print("HOG+NN time elapsed per image:", (end-start)/num_images)
