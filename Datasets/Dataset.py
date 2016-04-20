import os

import cv2
import numpy as np

def normalise_bbox(bbox_str_tuple, input_width, input_height):
    '''
    Take a tuple of bounding box coordinate strings, convert them to ints, and normalise to the unit image.
    '''
    size_array = [input_width, input_height, input_width, input_height]
    return tuple(int(coord)/length for coord, length in zip(bbox_str_tuple, size_array)) # min_x, min_y, max_x, max_y

def render_bboxes_image(bboxes, im_w, im_h):
    '''
    Render the bounding boxes (bboxes) on an image of size (output_width, output_height),
    where the bounding box coordinates normalised into a space ranging from (0,0) to (1,1).
    '''
    output_image = np.zeros((im_w,im_h), dtype=np.uint8)

    for min_x, min_y, max_x, max_y in bboxes:
        pt1 = (int(min_x*im_w), int(min_y*im_h))
        pt2 = (int(max_x*im_w), int(max_y*im_h))
        cv2.rectangle(output_image,pt1, pt2, 255, cv2.FILLED)
    return output_image

class Dataset:
    def __init__(self, image_iterator, lazy_load=True, lazy_output_gen=True):
        '''
        Iterate through the image_iterator, which outputs tuples of the form
            (image path, [bounding boxes])
        where bounding boxes are of the form
            (min_x, min_y, max_x, max_y).
        For each image, add it to the dataset to be loaded later.

        Parameters:
        - image_iterator: as above.
        - lazy_load: Whether to load the images from disk when they are requested (saves memory), rather than in this constructor (saves training time).
        - lazy_output_gen: Whether to generate the output images when they are requested, rather than at load.
        '''
        self.images = list(image_iterator)
    def add_image(self, image_tuple):
        self.images.append(image_tuple)
    def iter_batches(self, im_w, im_h, batch_size=50):
        batch = ([],[])
        for image_path, bboxes in self.images:
            if len(batch) == batch_size:
                yield batch
                batch = ([],[])

            im = cv2.imread(image_path)
            if im is None:
                raise Exception('Image did not load!' + image_path)
            im = cv2.resize(cv2.imread(image_path), (im_w, im_h))
            y = render_bboxes_image(bboxes, im_w, im_h)
            batch[0].append(im)
            batch[1].append(y)
    def iter(self, im_w, im_h):
        for image_path, bboxes in self.images:
            im = cv2.imread(image_path)
            if im is None:
                raise Exception('Image did not load!' + image_path)
            im = cv2.resize(cv2.imread(image_path), (im_w, im_h))
            y = render_bboxes_image(bboxes, im_w, im_h)

            #im = cv2.bitwise_and(im, im, mask=255-y) # hide annotated people
            yield (im, y)

class DatasetGroup:
    def __init__(self, test, train, validation=None):
        if type(test) is not Dataset:
             # Python has duck typing, so test doesn't necessarily need to subclass from Dataset
            print('Passed a non-dataset object (ensure the test object implements the methods of dataset!)')
        if type(train) is not Dataset:
             # Python has duck typing, so train doesn't necessarily need to subclass from Dataset
            print('Passed a non-dataset object (ensure the train object implements the methods of dataset!)')
        if type(validation) is not Dataset and validation is not None:
             # Python has duck typing, so validation doesn't necessarily need to subclass from Dataset
            print('Passed a non-dataset object (ensure the validation object implements the methods of dataset!)')
        self.test = test
        self.train = train
        self.validation = validation
