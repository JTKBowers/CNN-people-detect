import os

import cv2
import numpy as np

def cast_bbox(bbox_str_tuple):
    '''
    Take a tuple of bounding box coordinate strings and convert them to ints.
    '''
    return tuple(int(coord) for coord in bbox_str_tuple) # min_x, min_y, max_x, max_y

def render_bboxes_image(bboxes, output_width, output_height, input_width, input_height):
    '''
    Render the bounding boxes (bboxes) on an image of size (output_width, output_height),
    where the bounding box coordinates exist between (0,0) and (input_width, input_height).
    '''
    output_image = np.zeros((output_height, output_width), dtype=np.uint8)

    w_scale = output_width/input_width
    h_scale = output_height/input_height
    for min_x, min_y, max_x, max_y in bboxes:
        pt1 = (int(min_x*w_scale), int(min_y*h_scale))
        pt2 = (int(max_x*w_scale), int(max_y*h_scale))
        cv2.rectangle(output_image,pt1, pt2, 255, cv2.FILLED)
    return output_image

class Dataset:
    def __init__(self, image_iterator, lazy_load=True, lazy_output_gen=True):
        '''
        Iterate through the image_iterator, which outputs tuples of the form
            (image path, image_width, image_height, [bounding boxes])
        where bounding boxes are of the form
            (min_x, min_y, max_x, max_y).
        For each image, add it to the dataset to be loaded later.
        If image width and hight are 0, they will be calculated from the image itself.

        Parameters:
        - image_iterator: as above.
        - lazy_load: Whether to load the images from disk when they are requested (saves memory), rather than in this constructor (saves training time).
        - lazy_output_gen: Whether to generate the output images when they are requested, rather than at load.
        '''
        self.images = list(image_iterator)
    def add_image(self, image_tuple):
        self.images.append(image_tuple)
    def iter_batches(self, im_w, im_h, output_w, output_height, batch_size=50):
        input_row_size = im_w*im_h*3
        output_row_size = output_w*output_height
        input_batch = np.empty((batch_size, input_row_size), dtype=np.float32) # 3 elements per pixel
        output_batch = np.empty((batch_size, output_row_size), dtype=np.float32)

        batch_index = 0
        for image_path, image_width, image_height, bboxes in self.images:
            im = cv2.imread(image_path)
            if im is None:
                raise Exception('Image did not load!' + image_path)
            if image_width == 0 or image_height == 0:
                image_height, image_width, _ = im.shape
            im = cv2.resize(im, (im_w, im_h))
            input_batch[batch_index] = im.reshape((1, input_row_size))


            y = render_bboxes_image(bboxes, output_w, output_height, image_width, image_height)
            output_batch[batch_index]= y.reshape((1, output_row_size))

            batch_index += 1
            if batch_index >= batch_size:
                batch_index = 0
                yield input_batch, output_batch

        # If there are any remaining entries
        if batch_index != 0:
            #trim the arrays and yield
            yield np.resize( input_batch, (batch_index, input_row_size)),\
                  np.resize(output_batch, (batch_index, output_row_size))

    def iter(self, im_w, im_h, output_w, output_height):
        input_row_size = im_w*im_h*3
        output_row_size = output_w*output_height
        for image_path, image_width, image_height, bboxes in self.images:
            im = cv2.imread(image_path)
            if im is None:
                raise Exception('Image did not load!' + image_path)
            if image_width == 0 or image_height == 0:
                image_height, image_width, _ = im.shape

            im = cv2.resize(im, (im_w, im_h))

            y = render_bboxes_image(bboxes, output_w, output_height, image_width, image_height)

            yield im.reshape((1, input_row_size)),\
                   y.reshape((1, output_row_size))
    def __len__(self):
        return len(self.images)
    def __add__(self, other):
        new = Dataset([])
        new.images.extend(self.images)
        new.images.extend(other.images)
        return new
    def __iadd__(self, other):
        self.images.extend(other.images)

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
    def __add__(self, other):
        return DatasetGroup(self.test + other.test, self.train + other.train)
    def __iadd__(self, other):
        self.test += other.test
        self.train += other.train
        if self.validation is not None and other.validation is not None:
            self.validation += other.validation
