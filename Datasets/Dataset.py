import os, random

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

def batcher(iterator, batch_size=50, normalize=True):
    '''
    Converts iterator into one that emits batches of items,
    where each item is inserted into a numpy array (and co-erced into an array).

    I.E. [(x1,y1), (x2,y2)] -> (np.array([x1,x2]),np.array([y1,y2])) etc
    '''
    item_input_size = None
    item_output_size = None
    input_batch = None
    output_batch = None
    batch_index = 0
    for item_input, item_output in iterator:
        if type(item_output) is bool:
            item_output = 255*np.array([item_output]).astype(np.float32)
        if input_batch is None:
            input_batch = np.empty((batch_size, item_input.size), dtype=np.float32)
            item_input_size = item_input.size
        if output_batch is None:
            item_output_size = item_output.size
            output_batch = np.empty((batch_size, item_output_size), dtype=np.float32)
        #normalize
        if normalize:
            item_input = item_input.astype(np.float32)/255
            item_output = item_output.astype(np.float32)/255
        input_batch[batch_index] = item_input.reshape((1, item_input_size))
        output_batch[batch_index]= item_output.reshape((1, item_output_size))

        batch_index += 1
        if batch_index >= batch_size:
            batch_index = 0
            yield input_batch, output_batch
    # If there are any remaining entries
    if batch_index != 0:
        #trim the arrays and yield
        yield np.resize( input_batch, (batch_index, item_input_size)),\
              np.resize(output_batch, (batch_index, item_output_size))
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
    def iter_batches(self, im_w, im_h, output_w, output_height, batch_size=50, normalize=True):
        yield from batcher(self.iter(im_w, im_h, output_w, output_height), batch_size, normalize)

    def iter(self, im_w, im_h, output_w, output_height):
        for image_path, image_width, image_height, bboxes in self.images:
            im = cv2.imread(image_path)
            if im is None:
                raise Exception('Image did not load!' + image_path)
            if image_width == 0 or image_height == 0:
                image_height, image_width, _ = im.shape

            im = cv2.resize(im, (im_w, im_h))

            y = render_bboxes_image(bboxes, output_w, output_height, image_width, image_height)

            yield im, y
    def iter_people(self, batch_size=50, person_w=60, person_h=160, generate_negatives=True):
        '''
        Iterate over the images in a dataset, and for each image yield a cropped & resized image for each person.
        '''
        num_needed_negatives = 0
        for image_path, image_width, image_height, bboxes in self.images:
            im = cv2.imread(image_path)
            if im is None:
                raise Exception('Image did not load!' + image_path)
            if image_width == 0 or image_height == 0:
                image_height, image_width, _ = im.shape

            if generate_negatives and len(bboxes) == 0:
                # Generate {num_needed_negatives} random bboxes of up to 60x160
                while num_needed_negatives > 0:
                    if image_width > 60 and image_height > 160:
                        min_x = random.randint(0, image_width-60)
                        max_x = min_x + 60
                        min_y = random.randint(0, image_height-160)
                        max_y = min_y + 160
                        yield im[min_y:max_y, min_x:max_x], False
                        num_needed_negatives -= 1
                    else:
                        break
            num_needed_negatives += len(bboxes)
            for min_x, min_y, max_x, max_y in bboxes:
                # BBOX dimensions are not necessarily in order :/
                min_x, max_x = min(min_x, max_x), max(min_x, max_x)
                min_y, max_y = min(min_y, max_y), max(min_y, max_y)
                yield cv2.resize(im[min_y:max_y, min_x:max_x], (60,160)), True
    def __len__(self):
        return len(self.images)
    def __add__(self, other):
        new = Dataset([])
        new.images.extend(self.images)
        new.images.extend(other.images)
        return new
    def __iadd__(self, other):
        self.images.extend(other.images)
    def shuffle(self):
        random.shuffle(self.images)
    def balance(self):
        '''
        Orders the images so that they alternate between positive and negative images.

        Returns any surplus images.
        '''
        iter_images = iter(self.images)
        positive_stack = []
        negative_stack = []
        want_positive = True

        new_image_list = []
        # input_complete = len(self.images) == 0
        while True:
            # Read from stacks until one is empty
            if want_positive and len(positive_stack) > 0:
                new_image_list.append(positive_stack.pop())
                want_positive = False
                continue
            if (not want_positive) and len(negative_stack) > 0:
                new_image_list.append(negative_stack.pop())
                want_positive = True
                continue
            # If the relevant stack is empty, read from input.
            try:
                image_tuple = next(iter_images)
            except StopIteration: # END OF THE LINE
                break
            _, _, _, bboxes = image_tuple
            positive = len(bboxes) != 0
            if positive == want_positive:
                new_image_list.append(image_tuple)
                want_positive = not want_positive
            else:
                if positive:
                    positive_stack.append(image_tuple)
                else:
                    negative_stack.append(image_tuple)
        if len(positive_stack) > 0:
            print("Warning: %i extra positive images have been discarded" % len(positive_stack))
        if len(negative_stack) > 0:
            print("Warning: %i extra negative images have been discarded" % len(negative_stack))
        self.images = new_image_list

        return positive_stack, negative_stack
    @property
    def num_negative_examples(self):
        return sum(map(lambda x: x[3]==[], self.images))
    @property
    def num_positive_examples(self):
        return sum(map(lambda x: x[3]!=[], self.images))

    def generate_negative_examples(self):
        '''
        Copies negative examples until there are an equal amount of positive and negative examples.
        '''
        num = self.num_positive_examples - self.num_negative_examples
        negative_examples = list(filter(lambda x: x[3] == [], self.images))
        self.images.extend(random.sample(negative_examples, num))

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
    def shuffle(self):
        self.test.shuffle()
        self.train.shuffle()
        if self.validation is not None:
            self.validation.shuffle()
    def balance(self):
        self.test.balance()
        self.train.balance()
        if self.validation is not None:
            self.validation.balance()
