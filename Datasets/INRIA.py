'''
Routines to handle the INRIA pedestrian dataset.

The dataset can be acquired from http://pascal.inrialpes.fr/data/human/

It does not appear to have a license, and is provided:
'"AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.'
'''

import os, re

#from Datasets.Dataset import *
from .Dataset import *

def load_pascal_annotation(path):
    attributes = {}
    # If any encoding errors occur, use the chardet python package to determine the file encoding.
    with open(path, 'r', encoding='ISO-8859-2') as pascal_annotation_file:
        for line in pascal_annotation_file:
            line = line.strip()
            if line == '':
                continue
            if line[0] == '#':
                continue
            key = line[:line.find(':')].strip()
            value = line[line.find(':')+1:].strip()
            attributes[key] = value
        return attributes

def get_bboxes(metadata_path):
    '''
    Output a list of normalised bounding boxes from the metadata.
    '''

    metadata = load_pascal_annotation(metadata_path)
    input_image_size = metadata['Image size (X x Y x C)']
    input_width, input_height, _ = re.match('([\d]+) x ([\d]+) x ([\d]+)', input_image_size).groups()
    input_width = int(input_width)
    input_height = int(input_height)

    #bounding boxes
    bboxes = []
    #centre_regex = 'Center point on object ([\d]+)' # Head centre
    bbox_regex = 'Bounding box for object ([\d]+)'
    for key in metadata:
        match = re.match(bbox_regex, key)
        if match:
            bbox_id = match.group(1)
            bbox_coords_regex = '\(([\d]+), ([\d]+)\) \- \(([\d]+), ([\d]+)\)'
            bbox_match = re.match(bbox_coords_regex, metadata[key])
            if bbox_match:
                bbox = cast_bbox(bbox_match.groups()) # min_x, min_y, max_x, max_y
                bboxes.append(bbox)
            else:
                print('Syntax error?: BBOX coordinates regex ({}) does not match {}'.format(bbox_coords_regex, metadata[key]))
    return metadata['Image filename'][1:-1], input_width, input_height, bboxes

def INRIADataset(path, subdir):
    # Iterate over annotation files, generate their bounding boxes, and yield them
    for annotation_filename in os.listdir(os.path.join(path, subdir, 'annotations')):
         im_path, input_width, input_height, bboxes = get_bboxes(os.path.join(path, subdir, 'annotations', annotation_filename))
         im_path = os.path.join(path, im_path)
         yield im_path, input_width, input_height, bboxes

    # Now yield negative examples
    for neg in os.listdir(os.path.join(path, subdir, 'neg')):
        yield os.path.join(path, subdir, 'neg', neg),0,0,[]

def loadINRIA(path):
    '''
    Load the INRIA dataset. Call this!

    Note that this loader skips the normalised images due to them being incompatible with recent versions of libpng.
        (see http://stackoverflow.com/questions/25543930/libpng-error-idat-invalid-distance-too-far-back-in-mac-osx-10-9)
    '''
    test_set = Dataset(INRIADataset(path,'Test'))
    train_set = Dataset(INRIADataset(path,'Train'))
    return DatasetGroup(test_set, train_set)

if __name__ == '__main__':
    grp = loadINRIA('/mnt/pedestrians/INRIA/INRIAPerson/')

    cv2.namedWindow('Input')
    cv2.namedWindow('Output')
    for im, y in grp.test.iter(256,256):
        cv2.imshow('Input',im)
        #cv2.imshow('Output',y)
        cv2.waitKey()
