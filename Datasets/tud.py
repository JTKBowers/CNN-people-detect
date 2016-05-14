'''
Routines to handle the TUD family of datasets.

They can be acquired from
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/people-detection-pose-estimation-and-tracking/multi-cue-onboard-pedestrian-detection/
and
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/people-detection-pose-estimation-and-tracking/people-tracking-by-detection-and-people-detection-by-tracking/#c5131

They have an unknown license, but are commonly cited through academia (so should be fine?).
'''

import os, glob, re, random

from .Dataset import *

def read_idl(path, idl_path):
    with open(idl_path) as idl:
        for line in idl:
            match = re.match('"(.+)"(: .*)?.*[;.]$', line.strip())
            if match:
                filename, bboxes_raw = match.groups()
                bboxes = []
                if bboxes_raw is not None:
                    for bbox_match in re.finditer('\((\d+), (\d+), (\d+), (\d+)\)', bboxes_raw[2:]):
                        bboxes.append(cast_bbox(bbox_match.groups())) #fix image dimensions?
                yield os.path.join(path, filename), 0, 0, bboxes #image width and height are set to 0 so that they are calculated later.
            else:
                raise Exception('IDL parsing error:'+line[:-1])

def TUD_iterator(path):
    # find all .idl files in the directory.
    for idl_path in glob.glob(os.path.join(path, '*.idl')):
        yield from read_idl(path, idl_path)

def loadTUD(path, test_train_segmentation_ratio=0.7):
    '''
    Loads a TUD dataset. Call this!

    test_train_segmentation_ratio is the proportion of the images that are part of the training set.
    '''
    tud_examples = list(TUD_iterator(path))

    cut = int(len(tud_examples)*test_train_segmentation_ratio)
    train_set = Dataset(tud_examples[:cut])
    test_set = Dataset(tud_examples[cut:])
    return DatasetGroup(test_set,train_set)
