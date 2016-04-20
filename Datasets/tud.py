'''
Routines to handle the TUD family of datasets.

They can be acquired from
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/people-detection-pose-estimation-and-tracking/multi-cue-onboard-pedestrian-detection/
and
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/people-detection-pose-estimation-and-tracking/people-tracking-by-detection-and-people-detection-by-tracking/#c5131

They have an unknown license, but are commonly cited through academia (so should be fine?).
'''

import os, glob, re, random

from Dataset import *

def read_idl(path, idl_path):
    with open(idl_path) as idl:
        for line in idl:
            match = re.match('"(.+)": (.+)$', line)
            if match:
                filename, bboxes_raw = match.groups()
                bboxes = []
                for bbox_match in re.finditer('\((\d+), (\d+), (\d+), (\d+)\)', bboxes_raw):
                    bboxes.append(normalise_bbox(bbox_match.groups(), 720, 576)) #fix image dimensions?
                yield os.path.join(path, filename), bboxes

def TUD_iterator(path):
    # find all .idl files in the directory.
    for idl_path in glob.glob(os.path.join(path, '*.idl')):
        yield from read_idl(path, idl_path)

def loadTUD(path, test_train_segmentation_ratio=0.7):
    '''
    Loads a INRIA dataset. Call this!

    test_train_segmentation_ratio is the proportion of the images that are part of the training set.
    '''
    tud_examples = list(TUD_iterator(path))

    random.shuffle(tud_examples)

    cut = int(len(tud_examples)*test_train_segmentation_ratio)
    train_set = Dataset(tud_examples[:cut])
    test_set = Dataset(tud_examples[cut:])
    return DatasetGroup(test_set,train_set)


if __name__ == '__main__':
    grp = loadTUD('/mnt/pedestrians/tud/tud-pedestrians')

    cv2.namedWindow('Input')
    #cv2.namedWindow('Output')
    for im, y in grp.test.iter(256,256):
        cv2.imshow('Input',im)
        #cv2.imshow('Output',y)
        cv2.waitKey()
