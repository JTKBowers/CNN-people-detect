'''
Routines to handle the ETH-Zurich dataset.

It can be acquired from http://www.vision.ee.ethz.ch/datasets/

They have an unknown license, but are commonly cited through academia (so should be fine?).
'''

import os, glob, random
from .Dataset import *
from .tud import read_idl

def Zurich_iterator(path):
    # find all .idl files in the directory.
    for idl_path in glob.glob(os.path.join(path, 'annotations/*.idl.txt')):
        yield from read_idl(os.path.join(path, 'images/'), idl_path)

def loadZurich(path, test_train_segmentation_ratio=0.7):
    '''
    Loads a Zurich dataset. Call this!

    test_train_segmentation_ratio is the proportion of the images that are part of the training set.
    '''
    zurich_examples = list(Zurich_iterator(path))

    random.shuffle(zurich_examples)

    cut = int(len(zurich_examples)*test_train_segmentation_ratio)
    train_set = Dataset(zurich_examples[:cut])
    test_set = Dataset(zurich_examples[cut:])
    return DatasetGroup(test_set,train_set)
