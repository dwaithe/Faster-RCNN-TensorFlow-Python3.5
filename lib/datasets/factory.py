# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.coco import coco
from lib.datasets.custom_classes import custom_classes

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up dominic 2017
#for year in ['2015']:
 # for split in ['test', 'test-dev']:
    #name = 'coco_{}_{}'.format(year, split)
for split in ['val', 'trainval_n10','trainval_n20','trainval_n30','train_n10','train_n20','train_n30','test_n30']:
  name = 'c127dapi_class_{}'.format(split)
  __sets[name] = (lambda split=split, year='2007': custom_classes(split, '2017','data_path_extras_c127dapi_class','c127dapi_class'))
  name = 'Isabel_{}'.format(split)
  __sets[name] = (lambda split=split, year='2007': custom_classes(split, '2017','data_path_extras_Isabella','Isabel_class'))
  name = 'nucleosome_class_{}'.format(split)
  __sets[name] = (lambda split=split, year='2007': custom_classes(split, '2017','data_path_extras_nucleosome_class','nucleosome_class'))
for split in ['test_n30','test_n50','test_n75','test_n100','test_n120','test_n150','test_n180','train_n30','train_n50','train_n75','train_n100','train_n120','train_n150','train_n180']:
  name = 'MP6843phal_class_{}'.format(split)
  __sets[name] = (lambda split=split, year='2007': custom_classes(split, '2017','data_path_extras_MP6843phal_class','MP6843phal_class'))
for split in ['test_n30','test_n50','test_n75','test_n100','test_n120','test_n150','test_n180','train_n30','train_n50','train_n75','train_n100','train_n120','train_n150','train_n180']:
  name = 'MP6843phaldapi_class_{}'.format(split)
  __sets[name] = (lambda split=split, year='2007': custom_classes(split, '2017','data_path_extras_MP6843phaldapi_class','MP6843phaldapi_class'))

for split in ['train_n10','train_n30','train_n55','test_n55','test_n30']:
  name = 'peroxisome_class_{}'.format(split)
  __sets[name] = (lambda split=split, year='2007': custom_classes(split, '2017','data_path_extras_peroxisome_class','peroxisome_class'))
for split in ['train_n10','train_n30','train_n55','test_n55','test_n30']:
  name = 'peroxisome_full_class_{}'.format(split)
  __sets[name] = (lambda split=split, year='2007': custom_classes(split, '2017','data_path_extras_peroxisome_full_class','peroxisome_full_class'))


for year in ['2007', '2012']:
  for split in ['trainval_1','trainval_5','trainval_10','trainval_25','trainval_50','trainval_75']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))



def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
