#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import argparse
import os

import lib.config.config as cfg
from lib.utils.test import test_net
from lib.datasets import roidb as rdl_roidb
from lib.datasets.factory import get_imdb
from lib.datasets.imdb import imdb as imdb2
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
import time, os, sys
import tensorflow as tf



NETS = {'vgg16': ('vgg16_faster_rcnn_iter_5000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'voc_2007_trainval+test': ('vgg16_faster_rcnn_iter_40000.ckpt',),'voc_2007_trainval+test+Isabel':('vgg16_faster_rcnn_iter_40000.ckpt',)}
NETS['voc_2007_trainval_25'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_50'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_75'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

NETS['c127dapi_class'] =('vgg16_faster_rcnn_iter_40000.ckpt',)


NETS['voc_2007_trainval_1+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_5+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_10+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_25+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_50+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)


NETS['nucleopore_class'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['nucleopore_class_35000'] =('vgg16_faster_rcnn_iter_35000.ckpt',)
NETS['nucleopore_class_30000'] =('vgg16_faster_rcnn_iter_30000.ckpt',)
NETS['nucleopore_class_25000'] =('vgg16_faster_rcnn_iter_25000.ckpt',)
NETS['nucleopore_class_20000'] =('vgg16_faster_rcnn_iter_20000.ckpt',)
NETS['nucleopore_class_15000'] =('vgg16_faster_rcnn_iter_15000.ckpt',)
NETS['nucleopore_class_10000'] =('vgg16_faster_rcnn_iter_10000.ckpt',)
NETS['nucleopore_class_5000']  =('vgg16_faster_rcnn_iter_5000.ckpt',)



NETS['voc_2007_trainval+nucleosome_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_1+nucleosome_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_5+nucleosome_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_10+nucleosome_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_25+nucleosome_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_50+nucleosome_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)



NETS['voc_2007_trainval+MP6843phal_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_1+MP6843phal_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_5+MP6843phal_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_10+MP6843phal_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_25+MP6843phal_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_50+MP6843phal_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

NETS['MP6843phal_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phal_class_train_n50'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phal_class_train_n75'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phal_class_train_n100'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phal_class_train_n120'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phal_class_train_n150'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phal_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

NETS['MP6843phal_class_train_n180_40000'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phal_class_train_n180_35000'] =('vgg16_faster_rcnn_iter_35000.ckpt',)
NETS['MP6843phal_class_train_n180_30000'] =('vgg16_faster_rcnn_iter_30000.ckpt',)
NETS['MP6843phal_class_train_n180_25000'] =('vgg16_faster_rcnn_iter_25000.ckpt',)
NETS['MP6843phal_class_train_n180_20000'] =('vgg16_faster_rcnn_iter_20000.ckpt',)
NETS['MP6843phal_class_train_n180_15000'] =('vgg16_faster_rcnn_iter_15000.ckpt',)
NETS['MP6843phal_class_train_n180_10000'] =('vgg16_faster_rcnn_iter_10000.ckpt',)
NETS['MP6843phal_class_train_n180_5000']  =('vgg16_faster_rcnn_iter_5000.ckpt',)

NETS['glycophorinA_class_train_n80'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['glycophorinA_class_train_n80_35000'] =('vgg16_faster_rcnn_iter_35000.ckpt',)
NETS['glycophorinA_class_train_n80_30000'] =('vgg16_faster_rcnn_iter_30000.ckpt',)
NETS['glycophorinA_class_train_n80_25000'] =('vgg16_faster_rcnn_iter_25000.ckpt',)
NETS['glycophorinA_class_train_n80_20000'] =('vgg16_faster_rcnn_iter_20000.ckpt',)
NETS['glycophorinA_class_train_n80_15000'] =('vgg16_faster_rcnn_iter_15000.ckpt',)
NETS['glycophorinA_class_train_n80_10000'] =('vgg16_faster_rcnn_iter_10000.ckpt',)
NETS['glycophorinA_class_train_n80_5000']  =('vgg16_faster_rcnn_iter_5000.ckpt',)


NETS['voc_2007_trainval+MP6843phaldapi_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_1+MP6843phaldapi_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_5+MP6843phaldapi_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_10+MP6843phaldapi_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_25+MP6843phaldapi_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_50+MP6843phaldapi_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

NETS['MP6843phaldapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phaldapi_class_train_n50'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phaldapi_class_train_n75'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phaldapi_class_train_n100'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phaldapi_class_train_n120'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phaldapi_class_train_n150'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phaldapi_class_train_n180'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

NETS['MP6843phaldapi_class_train_n180_40000'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['MP6843phaldapi_class_train_n180_35000'] =('vgg16_faster_rcnn_iter_35000.ckpt',)
NETS['MP6843phaldapi_class_train_n180_30000'] =('vgg16_faster_rcnn_iter_30000.ckpt',)
NETS['MP6843phaldapi_class_train_n180_25000'] =('vgg16_faster_rcnn_iter_25000.ckpt',)
NETS['MP6843phaldapi_class_train_n180_20000'] =('vgg16_faster_rcnn_iter_20000.ckpt',)
NETS['MP6843phaldapi_class_train_n180_15000'] =('vgg16_faster_rcnn_iter_15000.ckpt',)
NETS['MP6843phaldapi_class_train_n180_10000'] =('vgg16_faster_rcnn_iter_10000.ckpt',)
NETS['MP6843phaldapi_class_train_n180_5000']  =('vgg16_faster_rcnn_iter_5000.ckpt',)

NETS['peroxisome_full_class_train_n55'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['peroxisome_full_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['peroxisome_full_class_train_n10'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

NETS['peroxisome_class_train_n55'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['peroxisome_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['peroxisome_class_train_n10'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',), 'vgg16+test': ('voc_2007_val',)}
DATASETS['c127_dapiCell'] = ('c127dapi_class_test_n30',)
DATASETS['nucleoporeCell'] = ('nucleopore_class_test_n20',)
DATASETS['MP6843phalCell'] = ('MP6843phal_class_test_n30',)
DATASETS['MP6843phaldapiCell'] = ('MP6843phaldapi_class_test_n30',)
DATASETS['peroxisomeCell'] = ('peroxisome_class_test_n30',)
DATASETS['peroxisomeFullCell'] = ('peroxisome_full_class_test_n30',)
DATASETS['glycophorinATest'] = ('glycophorinA_class_test_n80',)
def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    
    

    if imdb_names.split('+').__len__() >1:

        
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('/scratch','dwaithe','models' , 'default',demonet ,'default', NETS[demonet][0])
    print('tfmodel',tfmodel)
    if not os.path.isfile(tfmodel + '.meta'):
        
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    imdb = combined_roidb(DATASETS[dataset][0])


    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    #if demonet == 'vgg16' or 'voc_2007_trainval+test':
    net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    #else:
    #    raise NotImplementedError
    net.create_architecture(sess, "TEST", cfg.FLAGS2["CLASSES"].__len__(), tag='default', anchor_scales=[8, 16, 32])
    

    # start a session
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    #print ('Loading model weights from {:s}').format(args.model)

    test_net(sess, net, imdb, demonet)

