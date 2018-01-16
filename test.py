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
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',), 'vgg16+test': ('voc_2007_test',)}

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

    imdb = combined_roidb(dataset)


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

