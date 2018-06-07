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

NETS['voc_2007_trainval_25'] =()
NETS['voc_2007_trainval_50'] =()
NETS['voc_2007_trainval_75'] =()
NETS['voc_2007_trainval'] =()


NETS['voc_2007_trainval_1+c127dapi_class_train_n30'] =()
NETS['voc_2007_trainval_5+c127dapi_class_train_n30'] =()
NETS['voc_2007_trainval_10+c127dapi_class_train_n30'] =()
NETS['voc_2007_trainval_25+c127dapi_class_train_n30'] =()
NETS['voc_2007_trainval_50+c127dapi_class_train_n30'] =()






NETS['voc_2007_trainval+nucleosome_class_train_n30'] =()
NETS['voc_2007_trainval_1+nucleosome_class_train_n30'] =()
NETS['voc_2007_trainval_5+nucleosome_class_train_n30'] =()
NETS['voc_2007_trainval_10+nucleosome_class_train_n30'] =()
NETS['voc_2007_trainval_25+nucleosome_class_train_n30'] =()
NETS['voc_2007_trainval_50+nucleosome_class_train_n30'] =()



NETS['voc_2007_trainval+MP6843phal_class_train_n180'] =()
NETS['voc_2007_trainval_1+MP6843phal_class_train_n180'] =()
NETS['voc_2007_trainval_5+MP6843phal_class_train_n180'] =()
NETS['voc_2007_trainval_10+MP6843phal_class_train_n180'] =()
NETS['voc_2007_trainval_25+MP6843phal_class_train_n180'] =()
NETS['voc_2007_trainval_50+MP6843phal_class_train_n180'] =()



NETS['glycophorinA_class_train_n80+nucleopore_class_train_n26+c127_dapi_class_train_n30+neuroblastoma_phal_class_train_n180+eukaryote_dapi_class_train_n40+peroxisome_full_class_train_n55'] =()



NETS['voc_2007_trainval+MP6843phaldapi_class_train_n180'] =()
NETS['voc_2007_trainval_1+MP6843phaldapi_class_train_n180'] =()
NETS['voc_2007_trainval_5+MP6843phaldapi_class_train_n180'] =()
NETS['voc_2007_trainval_10+MP6843phaldapi_class_train_n180'] =()
NETS['voc_2007_trainval_25+MP6843phaldapi_class_train_n180'] =()
NETS['voc_2007_trainval_50+MP6843phaldapi_class_train_n180'] =()


NETS['glycophorinA_class_train_n80'] =()
NETS['c127_dapi_class_train_n30'] =()
NETS['nucleopore_class_train_n26'] =()
NETS['neuroblastoma_phal_class_train_n180'] =()
NETS['eukaryote_dapi_class_train_n40'] =()

NETS['peroxisome_full_class_train_n55'] =()
NETS['peroxisome_full_class_train_n30'] =()
NETS['peroxisome_full_class_train_n10'] =()

NETS['peroxisome_class_train_n55'] =()
NETS['peroxisome_class_train_n30'] =()
NETS['peroxisome_class_train_n10'] =()

DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',), 'vgg16+test': ('voc_2007_val',)}
DATASETS['c127_dapi_class_test_n30'] = ('c127_dapi_class_test_n30',)
DATASETS['nucleopore_class_test_n20'] = ('nucleopore_class_test_n20',)
DATASETS['neuroblastoma_phal_class_test_n180'] = ('neuroblastoma_phal_class_test_n30',)
DATASETS['peroxisome_class_test_n55'] = ('peroxisome_class_test_n55',)
DATASETS['peroxisome_full_class_test_n55'] = ('peroxisome_full_class_test_n55',)
DATASETS['glycophorinA_class_test_n80'] = ('glycophorinA_class_test_n80',)
DATASETS['eukaryote_dapi_class_test_n40'] = ('eukaryote_dapi_class_test_n40',)
#DATASETS['global'] = ('eukaryote_dapi_class_test_n40',)

ITERATIONS ={}
ITERATIONS['100'] =()
ITERATIONS['200'] =()
ITERATIONS['300'] =()
ITERATIONS['400'] =()
ITERATIONS['500'] =()
ITERATIONS['1000'] =()
ITERATIONS['2000'] =()
ITERATIONS['3000'] =()
ITERATIONS['4000'] =()
ITERATIONS['5000'] =()
ITERATIONS['6000'] =()
ITERATIONS['7000'] =()
ITERATIONS['8000'] =()
ITERATIONS['9000'] =()
ITERATIONS['10000'] =()
ITERATIONS['15000'] =()
ITERATIONS['20000'] =()
ITERATIONS['25000'] =()
ITERATIONS['30000'] =()
ITERATIONS['35000'] =()
ITERATIONS['40000'] =()
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
    parser.add_argument('--iteration', dest='iteration', help='model iteration to evaluate',
                        choices=ITERATIONS.keys(), default='40000')
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
    model_iteration = args.iteration
    
    model_to_load = "vgg16_faster_rcnn_iter_"+str(model_iteration)+".ckpt"
    tfmodel = os.path.join('/scratch','dwaithe','models' , 'default',demonet ,'default',model_to_load )
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

