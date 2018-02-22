#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.

Some additions by Dominic Waithe 2017.
Manually allow use using additional classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer






NETS = {'vgg16': ('vgg16_faster_rcnn_iter_5000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'voc_2007_trainval+test': ('vgg16_faster_rcnn_iter_40000.ckpt',),'voc_2007_trainval+test+Isabel':('vgg16_faster_rcnn_iter_40000.ckpt',)}
NETS['voc_2007_trainval_25'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_50'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_75'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval'] =('vgg16_faster_rcnn_iter_40000.ckpt',)

NETS['c127dapi_class'] =('vgg16_faster_rcnn_iter_30000.ckpt',)


NETS['voc_2007_trainval_1+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_5+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_10+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_25+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['voc_2007_trainval_50+c127dapi_class_train_n30'] =('vgg16_faster_rcnn_iter_40000.ckpt',)


NETS['nucleosome_class_40000'] =('vgg16_faster_rcnn_iter_40000.ckpt',)
NETS['nucleosome_class_35000'] =('vgg16_faster_rcnn_iter_35000.ckpt',)
NETS['nucleosome_class_30000'] =('vgg16_faster_rcnn_iter_30000.ckpt',)
NETS['nucleosome_class_25000'] =('vgg16_faster_rcnn_iter_25000.ckpt',)
NETS['nucleosome_class_20000'] =('vgg16_faster_rcnn_iter_20000.ckpt',)
NETS['nucleosome_class_15000'] =('vgg16_faster_rcnn_iter_15000.ckpt',)
NETS['nucleosome_class_10000'] =('vgg16_faster_rcnn_iter_10000.ckpt',)
NETS['nucleosome_class_5000']  =('vgg16_faster_rcnn_iter_5000.ckpt',)



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

DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',), 'vgg16+test': ('voc_2007_val',)}
DATASETS['dapiCell'] = ('c127dapi_class_test_n30',)
DATASETS['nucleosomeCell'] = ('nucleosome_class_test_n30',)
DATASETS['MP6843phalCell'] = ('MP6843phal_class_test_n30',)
DATASETS['MP6843phaldapiCell'] = ('MP6843phaldapi_class_test_n30',)

def vis_detections(im, class_name, dets, thresh=0.1):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(cfg.FLAGS2["CLASSES"][1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    #tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    tfmodel = os.path.join('/scratch','dwaithe','models' , 'default',demonet ,'default', NETS[demonet][0])
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16' or 'voc_2007_trainval+test':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", cfg.FLAGS2["CLASSES"].__len__(),
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    im_names = []
    #im_names = ['010037.jpg','010038.jpg','010039.jpg','010040.jpg','010041.jpg','010057.jpg','010058.jpg','010117.jpg','010118.jpg','010119.jpg','010180.jpg','010181.jpg','010182.jpg']
    #im_names.append('010158.jpg')
    #im_names.append('010159.jpg')
    #im_names.append('010160.jpg')
    #im_names.append('010161.jpg')
    #im_names.append('010162.jpg')
    #im_names.append('010163.jpg')
    #im_names.append('010164.jpg')
    #im_names.append('010165.jpg')
    #im_names.append('010166.jpg')
    #im_names.append('010167.jpg')
    #im_names.append('010168.jpg')
    #im_names.append('010169.jpg')
    #im_names.append('010170.jpg')
    #im_names.append('010171.jpg')
    #im_names.append('010172.jpg')
    #im_names.append('010173.jpg')
    #im_names.append('010174.jpg')
    #im_names.append('010175.jpg')
    #im_names.append('010176.jpg')
    #im_names.append('010177.jpg')
    #im_names.append('010178.jpg')
    #im_names.append('010179.jpg')
    #im_names.append('010180.jpg')
    #im_names.append('010181.jpg')
    #im_names.append('010182.jpg')
    #im_names.append('010183.jpg')
    #im_names.append('010184.jpg')
    #im_names.append('010185.jpg')
    #im_names.append('010186.jpg')
    #im_names.append('010187.jpg')
    im_names.append('010188.jpg')
    im_names.append('010189.jpg')
    im_names.append('010039.jpg')
    im_names.append('010040.jpg')
    im_names.append('110096.jpg')
    im_names.append('110097.jpg')
    im_names.append('110098.jpg')
    im_names.append('110099.jpg')
    im_names.append('110100.jpg')
 

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
