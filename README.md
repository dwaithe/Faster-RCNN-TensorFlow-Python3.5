# Faster-RCNN used for the AMCA (Automous Microscope Control Environment)
For the project, to make it compatible with cellular imaging data created for the project, I made some changes.
The training is peformed by transfer learning, using VGG16 trained on ImageNet. Changes were made to several files to make it possible to train the network on some new classes which are not in the conventional databases used for object detection.
Here are list of the changes made, though greater detail is to be found using github commit history.
- to /lib/config/config.py I added a different set of classes to which to train and predict on, with
- to /lib/datasets/factory.py I added all the variations of training and prediction datasets for new classes.
- to /lib/datasets/imdb.py I changed this so there is the option to flip images vertically for data augmentation.
- to /lib/datasets/pascal_voc.py I removed the cached file method because I change the parameters so often.
- to /lib/datasets/voc_eval.py I adapted this so it can handle upper and lower case in class names.
- to /lib/utils/minibatch.py Added flipped images to the the image blob creator.
- to /train.py Made a number of changes to reflect the need for custom classes
- I added /lib/datasets/custom_classes.py This is a altered copy of pascal_voc whereby I have made changes to support cell classes.




## Training your own model.

Running this code yourself. What needs to be done.

### Create your own dataset. 
First of all you need some data. Either download the data used for this project or create your own.

If you have your own data, this needs to placed in its own directory in the root data directory(e.g. /data). Within the new folder (e.g. data/erythroblast_dapi_class), you need a folder with the year (e.g. 2018) and then within this directory a few additional folders:
'JPEGImages' - jpg versions of your image files.
'Annotations' - with Annotations for each image file. The Annotations follow the VOC format. See here for more details.
'ImageSets/Main' - This folder contains a subdirectory called 'Main' (e.g. ImageSets/Main). Within this folder you place files which reference images for training, testing and validation (e.g. test_n180.txt). Each file contains the image names used for an operation (e.g. 013000), with one image reference per line.
'results/Main' - This is folder for the output of results and contains a subdiretory called 'Main'.

### Planning your first experiment. 

Add a class file to the root folder. Any name with the prefix "classes_" and suffix ".txt".
e.g. classes_cells.txt.
This file tells the algorithm which classes to condition the output layers of Faster-RCNN with. This does not mean the network is necessarily trained with the corresponding training data for these classes however. 

Within each file write out each class name, followed by a comma and then the directory relative to the specified data dir. After another comma include the imagesets used for this file with a comma separating each file. This should match the files located in the 'ImageSets' folder described above in the 'Create your own dataset section'.
 with a new line for each class:
e.g.
'cell - neuroblastoma phalloidin', 'neuroblastoma_phal_class/2018','test_n30','test_n50','test_n75','test_n100','test_n120','test_n150','test_n180','train_n30','train_n50','train_n75','train_n100','train_n120','train_n150','train_n180'
'cell - erythroblast dapi', 'erythroblast_dapi_class/2018','train_n80','test_n80'
'cell - c127 dapi','c127_dapi_class/2018', 'train_n10','train_n20','train_n30','test_n30'
'cell - eukaryote dapi','eukaryote_dapi_class/2018', 'train_n40','test_n40'
'cell - fibroblast nucleopore','fibroblast_nucleopore_class/2018','train_n26','test_n20'
'cell - hek peroxisome all','hek_peroxisome_all_class/2018','train_n10','train_n30','train_n55','test_n55','test_n30'

### Initalisation of training.

From the command line:
python train.py [GPU EXPERIMENT FLIP]
e.g. python train.py 0 hek_peroxisome_all_class_train_n10 1
GPU - which GPU to use (e.g. 0 or 1 or -1 for cpu)
EXPERIMENT - the dataset to train on, class followed by imageset. To train on multiple dataset simultaneously seperate each with a '+', e.g. (glycophorinA_class_train_n80+nucleopore_class_train_n26+c127_dapi_class_train_n30)
FLIP - whether to include vertical flipping during the data import and augmentation phase.


# tf-faster-rcnn

Tensorflow Faster R-CNN for Windows, mac OS X and linux by using Python 3.5 


This is the branch to compile Faster R-CNN on Windows. It is heavily inspired by the great work done [here](https://github.com/smallcorgi/Faster-RCNN_TF) and [here](https://github.com/rbgirshick/py-faster-rcnn). I have not implemented anything new but I fixed the implementations for Windows and Python 3.5.


# How To Use This Branch

1- Install tensorflow, preferably GPU version. Follow [instructions for Windows]( https://www.tensorflow.org/install/install_windows). If you do not install GPU version, you need to comment out all the GPU calls inside code and replace them with relavent CPU ones.

or

1- Install tensorflow, preferably GPU version. Follow [instructions for mac ](https://www.tensorflow.org/install/install_mac). 

or 

1-  Install tensorflow, preferably GPU version. Follow [instructions for linux ](https://www.tensorflow.org/install/install_linux).

(I installed using pip on each system: pip3 install --upgrade tensorflow-gpu )


2- Install python packages (cython, python-opencv, easydict)

3- Checkout this branch

4- Go to  ./data/coco/PythonAPI


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext --inplace`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext install`


5- Go to  ./lib/utils


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext --inplace`


6- Follow this instruction to download PyCoco database. [Link]( https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)


I will be glad if you can contribute with a batch script to automatically download and fetch. The final structure has to look like

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"data/VOCDevkit2007/annotations_cache"
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"data/VOCDevkit2007/VOC2007"

 7- Download pre-trained VGG16 from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as "data\imagenet_weights\vgg16.ckpt"
 
 For rest of the models, please check [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
 
  8- Run train.py

  
  Notify me if there is any issue found. Please note that, I have compiled cython modules with sm61 architecture (GTX 1060, 1070 etc.). Compile support for other architectures will be added. 
 
