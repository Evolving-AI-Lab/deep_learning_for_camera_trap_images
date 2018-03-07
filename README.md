# deep_learning_for_camera_trap_images
This repository contains the code used for the following paper:

Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning

Authors: Mohammad Sadegh Norouzzadeh, Anh Nguyen, Margaret Kosmala, Ali Swanson, Meredith Palmer, Craig Packer, Jeff Clune

Most of the code in this repository is taken from here: https://github.com/arashno/tensorflow_multigpu_imagenet

This repository has four independent parts:

1- The code used for Task I: Detecting Images That Contain Animals (phase1 folder)

2- The code used for Task II,III, and IV: identifying, counting, and describing animals in images (phase 2 folder)

3- The code used for Task II only, (all the transfer learning experiments for Task II used this part of the repo) (phase2_recognition_only folder)

4- resize.py is used for resizing the input images for all the other parts


For more information on how to use this repo please refer to the base repo at this link: https://github.com/arashno/tensorflow_multigpu_imagenet

Pre-trained models could be found at the following links:

Phase 1 (VGG architecture):

http://www.cs.uwyo.edu/~mnorouzz/share/pretrained/phase1.zip

Phase 2 (ResNet-152 architecture):

http://www.cs.uwyo.edu/~mnorouzz/share/pretrained/phase2.zip

Phase 2 recognition only (ResNet-152 architecture):

http://www.cs.uwyo.edu/~mnorouzz/share/pretrained/phase2_recognition_only.zip
