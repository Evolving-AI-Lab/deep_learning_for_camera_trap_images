# deep_learning_for_camera_trap_images
This repository contains the code used for the following paper:

[Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning](http://www.pnas.org/content/early/2018/06/04/1719367115)

Authors: [Mohammad Sadegh Norouzzadeh](http://arash.norouzzadeh.com), Anh Nguyen, Margaret Kosmala, Ali Swanson, Meredith Palmer, Craig Packer, Jeff Clune

**If you use this code in an academic article, please cite the following paper:**
```
	@article {Norouzzadeh201719367,
		author = {Norouzzadeh, Mohammad Sadegh and Nguyen, Anh and Kosmala, Margaret and Swanson, Alexandra and Palmer, Meredith S. and Packer, Craig and Clune, Jeff},
		title = {Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning},
		year = {2018},
		doi = {10.1073/pnas.1719367115},
		publisher = {National Academy of Sciences},
		issn = {0027-8424},
		URL = {http://www.pnas.org/content/early/2018/06/04/1719367115},
		eprint = {http://www.pnas.org/content/early/2018/06/04/1719367115.full.pdf},
		journal = {Proceedings of the National Academy of Sciences}
	}
```

Most of the code in this repository is taken from [here](https://github.com/arashno/tensorflow_multigpu_imagenet)

This repository has four independent parts:

1- The code used for Task I: Detecting Images That Contain Animals (phase1 folder)

2- The code used for Task II,III, and IV: identifying, counting, and describing animals in images (phase 2 folder)

3- The code used for Task II only, (all the transfer learning experiments for Task II used this part of the repo) (phase2_recognition_only folder)

4- resize.py is used for resizing the input images for all the other parts


For more information on how to use this repo please refer to the base repo at [this link](https://github.com/arashno/tensorflow_multigpu_imagenet)

## 1. Requirements

### Requirements
To use this code, you will need to install the following:
* Python 2.7
* Tenorflow 
* NumPy
* SciPy
* MatPlot Lib

### 2. Running
Pre-trained models could be found at the following links:

* Phase 1 (VGG architecture):

https://drive.google.com/file/d/1Y-aDWNMfvgYUb-u-_cqzibZ6ePFOOLGj/view?usp=sharing

* Phase 2 (ResNet-152 architecture):

https://drive.google.com/file/d/1KTV9dmqkv0xrheIOEkPXbqeg36_rXJ_E/view?usp=sharing

* Phase 2 recognition only (ResNet-152 architecture):

https://drive.google.com/file/d/1cAcnyBTO5JeB2zSaEoGBWf0Jd-jAnguS/view?usp=sharing

## 3. Licenses
This code is licensed under MIT License. 

## 4. Questions?
For questions/suggestions, feel free to [email](mailto:arash.norouzzadeh@gmail.com), tweet to [@arashnorouzzade](https://twitter.com/arashnorouzzade) or create a github issue. 
