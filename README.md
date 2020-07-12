# 3D-GuidedGradCAM-for-Medical-Imaging
This Repo containes the implemnetation of generating Guided-GradCAM for 3D medical Imaging using Nifti file in tensorflow 2.0. Different input files can be used in that case need to edit the input to the Guided-gradCAM model.

## Files:
```ruby  
        i) guided_Gradcam3D.py        -|--> Generate Guided-GradCAM , input and output nifti data
       ii) Guided_GradCAM_3D_config.py |--> Configuration file for the Guided-GradCAM, Modify based on your need
      iii) Resnet_3D.py                |--> Network architecture
       iv) deploy_config.py               |--> Configuration file for the Network, Modify based on your need
        v) loss_funnction_And_matrics.py -|--> Loss functions for CNN
```     
## How to run

To run and generate Guided-GardCAM all is to need to configure the `Guided_GradCAM_3D_config.py` and `deploy_config.py`  based on your requiremnet.

### deploy_config.py-
CNN configuration Change based on your Network or complete replace by your CNN
```ruby

import tensorflow as tf
import math
from loss_funnction_And_matrics import*
###---Number-of-GPU
NUM_OF_GPU=1
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1","gpu:2"]
##Network Configuration
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(224,160,160, 1)
TRAIN_NUM_RES_UNIT=3
TRAIN_NUM_FILTERS=(16, 32, 64, 128)
TRAIN_STRIDES=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
TRAIN_CLASSIFY_LOSS=Weighted_BCTL
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()
```

### Guided_GradCAM_3D_config.py
Input Configuration for the Guided-GradCAM
```ruby
MODEL_WEIGHT="Path/of/Model/Weight/XXX.h5"
CLASS_INDEX=1 # Index of the class for which you want to see the Guided-gradcam
INPUT_PATCH_SIZE_SLICE_NUMBER=64 # Input patch slice you want to feed at a time
LAYER_NAME='conv3d_18' # Name of the layer from where you want to get the Guided-GradCAM
NIFTI_PATH="imput/niftidata/path/XXX.nii.gz"
SAVE_PATH="/Output/niftydata/path/ML_Guided_GradCaN_XXXX.nii.gz"
```

## Sample Guided-GardCAM

![SAMPLE Guided-GradCAM1]()
![SAMPLE Guided-GradCAM2]()
![SAMPLE Guided-GradCAM3]()
