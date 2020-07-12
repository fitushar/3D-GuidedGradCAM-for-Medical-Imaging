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
