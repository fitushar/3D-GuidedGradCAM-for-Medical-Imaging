MODEL_WEIGHT="Path/of/Model/Weight/XXX.h5"
CLASS_INDEX=1 # Index of the class for which you want to see the Guided-gradcam
INPUT_PATCH_SIZE_SLICE_NUMBER=64 # Input patch slice you want to feed at a time
LAYER_NAME='conv3d_18' # Name of the layer from where you want to get the Guided-GradCAM
NIFTI_PATH="imput/niftidata/path/XXX.nii.gz"
SAVE_PATH="/Output/niftydata/path/ML_Guided_GradCaN_XXXX.nii.gz"
