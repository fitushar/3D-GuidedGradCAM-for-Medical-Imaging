## Import Libararies
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
import math
import cv2
from deploy_config import*
from loss_funnction_And_matrics import*
from Resnet_3D import Resnet3D
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from skimage.transform import resize
from Guided_GradCAM_3D_config import*

# Function to get the image chunk fot guided GradCAM
def Get_image_array_Array_and_give_chunk(image_array,patch_slice_slice):

    Devide_integer=image_array.shape[0] // patch_slice_slice
    Reminder= image_array.shape[0] % patch_slice_slice
    print('CT Volume_Shape={}'.format(image_array.shape))
    print('Devide_integer={}'.format(Devide_integer))
    print('Reminder={}'.format(Reminder))
    print('Total of {} + {} ={} Should ={}'.format(patch_slice_slice*Devide_integer,Reminder,patch_slice_slice*Devide_integer+Reminder,image_array.shape[0]))

    lastpatch_starts_from= (image_array.shape[0])-patch_slice_slice
    print(lastpatch_starts_from)

    patch_list=[]
    patch_start=0
    patch_end=patch_slice_slice
    for i in range(Devide_integer):
        #print(patch_start)
        #print(patch_end)
        ct_volume=image_array[patch_start:patch_end,:,:]
        #print(ct_volume.shape)
        patch_list.append(ct_volume)
        patch_start+=patch_slice_slice
        patch_end+=patch_slice_slice

    last_slice_number_would_be=image_array.shape[0]
    print(last_slice_number_would_be)
    last_patch_When_making_nifty=(patch_slice_slice)-Reminder
    print(last_patch_When_making_nifty)
    Slice_will_start_from_here=last_slice_number_would_be-patch_slice_slice
    print(Slice_will_start_from_here)
    last_patch=image_array[Slice_will_start_from_here:,:,:]
    last_patch.shape
    patch_list.append(last_patch)

    return patch_list,last_patch_When_making_nifty

def Get_Build_model(Input_patch_size,Model_weight,Layer_name):
    inputs = tf.keras.Input(shape=Input_patch_size, name='CT')
    Model_3D=Resnet3D(inputs,num_classes=NUMBER_OF_CLASSES)
    Model_3D.load_weights(Model_weight)
    print('Loading The Model from this path--{}'.format(MODEL_PATH))
    Model_3D.summary()
    Build_model=tf.keras.models.Model([Model_3D.inputs], [Model_3D.get_layer(Layer_name).output, Model_3D.output])

    return Build_model


def Guided_GradCAM_3D(Grad_model,ct_io,Class_index):

    # Create a graph that outputs target convolution and output
    grad_model = Grad_model
    input_ct_io=tf.expand_dims(ct_io, axis=-1)
    input_ct_io=tf.expand_dims(input_ct_io, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_ct_io)
        loss = predictions[:, Class_index]
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    ##--Guided Gradient Part
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    # Average gradients spatially
    weights = tf.reduce_mean(guided_grads, axis=(0, 1,2))
    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:3], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * output[:, :, :, index]

    capi=resize(cam,(ct_io.shape))
    print(capi.shape)
    capi = np.maximum(capi,0)
    heatmap = (capi - capi.min()) / (capi.max() - capi.min())
    return heatmap

def generate_guided_grad_cam(Nifti_path,Model_weight,Class_index,Input_patch_size_slice_number,Layer_name,Save_path):
    # Reading the CT
    img_path=Nifti_path
    Class_index=Class_index
    Model_weight=Model_weight
    Layer_name=Layer_name
    img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
    image= sitk.GetArrayFromImage(img_sitk)
    Input_patch_size=[Input_patch_size_slice_number,image.shape[1],image.shape[2],1]

    get_grad_model=Get_Build_model(Input_patch_size,Model_weight,Layer_name)
    ct_patch_chunk_List,last_patch_number=Get_image_array_Array_and_give_chunk(image_array=image,patch_slice_slice=Input_patch_size_slice_number)
    first_heatmap=Guided_GradCAM_3D(get_grad_model,ct_patch_chunk_List[0],Class_index=Class_index)
    heatmap_concat=first_heatmap
    for i in range(1,(len(ct_patch_chunk_List)-1)):
        from Resnet_3D import Resnet3D
        get_heatmap=Guided_GradCAM_3D(get_grad_model,ct_patch_chunk_List[i],Class_index=Class_index)
        heatmap_concat=np.concatenate((heatmap_concat, get_heatmap), axis=0)
    last_heatmap=Guided_GradCAM_3D(get_grad_model,ct_patch_chunk_List[-1],Class_index=Class_index)
    heatmap_concat=np.concatenate((heatmap_concat, last_heatmap[last_patch_number:,:,:]), axis=0)
    s_itk_image = sitk.GetImageFromArray(heatmap_concat)
    s_itk_image.CopyInformation(img_sitk)
    sitk.WriteImage(s_itk_image, Save_path)
    return

if __name__ == '__main__':

    img_path=NIFTI_PATH
    Model_weight=MODEL_WEIGHT
    Class_index=CLASS_INDEX
    Input_patch_size_slice_number=INPUT_PATCH_SIZE_SLICE_NUMBER
    Layer_name=LAYER_NAME
    Save_path=SAVE_PATH
    generate_guided_grad_cam(img_path,Model_weight,Class_index,Input_patch_size_slice_number,Layer_name,Save_path)
