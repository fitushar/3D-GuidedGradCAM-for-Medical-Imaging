from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from deploy_config import*
from loss_funnction_And_matrics import*
import numpy as np

###Residual Block
def Residual_Block(inputs,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 use_bias=False,
                 activation=tf.nn.relu6,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                 bias_regularizer=None,
                 **kwargs):


    conv_params={'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    in_filters = inputs.get_shape().as_list()[-1]
    x=inputs
    orig_x=x

    ##building
    # Adjust the strided conv kernel size to prevent losing information
    k = [s * 2 if s > 1 else k for k, s in zip(kernel_size, strides)]

    if np.prod(strides) != 1:
            orig_x = tf.keras.layers.MaxPool3D(pool_size=strides,strides=strides,padding='valid')(orig_x)

    ##sub-unit-0
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=k,strides=strides,**conv_params)(x)

    ##sub-unit-1
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_params)(x)

        # Handle differences in input and output filter sizes
    if in_filters < out_filters:
        orig_x = tf.pad(tensor=orig_x,paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

    elif in_filters > out_filters:
        orig_x = tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_params)(orig_x)

    x += orig_x
    return x



## Resnet----3D
def Resnet3D(inputs,
              num_classes,
              num_res_units=TRAIN_NUM_RES_UNIT,
              filters=TRAIN_NUM_FILTERS,
              strides=TRAIN_STRIDES,
              use_bias=False,
              activation=TRAIN_CLASSIFY_ACTICATION,
              kernel_initializer=TRAIN_KERNAL_INITIALIZER,
              bias_initializer=tf.zeros_initializer(),
              kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
              bias_regularizer=None,
              **kwargs):
    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}


    ##building
    k = [s * 2 if s > 1 else 3 for s in strides[0]]


    #Input
    x = inputs
    #1st-convo
    x=tf.keras.layers.Conv3D(filters[0], k, strides[0], **conv_params)(x)

    for res_scale in range(1, len(filters)):
        x = Residual_Block(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                name='unit_{}_0'.format(res_scale))
        for i in range(1, num_res_units):
            x = Residual_Block(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    name='unit_{}_{}'.format(res_scale, i))


    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    #axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
    #x = tf.reduce_mean(x, axis=axis, name='global_avg_pool')
    x=tf.keras.layers.GlobalAveragePooling3D()(x)
    x =tf.keras.layers.Dropout(0.5)(x)
    classifier=tf.keras.layers.Dense(units=num_classes,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=classifier)
    #model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss=[TRAIN_CLASSIFY_LOSS], metrics=[TRAIN_CLASSIFY_METRICS,tf.keras.metrics.AUC()])

    return model
