#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 3/7/20
@author: atoultaro
"""

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Reshape, Lambda, GlobalMaxPooling2D, \
    ZeroPadding2D, AveragePooling2D, Activation, add, GlobalAveragePooling2D, ConvLSTM2D, Conv2D, MaxPooling2D, \
    Conv1D, MaxPooling1D, Input, TimeDistributed, LSTM, Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow_addons.layers.spatial_pyramid_pooling as spp


# from residual_attention.models import AttentionResNetCifar10_mod, \
#     AttentionResNet56, AttentionResNetCifar10_mod_v2, \
#     AttentionResNetCifar10_mod_v3


# Kong's attention
# def max_pooling(inputs, **kwargs):
#     input = inputs[0]   # (batch_size, time_steps, freq_bins)
#     return K.max(input, axis=1)
def max_pooling(inputs, **kwargs):
    # input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(inputs, axis=1)


def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):

    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return (sample_num, freq_bins)


def model_cnn14_spp(time_steps, freq_bins, classes_num, conv_dim=64, rnn_dim=128, pool_size=2, pool_stride=2,
                    hidden_units=512, l2_regu=0., drop_rate=0., multilabel=True):
    # cnn14 SPP
    input_layer = Input(shape=(time_steps, freq_bins, 1), name='input')
    # group 1
    y = Conv2D(conv_dim, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(
        input_layer)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 2
    y = Conv2D(conv_dim * 2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 3
    y = Conv2D(conv_dim * 4, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 4, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 4
    y = Conv2D(conv_dim * 8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 5
    y = Conv2D(conv_dim * 16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    #     y = MaxPooling2D(pool_size=(pool_size, 2), strides=(1, 2), padding='same')(y)
    #     y = Dropout(drop_rate)(y)

    #     # group 6
    #     y = Conv2D(conv_dim*32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    #     y = BatchNormalization()(y)
    #     y = Activation(activation='relu')(y)
    #     y = Conv2D(conv_dim*32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    #     y = BatchNormalization()(y)
    #     y = Activation(activation='relu')(y)
    #     y = MaxPooling2D(pool_size=(pool_size, 2), strides=(1, 2), padding='same')(y)
    #     y = Dropout(drop_rate)(y)

    # change dimensions: samples, time, frequency, channels => samples, time, frequency*channels
    #  dim_cnn = K.int_shape(y)
    # y = Reshape((dim_cnn[1], dim_cnn[2]*dim_cnn[3]))(y)

    y = spp.SpatialPyramidPooling2D(bins=[[1, 1], [2, 2], [4, 4]], data_format='channels_last')(y)
    dim_spp = K.int_shape(y)
    y = Reshape((dim_spp[1] * dim_spp[2],))(y)

    # FC block
    a1 = Dense(hidden_units)(y)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)

    a2 = Dense(hidden_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)

    a3 = Dense(hidden_units)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(drop_rate)(a3)

    #     y = Dense(hidden_units, activation='relu', name='cnn14_fcn')(y)  # original 512
    #      y = Dense(hidden_units, activation='relu', name='cnn14_fcn2')(y)  # original 512
    # x = Dense(classes_num, activation='softmax')(y)
    #     x = Dense(classes_num, activation='sigmoid')(y)
    x = Dense(classes_num, activation='sigmoid')(a3)

    # Build model
    model = Model(inputs=input_layer, outputs=x)

    return model


#     a1 = Dense(hidden_units)(y)
#     a1 = BatchNormalization()(a1)
#     a1 = Activation('relu')(a1)
#     a1 = Dropout(drop_rate)(a1)

#     a2 = Dense(hidden_units)(a1)
#     a2 = BatchNormalization()(a2)
#     a2 = Activation('relu')(a2)
#     a2 = Dropout(drop_rate)(a2)

#     output_layer = Dense(classes_num, activation='softmax')(a2)

#     if False:
#         # Pooling layers 'decision_level_max_pooling':
#         '''Global max pooling.

#         [1] Choi, Keunwoo, et al. "Automatic tagging using deep convolutional
#         neural networks." arXiv preprint arXiv:1606.00298 (2016).
#         '''
#         cla = Dense(classes_num, activation='sigmoid')(a2)

#         # output_layer = Lambda(
#         #    max_pooling,
#         #    output_shape=pooling_shape)(
#         #    [cla])
#         output_layer = Lambda(max_pooling)(cla)

#     # Build model
#     model = Model(inputs=input_layer, outputs=output_layer)

#     return model


# cnn14 attention with customized maxpooling
def model_cnn14_attention_multi(time_steps, freq_bins, classes_num, model_type='feature_level_attention', conv_dim=64,
                                rnn_dim=128, pool_size=2, pool_stride=2, hidden_units=512, l2_regu=0., drop_rate=0.,
                                multilabel=True):
    # Kong's attention
    # model_type = 'decision_level_max_pooling'  # problem with dimensions of the Lambda layer after training
    # model_type = 'decision_level_average_pooling' # problem with dimensions of the Lambda layer after training
    # model_type = 'decision_level_single_attention'
    # model_type = 'decision_level_multi_attention'
    # model_type = 'feature_level_attention'

    input_layer = Input(shape=(time_steps, freq_bins, 1), name='input')
    # group 1
    y = Conv2D(conv_dim, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(
        input_layer)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 2
    y = Conv2D(conv_dim * 2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 3
    y = Conv2D(conv_dim * 4, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 4, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 4
    y = Conv2D(conv_dim * 8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(pool_stride, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 5
    y = Conv2D(conv_dim * 16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(1, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # group 6
    y = Conv2D(conv_dim * 32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = Conv2D(conv_dim * 32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_regu))(y)
    y = BatchNormalization()(y)
    y = Activation(activation='relu')(y)
    y = MaxPooling2D(pool_size=(pool_size, 2), strides=(1, 2), padding='same')(y)
    y = Dropout(drop_rate)(y)

    # change dimensions: samples, time, frequency, channels => samples, time, frequency*channels
    dim_cnn = K.int_shape(y)
    y = Reshape((dim_cnn[1], dim_cnn[2] * dim_cnn[3]))(y)

    a1 = Dense(hidden_units)(y)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)

    a2 = Dense(hidden_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)

    a3 = Dense(hidden_units)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(drop_rate)(a3)

    # Pooling layers
    if model_type == 'decision_level_max_pooling':
        '''Global max pooling.

        [1] Choi, Keunwoo, et al. "Automatic tagging using deep convolutional 
        neural networks." arXiv preprint arXiv:1606.00298 (2016).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)

        # output_layer = Lambda(
        #    max_pooling,
        #    output_shape=pooling_shape)(
        #    [cla])
        output_layer = Lambda(max_pooling)(cla)

    elif model_type == 'decision_level_average_pooling':
        '''Global average pooling.

        [2] Lin, Min, et al. Qiang Chen, and Shuicheng Yan. "Network in 
        network." arXiv preprint arXiv:1312.4400 (2013).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        # output_layer = Lambda(
        #    average_pooling,
        #    output_shape=pooling_shape)(
        #    [cla])
        output_layer = Lambda(average_pooling)(cla)

    elif model_type == 'decision_level_single_attention':
        '''Decision level single attention pooling.
        [3] Kong, Qiuqiang, et al. "Audio Set classification with attention
        model: A probabilistic perspective." arXiv preprint arXiv:1711.00927
        (2017).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        att = Dense(classes_num, activation='softmax')(a3)
        output_layer = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])

    elif model_type == 'decision_level_multi_attention':
        '''Decision level multi attention pooling.
        [4] Yu, Changsong, et al. "Multi-level Attention Model for Weakly
        Supervised Audio Classification." arXiv preprint arXiv:1803.02353
        (2018).
        '''
        cla1 = Dense(classes_num, activation='sigmoid')(a2)
        att1 = Dense(classes_num, activation='softmax')(a2)
        out1 = Lambda(attention_pooling, output_shape=pooling_shape)([cla1, att1])

        cla2 = Dense(classes_num, activation='sigmoid')(a3)
        att2 = Dense(classes_num, activation='softmax')(a3)
        out2 = Lambda(attention_pooling, output_shape=pooling_shape)([cla2, att2])

        b1 = Concatenate(axis=-1)([out1, out2])
        b1 = Dense(classes_num)(b1)

        if multilabel:
            output_layer = Activation('sigmoid')(b1)
        else:
            output_layer = Activation('softmax')(b1)

    elif model_type == 'feature_level_attention':
        '''Feature level attention.
        [1] Kong, Qiuqiang, et al. "Weakly labelled audioset tagging with 
        attention neural networks." (2019).
        '''
        cla = Dense(hidden_units, activation='linear')(a3)
        att = Dense(hidden_units, activation='sigmoid')(a3)
        b1 = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])

        b1 = BatchNormalization()(b1)
        b1 = Activation(activation='relu')(b1)
        b1 = Dropout(drop_rate)(b1)

        if multilabel:
            output_layer = Dense(classes_num, activation='sigmoid')(b1)
        else:
            output_layer = Dense(classes_num, activation='softmax')(b1)

    else:
        raise Exception("Incorrect model_type!")

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def lenet_dropout_input_conv(conf):
    """
    Buidling the model of LeNet with dropout on the input, both
    convolutional layers and full-connected layer

    Args:
        conf: configuration class object

        input_shape: (conf.img_F, conf.img_t, 1)
        conf.num_class: numbwer of classes
        conf.RATE_DROPOUT_INPUT: dropout rate
        conf.RATE_DROPOUT_CONV
        conf.RATE_DROPOUT_FC
    Returns:
        model: built model
    """
    model = Sequential()
    model.add(Dropout(conf['dropout'],
                      input_shape=(conf['img_f'], conf['img_t'], 1)))

    # conv group 1
    model.add(
        Conv2D(8, kernel_size=(4, 2), strides=(1, 1),
               kernel_regularizer=regularizers.l2(conf['l2_regu']),
               activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(conf['dropout']))

    # conv group 2
    model.add(Conv2D(8, kernel_size=(4, 2), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(conf['dropout']))

    # conv group 3
    model.add(Conv2D(8, kernel_size=(4, 2), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(conf['dropout']))

    # conv group 4
    model.add(Conv2D(8, kernel_size=(4, 2), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(conf['dropout']))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(conf['dropout']))
    model.add(Dense(conf['num_class'], activation='softmax'))

    return model


def convnet_long_time(conf):
    model = Sequential()
    # Group 1
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     input_shape=(conf['img_f'], conf['img_t'], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 2
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 3
    model.add(
        Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 4
    model.add(
        Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 5
    model.add(
        Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 6
    model.add(
        Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))

    # 1x1 convolution
    model.add(
        Conv2D(8, kernel_size=(1, 1), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same'))
    model.add(GlobalMaxPooling2D())
    # model.add(Dropout(conf['dropout']))
    model.add(Dense(conf['num_class'], activation='softmax'))

    return model


def birdnet(conf):
    """
    Birdnet
    """
    model = Sequential()
    # Group 1
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     input_shape=(conf['img_f'], conf['img_t'], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 2
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 3
    model.add(
        Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 4
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 5
    model.add(
        Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 6
    model.add(
        Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Dropout(conf['dropout']))
    model.add(Dense(conf['num_class'], activation='softmax'))

    return model


def birdnet_5layers(conf):
    """
    Birdnet
    """
    model = Sequential()
    # Group 1
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     input_shape=(conf['img_f'], conf['img_t'], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 2
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 3
    model.add(
        Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 4
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 5
    model.add(
        Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Dropout(conf['dropout']))
    model.add(Dense(conf['num_class'], activation='softmax'))

    return model


def birdnet_7layers(conf):
    """
    Birdnet
    """
    model = Sequential()
    # Group 1
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     input_shape=(conf['img_f'], conf['img_t'], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 2
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 3
    model.add(
        Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 4
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 5
    model.add(
        Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 6
    model.add(
        Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 7
    model.add(
        Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Dropout(conf['dropout']))
    model.add(Dense(conf['num_class'], activation='softmax'))

    return model


def birdnet_8layers(conf):
    """
    Birdnet
    """
    model = Sequential()
    # Group 1
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     input_shape=(conf['img_f'], conf['img_t'], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 2
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 3
    model.add(
        Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 4
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 5
    model.add(
        Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 6
    model.add(
        Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 7
    model.add(
        Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 8
    model.add(
        Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(conf['dropout']))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Dropout(conf['dropout']))
    model.add(Dense(conf['num_class'], activation='softmax'))

    return model


def birdnet_freq(conf):
    """
    Birdnet
    """
    model = Sequential()
    # Group 1
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),
                     input_shape=(conf['img_f'], conf['img_t'], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 2
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
                     kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 3
    model.add(
        Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 4
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 5
    model.add(
        Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(conf['dropout']))
    # Group 6
    model.add(
        Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu',
               kernel_initializer='he_normal', padding='same',
               kernel_regularizer=regularizers.l2(conf['l2_regu']),))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(conf['dropout']))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Dropout(conf['dropout']))
    model.add(Dense(conf['num_class'], activation='softmax'))

    return model

# Resnet
# from tensorflow.keras.layers.advanced_activations import PReLU


# def name_builder(type, stage, block, name):
#     return "{}{}{}_branch{}".format(type, stage, block, name)
#
#
# def identity_block(input_tensor, kernel_size, filters, stage, block):
#     F1, F2, F3 = filters
#     def name_fn(type, name):
#         return name_builder(type, stage, block, name)
#
#     x = Conv2D(F1, (1, 1), name=name_fn('res', '2a'))(input_tensor)
#     x = BatchNormalization(name=name_fn('bn', '2a'))(x)
#     x = PReLU()(x)
#
#     x = Conv2D(F2, kernel_size, padding='same', name=name_fn('res', '2b'))(x)
#     x = BatchNormalization(name=name_fn('bn', '2b'))(x)
#     x = PReLU()(x)
#
#     x = Conv2D(F3, (1, 1), name=name_fn('res', '2c'))(x)
#     x = BatchNormalization(name=name_fn('bn', '2c'))(x)
#     x = PReLU()(x)
#
#     x = add([x, input_tensor])
#     x = PReLU()(x)
#
#     return x
#
#
# def conv_block(input_tensor, kernel_size, filters, stage, block,
#                strides=(2, 2)):
#     def name_fn(type, name):
#         return name_builder(type, stage, block, name)
#
#     F1, F2, F3 = filters
#
#     x = Conv2D(F1, (1, 1), strides=strides, name=name_fn("res", "2a"))(
#         input_tensor)
#     x = BatchNormalization(name=name_fn("bn", "2a"))(x)
#     x = PReLU()(x)
#
#     x = Conv2D(F2, kernel_size, padding='same', name=name_fn("res", "2b"))(x)
#     x = BatchNormalization(name=name_fn("bn", "2b"))(x)
#     x = PReLU()(x)
#
#     x = Conv2D(F3, (1, 1), name=name_fn("res", "2c"))(x)
#     x = BatchNormalization(name=name_fn("bn", "2c"))(x)
#
#     sc = Conv2D(F3, (1, 1), strides=strides, name=name_fn("res", "1"))(
#         input_tensor)
#     sc = BatchNormalization(name=name_fn("bn", "1"))(sc)
#
#     x = add([x, sc])
#     x = PReLU()(x)
#
#     return x
#
#
# # resnet
# def resnet(conf):
#     input_tensor = Input(shape=(conf['img_f'], conf['img_t'], 1))
#     net = ZeroPadding2D((3, 3))(input_tensor)
#     net = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(net)
#     net = BatchNormalization(name="bn_conv1")(net)
#     net = PReLU()(net)
#     net = MaxPooling2D((3, 3), strides=(2, 2))(net)
#
#     net = conv_block(net, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     net = identity_block(net, 3, [64, 64, 256], stage=2, block='b')
#     net = identity_block(net, 3, [64, 64, 256], stage=2, block='c')
#
#     net = conv_block(net, 3, [128, 128, 512], stage=3, block='a')
#     net = identity_block(net, 3, [128, 128, 512], stage=3, block='b')
#     net = identity_block(net, 3, [128, 128, 512], stage=3, block='c')
#     net = identity_block(net, 3, [128, 128, 512], stage=3, block='d')
#
#     net = conv_block(net, 3, [256, 256, 1024], stage=4, block='a')
#     net = identity_block(net, 3, [256, 256, 1024], stage=4, block='b')
#     net = identity_block(net, 3, [256, 256, 1024], stage=4, block='c')
#     net = identity_block(net, 3, [256, 256, 1024], stage=4, block='d')
#     net = identity_block(net, 3, [256, 256, 1024], stage=4, block='e')
#     net = identity_block(net, 3, [256, 256, 1024], stage=4, block='f')
#     net = AveragePooling2D((2, 2))(net)
#
#     net = Flatten()(net)
#     net = Dense(conf['num_class'], activation="softmax", name="softmax")(net)
#
#     model = Model(inputs=input_tensor, outputs=net)
#
#     return model


def bn(x, name, zero_init=False):
    return BatchNormalization(name=name,)(x)

    # return BatchNormalization(
    #     axis=1, name=name,
    #     momentum=0.9, epsilon=1e-5,
    #     gamma_initializer='zeros' if zero_init else 'ones')(x)

    # return BatchNormalization(
    #     axis=1, name=name, fused=True,
    #     momentum=0.9, epsilon=1e-5,
    #     gamma_initializer='zeros' if zero_init else 'ones')(x)


def conv(x, filters, kernel, strides=1, name=None):
    return Conv2D(filters, kernel, name=name,
                  strides=strides, use_bias=False, padding='same',
                  kernel_regularizer=regularizers.l2(1e-4))(x)


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, (1, 1), name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, strides=strides, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv(x, filters3, (1, 1), name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    shortcut = conv(
        input_tensor,
        filters3, (1, 1), strides=strides,
        name=conv_name_base + '1')
    shortcut = bn(shortcut, name=bn_name_base + '1')

    # x = tf.keras.layers.add([x, shortcut])
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, 1, name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv(x, filters3, (1, 1), name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    # x = tf.keras.layers.add([x, input_tensor])
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_simple(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, kernel_size, name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, strides=strides, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b', zero_init=True)

    shortcut = conv(
        input_tensor,
        filters2, (1, 1), strides=strides,
        name=conv_name_base + '1')
    shortcut = bn(shortcut, name=bn_name_base + '1')

    # x = tf.keras.layers.add([x, shortcut])
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block_simple(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, kernel_size, name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    # x = tf.keras.layers.add([x, input_tensor])
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_2_no_direct(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, kernel_size, name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, strides=strides, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b', zero_init=True)
    x = Activation('relu')(x)
    return x


def resnet18(conf):
    # input_shape = (conf['img_t'], conf['img_f'], 1)
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']

    inputs = Input(shape=input_shape)

    x = conv(inputs, 64, (7, 7), strides=2, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block_simple(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [64, 64], stage=2, block='b')

    x = conv_block_simple(x, 3, [128, 128], stage=3, block='a')
    x = identity_block_simple(x, 3, [128, 128], stage=3, block='b')

    x = conv_block_simple(x, 3, [256, 256], stage=4, block='a')
    x = identity_block_simple(x, 3, [256, 256], stage=4, block='b')

    x = conv_block_simple(x, 3, [512, 512], stage=5, block='a')
    x = identity_block_simple(x, 3, [512, 512], stage=5, block='b')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet34(conf):
    # input_shape = (conf['img_t'], conf['img_f'], 1)
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']

    inputs = Input(shape=input_shape)

    x = conv(inputs, 64, (7, 7), strides=2, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block_simple(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [64, 64], stage=2, block='b')
    x = identity_block_simple(x, 3, [64, 64], stage=2, block='c')

    x = conv_block_simple(x, 3, [128, 128], stage=3, block='a')
    x = identity_block_simple(x, 3, [128, 128], stage=3, block='b')
    x = identity_block_simple(x, 3, [128, 128], stage=3, block='c')
    x = identity_block_simple(x, 3, [128, 128], stage=3, block='d')

    x = conv_block_simple(x, 3, [256, 256], stage=4, block='a')
    x = identity_block_simple(x, 3, [256, 256], stage=4, block='b')
    x = identity_block_simple(x, 3, [256, 256], stage=4, block='c')
    x = identity_block_simple(x, 3, [256, 256], stage=4, block='d')
    x = identity_block_simple(x, 3, [256, 256], stage=4, block='e')
    x = identity_block_simple(x, 3, [256, 256], stage=4, block='f')

    x = conv_block_simple(x, 3, [512, 512], stage=5, block='a')
    x = identity_block_simple(x, 3, [512, 512], stage=5, block='b')
    x = identity_block_simple(x, 3, [512, 512], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet50(conf):
    # input_shape = (conf['img_t'], conf['img_f'], 1)
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']

    inputs = Input(shape=input_shape)

    x = conv(inputs, 64, (7, 7), strides=2, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_cifar10_expt(conf):
    # resemble resnet_v1 using conv_block, identity_block
    # stack=4, filter_num=16
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_cifar10_expt_maxpool(conf):
    # resemble resnet_v1 using conv_block, identity_block
    # stack=4, filter_num=16
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_cifar10_expt_no_direct(conf):
    # resemble resnet_v1 using conv_block, identity_block
    # stack=4, filter_num=16
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_2_no_direct(x, 3, [num_filters, num_filters], stage=2, block='a')
    x = conv_block_2_no_direct(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = conv_block_2_no_direct(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_2_no_direct(x, 3, [num_filters*2, num_filters*2], stage=3, block='a', strides=(2, 2))
    x = conv_block_2_no_direct(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = conv_block_2_no_direct(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_2_no_direct(x, 3, [num_filters*4, num_filters*4], stage=4, block='a', strides=(2, 2))
    x = conv_block_2_no_direct(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = conv_block_2_no_direct(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = conv_block_2_no_direct(x, 3, [num_filters*8, num_filters*8], stage=5, block='a', strides=(2, 2))
    x = conv_block_2_no_direct(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = conv_block_2_no_direct(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_cifar10_expt_not_global(conf):
    # resemble resnet_v1 using conv_block, identity_block
    # stack=4, filter_num=16
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    # x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = AveragePooling2D(pool_size=(int(num_filters/2), int(num_filters/2)), name='avg_pool')(x)
    x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_cifar10_expt_deep3(conf):
    # resemble resnet_v1 using conv_block, identity_block
    # stack=4, filter_num=16
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(int(num_filters/2), int(num_filters/2)), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_cifar10_expt_deep5(conf):
    # resemble resnet_v1 using conv_block, identity_block
    # stack=4, filter_num=16
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    x = conv_block_simple(x, 3, [num_filters*16, num_filters*16], stage=6, block='a')
    x = identity_block_simple(x, 3, [num_filters*16, num_filters*16], stage=6, block='b')
    x = identity_block_simple(x, 3, [num_filters*16, num_filters*16], stage=6, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(int(num_filters/2), int(num_filters/2)), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_cifar10_expt_deep6(conf):
    # resemble resnet_v1 using conv_block, identity_block
    # stack=4, filter_num=16
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    x = conv_block_simple(x, 3, [num_filters*16, num_filters*16], stage=6, block='a')
    x = identity_block_simple(x, 3, [num_filters*16, num_filters*16], stage=6, block='b')
    x = identity_block_simple(x, 3, [num_filters*16, num_filters*16], stage=6, block='c')

    x = conv_block_simple(x, 3, [num_filters*32, num_filters*32], stage=7, block='a')
    x = identity_block_simple(x, 3, [num_filters*32, num_filters*32], stage=7, block='b')
    x = identity_block_simple(x, 3, [num_filters*32, num_filters*32], stage=7, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(int(num_filters/2), int(num_filters/2)), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet18_expt(conf):
    # input_shape = (conf['img_t'], conf['img_f'], 1)
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    # x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a')
    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet34_expt(conf):
    # input_shape = (conf['img_t'], conf['img_f'], 1)
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')
    x = identity_block_simple(x, 3, [num_filters * 2, num_filters * 2], stage=3, block='d')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='d')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='e')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='f')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def conv_talos(x, filters, kernel, strides=1, l2=1e-4, name=None):
    return Conv2D(filters, kernel, name=name,
                  strides=strides, use_bias=False, padding='same',
                  kernel_regularizer=regularizers.l2(l2))(x)


def conv_block_simple_talos(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), l2=1e-4):
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv_talos(input_tensor, filters1, kernel_size, name=conv_name_base + '2a', l2=l2)
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv_talos(x, filters2, kernel_size, strides=strides, name=conv_name_base + '2b', l2=l2)
    x = bn(x, name=bn_name_base + '2b', zero_init=True)

    shortcut = conv_talos(
        input_tensor,
        filters2, (1, 1), strides=strides,
        name=conv_name_base + '1',
        l2=l2)
    shortcut = bn(shortcut, name=bn_name_base + '1')

    # x = tf.keras.layers.add([x, shortcut])
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block_simple_talos(input_tensor, kernel_size, filters, stage, block, l2=1e-4):
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv_talos(input_tensor, filters1, kernel_size, name=conv_name_base + '2a', l2=l2)
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv_talos(x, filters2, kernel_size, name=conv_name_base + '2b', l2=l2)
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    # x = tf.keras.layers.add([x, input_tensor])
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet34_expt_talos(conf, params):
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = params['num_filters']

    inputs = Input(shape=input_shape)
    x = conv_talos(inputs, num_filters, (3, 3), strides=1, l2=params['l2_regu'], name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block_simple_talos(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1), l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters, num_filters], stage=2, block='b', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters, num_filters], stage=2, block='c', l2=params['l2_regu'])

    x = conv_block_simple_talos(x, 3, [num_filters*2, num_filters*2], stage=3, block='a', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*2, num_filters*2], stage=3, block='b', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*2, num_filters*2], stage=3, block='c', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters * 2, num_filters * 2], stage=3, block='d', l2=params['l2_regu'])

    x = conv_block_simple_talos(x, 3, [num_filters*4, num_filters*4], stage=4, block='a', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*4, num_filters*4], stage=4, block='b', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*4, num_filters*4], stage=4, block='c', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*4, num_filters*4], stage=4, block='d', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*4, num_filters*4], stage=4, block='e', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*4, num_filters*4], stage=4, block='f', l2=params['l2_regu'])

    x = conv_block_simple_talos(x, 3, [num_filters*8, num_filters*8], stage=5, block='a', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*8, num_filters*8], stage=5, block='b', l2=params['l2_regu'])
    x = identity_block_simple_talos(x, 3, [num_filters*8, num_filters*8], stage=5, block='c', l2=params['l2_regu'])

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet34_expt_maxpool(conf):
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block_simple(x, 3, [num_filters, num_filters], stage=2, block='a', strides=(1, 1))
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='b')
    x = identity_block_simple(x, 3, [num_filters, num_filters], stage=2, block='c')

    x = conv_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='a')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='b')
    x = identity_block_simple(x, 3, [num_filters*2, num_filters*2], stage=3, block='c')

    x = conv_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='a')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='b')
    x = identity_block_simple(x, 3, [num_filters*4, num_filters*4], stage=4, block='c')

    x = conv_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='a')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='b')
    x = identity_block_simple(x, 3, [num_filters*8, num_filters*8], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet50_expt(conf):
    # resemble resnet_v1 using conv_block, identity_block
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='b')
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='c')

    x = conv_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='a')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='b')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='c')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='d')

    x = conv_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='a')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='b')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='c')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='d')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='e')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='f')

    x = conv_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='a')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='b')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet50_expt_nonglobal(conf):
    # resemble resnet_v1 using conv_block, identity_block
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='b')
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='c')

    x = conv_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='a')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='b')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='c')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='d')

    x = conv_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='a')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='b')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='c')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='d')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='e')
    x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='f')

    x = conv_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='a')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='b')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='c')

    # x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet101_expt(conf):
    # resemble resnet_v1 using conv_block, identity_block
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='b')
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='c')

    x = conv_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='a')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='b')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='c')
    x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='d')

    x = conv_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='_block1')
    for ii in range(1, 23):
        x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='_block'+str(ii+1))

    x = conv_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='a')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='b')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet152_expt(conf):
    # resemble resnet_v1 using conv_block, identity_block
    input_shape = (conf['img_t'], conf['img_f'], 1)
    num_class = conf['num_class']
    num_filters = conf['num_filters']

    inputs = Input(shape=input_shape)
    x = conv(inputs, num_filters, (3, 3), strides=1, name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)

    x = conv_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='b')
    x = identity_block(x, 3, [num_filters, num_filters, num_filters*4], stage=2, block='c')

    x = conv_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='_block1')
    for ii in range(1, 8):
        x = identity_block(x, 3, [num_filters * 2, num_filters * 2, num_filters * 8], stage=3, block='_block'+str(ii+1))
        # x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='b')
        # x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='c')
        # x = identity_block(x, 3, [num_filters*2, num_filters*2, num_filters*8], stage=3, block='d')

    x = conv_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='_block1')
    for ii in range(1, 36):
        x = identity_block(x, 3, [num_filters*4, num_filters*4, num_filters*16], stage=4, block='_block'+str(ii+1))

    x = conv_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='a')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='b')
    x = identity_block(x, 3, [num_filters*8, num_filters*8, num_filters*32], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = AveragePooling2D(pool_size=(8, 8), name='avg_pool')(x)
    # x = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(conf):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_class (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    input_shape = (conf['img_t'], conf['img_f'], 1)
    depth = conf['depth']
    num_class = conf['num_class']
    num_stack = conf['num_stack']
    num_filters = conf['num_filters']
    kernel_size = conf['kernel_size']

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    # num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters)
    # Instantiate the stack of residual units
    # for stack in range(3):
    for stack in range(num_stack):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             kernel_size=kernel_size,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             kernel_size=kernel_size,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=(8, 8))(x)
    y = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_class=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_class (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(8, 8))(x)
    y = Flatten()(x)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def lstm_2lay(conf):
    """
    LSTM
    :param data_shape:
    :param conf:
    :return:
    """
    l2_regu = conf["l2_regu"]
    dropout_rate = conf["dropout"]
    recurr_dropout_rate = conf["recurrent_dropout"]
    data_shape = (conf['img_t'], conf['img_f'])
    num_chan1 = conf["lstm1"]
    num_chan2 = conf["lstm2"]
    num_class = conf['num_class']
    dense_size = conf["dense_size"]

    model_rnn = Sequential()
    model_rnn.add(LSTM(num_chan1,
                       input_shape=data_shape,
                       return_sequences=True,
                       kernel_regularizer=regularizers.l2(l2_regu),
                       recurrent_regularizer=regularizers.l2(l2_regu),
                       dropout=dropout_rate,
                       recurrent_dropout=recurr_dropout_rate
                       ))
    model_rnn.add(BatchNormalization(center=False, scale=False))

    model_rnn.add(LSTM(num_chan2,
                       return_sequences=False,
                       kernel_regularizer=regularizers.l2(l2_regu),
                       recurrent_regularizer=regularizers.l2(l2_regu),
                       dropout=dropout_rate,
                       recurrent_dropout=recurr_dropout_rate))
    model_rnn.add(BatchNormalization(center=False, scale=False))

    # model_rnn.add(Dense(dense_size))
    model_rnn.add(Dense(num_class, activation='softmax'))
    return model_rnn


def conv2d_lstm(conf):  # expect input shape 5
    l2_regu = conf["l2_regu"]
    dropout_rate = conf["dropout"]
    recurr_dropout_rate = conf["recurrent_dropout"]
    data_shape = (conf['img_t'], conf['img_f'], 1, 1)  # last two dims are image's y-axis/freq and num of rbg
    num_chan1 = conf["lstm1"]
    num_chan2 = conf["lstm2"]
    dense_size = conf["dense_size"]

    model = Sequential()
    model.add(ConvLSTM2D(filters=num_chan1, kernel_size=(4, 1),
                         strides=(2, 1),
                         input_shape=data_shape, padding='valid',
                         kernel_regularizer=regularizers.l2(l2_regu),
                         recurrent_regularizer=regularizers.l2(l2_regu),
                         dropout=dropout_rate,
                         recurrent_dropout=recurr_dropout_rate,
                         return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=num_chan2, kernel_size=(4, 1),
                         strides=(2, 1),
                         padding='valid',
                         kernel_regularizer=regularizers.l2(l2_regu),
                         recurrent_regularizer=regularizers.l2(l2_regu),
                         dropout=dropout_rate,
                         recurrent_dropout=recurr_dropout_rate,
                         return_sequences=False))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(dense_size))
    model.add(Dense(conf["num_class"], activation='softmax'))

    return model


# def attention_resnet56(conf):
#     input_shape = (conf['img_t'], conf['img_f'], 1)
#
#     model = AttentionResNet56(shape=input_shape,
#                               n_channels=conf['num_filters'],
#                               n_classes=conf["num_class"])
#     return model
#
#
# def attention_res_cifar10_v1(conf):
#     input_shape = (conf['img_t'], conf['img_f'], 1)
#     model = AttentionResNetCifar10_mod(shape=input_shape,
#                                    n_channels=conf['num_filters'],
#                                    n_classes=conf["num_class"])
#     return model
#
#
# def attention_res_cifar10_v2(conf):
#     input_shape = (conf['img_t'], conf['img_f'], 1)
#
#     # model = AttentionResNetCifar10_mod(shape=input_shape,
#     #                                n_channels=conf['num_filters'],
#     #                                n_classes=conf["num_class"])
#     model = AttentionResNetCifar10_mod_v2(shape=input_shape,
#                                    num_filters=conf['num_filters'],
#                                    n_classes=conf["num_class"])
#
#     return model
#
#
# def attention_res_cifar10_v3(conf):
#     input_shape = (conf['img_t'], conf['img_f'], 1)
#
#     model = AttentionResNetCifar10_mod_v3(shape=input_shape,
#                                    num_filters=conf['num_filters'],
#                                    n_classes=conf["num_class"])
#
#     return model