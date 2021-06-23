#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/14/21
@author: atoultaro

Title: MixUp augmentation for image classification
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/03/06
Last modified: 2021/03/06
Description: Data augmentation using the mixup technique for image classification.
"""
import os
import numpy as np
import pandas as pd
from os import makedirs
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

# import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
from focal_loss import BinaryFocalLoss
from lib_validation import DataGenerator, find_best_model
from lib_model import model_cnn14_spp, model_cnn14_attention_multi
from lib_augment import mix_up
"""
## Define the mixup technique function

To perform the mixup routine, we create new virtual datasets using the training data from
the same dataset, and apply a lambda value within the [0, 1] range sampled from a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
— such that, for example, `new_x = lambda * x1 + (1 - lambda) * x2` (where
`x1` and `x2` are images) and the same equation is applied to the labels as well.
"""

"""
## Define hyperparameters
"""
learning_rate = 1.e-4  # bce
# learning_rate = 1.e-3  # focal loss
conv_dim = 64
pool_size = 2
pool_stride = 2
l2_regu = 0.0001
drop_rate = 0.2
hidden_units = 512
fcn_dim = 512

run_num = 0
# num_epoch = 200
num_epoch = 1  # debug
batch_size = 32  # for cnn14+attention
copies_of_aug = 10  # cannot be changed

num_patience = 20
num_fold = 5
"""
## Prepare the dataset

In this example, we will be using the [FashionMNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset. But this same recipe can
be used for other classification datasets as well.
"""
# root_dir = '/home/ys587/__Data/__whistle'  # where we have __whislte_30_species folder
root_dir = '/home/ubuntu'  # where we have __whislte_30_species folder
work_path = os.path.join(root_dir, '__whistle_30_species')
fit_result_path =  os.path.join(work_path, '__fit_result_species')

species_dict = {'BD': 0, 'CD': 1, 'STR': 2, 'SPT': 3, 'SPIN': 4, 'PLT': 5, 'RT': 6,  'FKW': 7}
num_species = len(species_dict)
species_list = list(species_dict.keys())
species_id = list(species_dict.values())

# __feature_crossnoise, __feature_singlenoise, __feature_nonoise
feature_path = os.path.join(root_dir, '__whistle_30_species/__dataset/__feature_crossnoise')
# feature_path = os.path.join(root_dir, '__whistle_30_species/__dataset/__feature_nonoise')

# df_species = pd.read_csv(os.path.join(feature_path, 'all.csv'))
df_species = pd.read_csv(os.path.join(feature_path, 'all_species.csv'))
df_noise = pd.read_csv(os.path.join(feature_path, 'all_noise.csv'))

today = datetime.now()
# create a folder based on date & time
# fit_result_path1 = os.path.join(fit_result_path, today.strftime('%Y%m%d_%H%M%S'))
fit_result_path1 = os.path.join(fit_result_path, today.strftime('%Y%m%d_%H%M%S')+'_encounter_run'+str(run_num))

# deploy_list = ['STAR2000', 'STAR2003', 'STAR2006', 'HICEAS2002', 'PICEAS2005']
random_list0 = [0, 10, 20, 30, 40]
random_list = [rr + run_num for rr in random_list0]

label_pred_all = []
label_test_all = []

skf = StratifiedKFold(n_splits=num_fold)

# k-fold split
fea_temp_orig = np.load(os.path.join(feature_path, 'all_orig.npz'))
labels_orig = fea_temp_orig['labels_orig']
del fea_temp_orig

fold_id = 0
for train_set, test_set in skf.split(np.arange(labels_orig.shape[0]), labels_orig):
    print('Fold ' + str(fold_id) + ':')
    print('train_set')
    print(train_set)
    print('test_set')
    print(test_set)

    # (a) testing
    # loading
    fea_temp_orig = np.load(os.path.join(feature_path, 'all_orig.npz'))
    feas_orig = fea_temp_orig['feas_orig']
    labels_orig = fea_temp_orig['labels_orig']
    print('The shape of feas_orig: ', end='')
    print(feas_orig.shape)

    # fea_test = feas_orig[test_set, :, :]  ## << ==
    fea_test = feas_orig[list(test_set), :, :]  ## replace ndarray by list
    label_test = labels_orig[list(test_set)]
    label_test = np.array([species_dict[ll] for ll in label_test])
    print('')
    print(len(test_set))
    print(fea_test.shape)
    print('')

    del feas_orig, labels_orig

    # (b) training
    # loading
    fea_temp_aug = np.load(os.path.join(feature_path, 'all_aug.npz'))
    feas_aug = fea_temp_aug['feas_aug']
    labels_aug = fea_temp_aug['labels_aug']
    print('The shape of feas_aug: ', end='')
    print(feas_aug.shape)

    # augmented features & labels
    fea_ind_aug = []
    for ff in list(train_set):
        for ii in range(copies_of_aug):
            fea_ind_aug.append(ff * copies_of_aug + ii)

    fea_train = feas_aug[fea_ind_aug, :, :]
    label_train = labels_aug[fea_ind_aug]
    label_train = np.array([species_dict[ll] for ll in label_train])
    print('')
    print(len(train_set))
    print(fea_train.shape)
    print('')

    del feas_aug, labels_aug

    # summary
    print('feature train shape: ' + str(x_train.shape))
    print('feature test shape: ' + str(x_test.shape))
    print('label train shape: ' + str(y_train.shape))
    print('label test shape: ' + str(y_test.shape))

    dim_time = x_train.shape[1]
    dim_freq = x_train.shape[2]
    print('dim_time: ' + str(dim_time))
    print('dim_freq: ' + str(dim_freq))

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_species)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_species)
    # y_train = tf.one_hot(y_train, num_species)
    # y_test = tf.one_hot(y_test, num_species)

    # shuffle features & labels
    x_train, y_train = shuffle(x_train, y_train, random_state=random_list[fold_id])
    x_test, y_test = shuffle(x_test, y_test, random_state=random_list[fold_id])

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.10,
                                                                            random_state=random_list[fold_id])

    # train_generator = DataGenerator(x_train, y_train, batch_size=batch_size, num_classes=num_species)
    # del x_train
    # validate_generator = DataGenerator(x_validate, y_validate, batch_size=batch_size, num_classes=num_species)
    # del x_validate

    train_ds_one = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size)
    )
    train_ds_two = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size)
    )
    del x_train  #, y_train

    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
    # train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_one))
    del train_ds_one, train_ds_two
    val_ds = tf.data.Dataset.from_tensor_slices((x_validate, y_validate)).batch(batch_size)
    del x_validate
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    del x_test

    # First create the new dataset using our `mix_up` utility
    train_ds_mu = train_ds.map(
        # lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2)
    )
    del train_ds

    """
    ## Model building
    """
    # initial_model = get_training_model()
    # initial_model.save_weights("initial_weights.h5")
    """
    ## 1. Train the model with the mixed up dataset
    """
    # model = get_training_model()
    # model.load_weights("initial_weights.h5")

    # model_cnn14_attention_multi
    # model = model_cnn14_attention_multi(dim_time, dim_freq, num_species, conv_dim=conv_dim, pool_size=pool_size,
    #                         pool_stride=pool_stride, hidden_units=hidden_units, l2_regu=l2_regu, drop_rate=drop_rate)
    # model_cnn14_spp
    model = model_cnn14_spp(dim_time, dim_freq, num_species, conv_dim=conv_dim, pool_size=pool_size,
                            pool_stride=pool_stride, hidden_units=hidden_units, l2_regu=l2_regu, drop_rate=drop_rate)

    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.fit(train_ds_mu, validation_data=val_ds, epochs=num_epoch)
    # _, test_acc = model.evaluate(test_ds)
    # print("Test accuracy: {:.2f}%".format(test_acc * 100))

    loss = tf.keras.losses.binary_crossentropy
    # loss = BinaryFocalLoss(gamma=2)
    # deployment folder
    fit_result_path2 = os.path.join(fit_result_path1, 'fold' + str(fold_id))
    if not os.path.exists(fit_result_path2):
        makedirs(fit_result_path2)
    # class weight
    y_train0 = np.argmax(y_train, axis=1)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train0), y=y_train0)
    class_weights = dict()
    for ii in range(num_species):
        class_weights[ii] = weights[ii]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
    history = model.fit(train_ds_mu, validation_data=val_ds, class_weight=class_weights, epochs=num_epoch, callbacks=[
        EarlyStopping(patience=num_patience, monitor='val_accuracy', mode='max', verbose=1),
        TensorBoard(log_dir=fit_result_path2),
        ModelCheckpoint(filepath=os.path.join(fit_result_path2, 'epoch_{epoch:02d}_valloss_{val_loss:.4f}_valacc_{val_accuracy:.4f}.hdf5' ), verbose=1, monitor="val_accuracy", save_best_only=True)])

    # Testing
    _, test_acc = model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
    the_best_model, _ = find_best_model(fit_result_path2, purge=False)
    model = load_model(the_best_model)
    y_pred = model.predict(test_ds)

    # save the testing results
    # np.savez(os.path.join(fit_result_path1, ee + '_test_results.npz'), label_test=np.argmax(y_test, axis=1), label_pred=y_pred)
    np.savez(os.path.join(fit_result_path1, 'fold' + str(fold_id) + '_test_results.npz'), label_test=np.argmax(y_test, axis=1), label_pred=y_pred)
    label_pred_all.append(y_pred)
    label_test_all.append(np.argmax(y_test, axis=1))

    del train_ds_mu, val_ds

    fold_id += 1

label_pred_all = np.concatenate(label_pred_all)
label_test_all = np.concatenate(label_test_all)

# """
# **Note** that here , we are combining two images to create a single one. Theoretically,
# we can combine as many we want but that comes at an increased computation cost. In
# certain cases, it may not help improve the performance as well.
# """
#
# """
# ## Visualize the new augmented dataset
# """
#
# if False:
#     # Let's preview 9 samples from the dataset
#     sample_images, sample_labels = next(iter(train_ds_mu))
#     plt.figure(figsize=(10, 10))
#     for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(image.numpy().squeeze())
#         print(label.numpy().tolist())
#         plt.axis("off")
#
# """
# For the sake of reproducibility, we serialize the initial random weights of our shallow
# network.
# """
# if False:
#     initial_model = get_training_model()
#     initial_model.save_weights("initial_weights.h5")
#
#     """
#     ## 1. Train the model with the mixed up dataset
#     """
#
#     model = get_training_model()
#     model.load_weights("initial_weights.h5")
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     model.fit(train_ds_mu, validation_data=val_ds, epochs=num_epoch)
#     _, test_acc = model.evaluate(test_ds)
#     print("Test accuracy: {:.2f}%".format(test_acc * 100))
#
#     """
#     ## 2. Train the model *without* the mixed up dataset
#     """
#
#     model = get_training_model()
#     model.load_weights("initial_weights.h5")
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     # Notice that we are NOT using the mixed up dataset here
#     model.fit(train_ds_one, validation_data=val_ds, epochs=num_epoch)
#     _, test_acc = model.evaluate(test_ds)
#     print("Test accuracy: {:.2f}%".format(test_acc * 100))
#
# """
# Readers are encouraged to try out mixup on different datasets from different domains and
# experiment with the lambda parameter. You are strongly advised to check out the
# [original paper](https://arxiv.org/abs/1710.09412) as well - the authors present several ablation studies on mixup
# showing how it can improve generalization, as well as show their results of combining
# more than two images to create a single one.
# """
#
# """
# ## Notes
#
# * With mixup, you can create synthetic examples — especially when you lack a large
# dataset - without incurring high computational costs.
# * [Label smoothing](https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/) and mixup usually do not work well together because label smoothing
# already modifies the hard labels by some factor.
# * mixup does not work well when you are using [Supervised Contrastive
# Learning](https://arxiv.org/abs/2004.11362) (SCL) since SCL expects the true labels
# during its pre-training phase.
# * A few other benefits of mixup include (as described in the [paper](https://arxiv.org/abs/1710.09412)) robustness to
# adversarial examples and stabilized GAN (Generative Adversarial Networks) training.
# * There are a number of data augmentation techniques that extend mixup such as
# [CutMix](https://arxiv.org/abs/1905.04899) and [AugMix](https://arxiv.org/abs/1912.02781).
# """