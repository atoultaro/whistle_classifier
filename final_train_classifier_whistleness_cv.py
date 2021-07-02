#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train a classifier using the dclde 2011 data

Created on 01/12/21
@author: atoultaro
"""
import os
import numpy as np
from math import floor

from lib_preprocess import bin_extract, contour_target_retrieve
from lib_validation import fea_ext_dcldc2011, all_data_train_validate, one_fold_validate


# bin files for training & testing
# bin_dir = '/home/ys587/__Data/__whistle/__dclde2011/__dclde2011_tonals_20190210/label_bin_files/cv4'
bin_dir = '/home/ys587/__Data/__whistle/__whislte_30_species/__dclde2011_label_bin_files/cv4'
bin_dir_fold = dict()
for pp in range(4):  # 'bin_dir_fold['pie1']'
    bin_dir_fold['pie'+str(pp+1)] = os.path.join(bin_dir, 'pie'+str(pp+1))

sound_dir = '/home/ys587/__Data/__whistle/__whislte_30_species/__sound_48k/__whistle_dclde2011/wav'
species_name = ['bottlenose', 'common', 'spinner', 'melon-headed']
species_id = {'bottlenose': 0, 'common': 1, 'spinner': 2, 'melon-headed': 3}

conf_gen = dict()
conf_gen['save_dir'] = '/home/ys587/__Data/__whistle/__whislte_30_species/__feature_whistleness'
conf_gen['log_dir'] = '/home/ys587/__Data/__whistle/__whislte_30_species/__fit_result_whistleness'

# fea_type = 'pcen_nopulse'
# fea_type = 'pcen'
conf_gen['bin_dir'] = bin_dir
conf_gen['patience'] = 40
conf_gen['species_name'] = species_name
conf_gen['species_id'] = species_id
conf_gen['time_reso'] = 0.02  # 20 ms

# cepstral coefficient
# conf_gen['sample_rate'] = 192000
conf_gen['sample_rate'] = 48000
# conf_gen["num_class"] = len(species_name)
conf_gen["num_class"] = 2-1

# conf_gen['context_winsize'] = 1.0  # sec
# conf_gen['context_hopsize'] = 0.1  # sec
# conf_gen['contour_timethre'] = 20  # 0.4 s

conf_gen['context_winsize'] = 2.0  # sec
conf_gen['context_hopsize'] = 0.2  # sec
conf_gen['contour_timethre'] = 25  # 25 for 0.5 s

# conf_gen['fft_size'] = 4096
conf_gen['hop_length'] = int(conf_gen['time_reso']*conf_gen['sample_rate'])

conf_gen['img_t'] = int(floor((conf_gen['context_winsize'] / conf_gen['time_reso'])))
conf_gen['f_low'] = 0
conf_gen['img_f'] = 128 - conf_gen['f_low']
conf_gen['input_shape'] = (conf_gen['img_t'], conf_gen['img_f'], 1)

conf_gen['l2_regu'] = 0.00
# conf_gen['l2_regu'] = 0.01

# conf_gen['dropout'] = 0.5
conf_gen['dropout'] = 0.2

# conf_gen['batch_size'] = 128  # lstm_2lay
# conf_gen['batch_size'] = 32  # resnet 18, 34
conf_gen['batch_size'] = 128
# conf_gen['batch_size'] = 256
conf_gen['epoch'] = 200
# conf_gen['epoch'] = 2  # debug
# conf_gen['learning_rate'] = 0.001
pie_num = 4
conf_gen['pie_num'] = pie_num

conf_gen['confusion_callback'] = False
conf_gen['spectro_dilation'] = False

conf_gen['numpy_data_use'] = True
conf_gen['img_data_output'] = False  # output image of spectrogram data

# add one more class 'noise': 4 species class + 1 noise class
conf_gen['class_noise'] = True  # add the fifth class: noise

for pp in range(pie_num):  # 'pie1_data.npz'
    conf_gen['save_file_pie'+str(pp+1)] = os.path.join(conf_gen['save_dir'],
                                               'pie'+str(pp+1)+'_data.npz')
for pp in range(pie_num):
    conf_gen['image_pie' + str(pp + 1)] = os.path.join(conf_gen['save_dir'],
                                                           'pie' + str(pp + 1))

conf_gen['network_type'] = 'cnn'
# conf_gen['recurrent_dropout'] = 0.01
# conf_gen['dense_size'] = 128

# Read species labels, filenames & extract time and frequency sequences
contour_pie_list = []
bin_wav_pair_pie_list = []
for pp in range(pie_num):
    contour_pie_curr, bin_wav_pair_pie_curr = bin_extract(
        bin_dir_fold['pie'+str(pp+1)], sound_dir, species_name)
    contour_pie_list.append(contour_pie_curr)
    bin_wav_pair_pie_list.append(bin_wav_pair_pie_curr)

# read contours from bin files
print('----read contours from bin files')
contour_pie_list_alllist =  []
for pp in range(pie_num):
    contour_pie_list_curr = contour_target_retrieve(contour_pie_list[pp],
                                                bin_dir_fold['pie'+str(pp+1)], conf_gen['time_reso'])
    contour_pie_list_alllist.append(contour_pie_list_curr)

# prepare training & testing data
if conf_gen['class_noise']:
    conf_gen['species_name'].append('noise')
    conf_gen['species_id'].update({'noise': 4})
    conf_gen["num_class"] += 1

# feature extraction: pie 1 - 4
whistle_image_pie_list = []
label_pie_list = []
for pp in range(pie_num):
    pie_curr_data_path = os.path.join(conf_gen['save_dir'], 'pie'+str(pp+1)+'_data.npz')
    if os.path.exists(pie_curr_data_path) & conf_gen['numpy_data_use']:
        print('Loading pie '+str(pp)+' data...')
        # data_temp = np.load(pie_curr_data_path)
        data_temp = np.load(pie_curr_data_path, allow_pickle=True)
        whistle_image_pie_curr = data_temp['whistle_image']
        label_pie_curr = data_temp['label'].tolist()
    else:  # extract features
        print("----extract features")
        whistle_image_pie_curr, label_pie_curr, _, _ = fea_ext_dcldc2011(
            contour_pie_list_alllist[pp], sound_dir, conf_gen, conf_gen['image_pie'+str(pp+1)],
            conf_gen['save_file_pie'+str(pp+1)])
    whistle_image_pie_list.append(whistle_image_pie_curr)
    label_pie_list.append(label_pie_curr)

# Change the dimensions
if conf_gen['network_type'] == 'cnn':
    whistle_image_pie_4d_list = []
    for pp in range(pie_num):
        whistle_image_pie_curr_4d = np.expand_dims(whistle_image_pie_list[pp], axis=3)
        whistle_image_pie_4d_list.append(whistle_image_pie_curr_4d)
else:
    raise('Only CNN is supported.')


# training begins here:
learning_rate_list = []
# for ii_lr in [3.33e-3, 1.0e-3, 3.33e-2, 1.0e-2, 3.3e-4]:
# for ii_lr in [1.0e-2, 1.0e-3, 1.0e-4, 3.33e-3, 3.33e-4]:
for ii_lr in [1.0e-3, 3.33e-3]:
    for ii_l2 in [0.0]:
        learning_rate_list.append(ii_lr)

for rr in range(len(learning_rate_list)):
    conf_gen['learning_rate'] = learning_rate_list[rr]
    conf_gen['num_filters'] = 16

    # # First case
    # # all data mixed; 80% train & 20% validate
    # model_type = 'resnet34_expt'
    # proj_name = model_type + '_alldata_run' + str(rr) + '_f1'
    # print(proj_name)
    # conf_gen['comment'] = proj_name+'_lr_'+str(conf_gen['learning_rate'])
    #
    # # all data
    # # train data
    # whistle_image_train_all_4d = np.vstack((whistle_image_pie_4d_list[0],
    #                                           whistle_image_pie_4d_list[1],
    #                                           whistle_image_pie_4d_list[2],
    #                                           whistle_image_pie_4d_list[3]))
    #
    # # # temp; freq dim change
    # # if whistle_image_train_all_4d.shape[2] == 128:
    # #     whistle_image_train_all_4d = whistle_image_train_all_4d[:, :, conf_gen['f_low']:, :]
    #
    # label_train_all0 = label_pie_list[0] + label_pie_list[1] + label_pie_list[2] + label_pie_list[3]
    # label_train_all = [1 if (x == 0) | (x == 1) | (x == 2) | (x == 3) else 0
    #                    for x in label_train_all0]
    #
    # # shuffle
    # from sklearn.utils import shuffle
    #
    # whistle_image_train_all_4d, label_train_all = shuffle(
    #     whistle_image_train_all_4d, label_train_all)
    #
    # best_model_path1 = all_data_train_validate(model_type,
    #                                           whistle_image_train_all_4d,
    #                                           label_train_all, conf_gen)
    #
    # # Second case
    # # three-quarter train & one-quarter validation
    # proj_name = model_type + '_3quarters_run' + str(rr) + '_f1'
    # print(proj_name)
    # conf_gen['comment'] = proj_name+'_lr_'+str(conf_gen['learning_rate'])
    #
    # # train data
    # whistle_image_train_fold1_4d = np.vstack((
    #     whistle_image_pie_4d_list[0],
    #     whistle_image_pie_4d_list[1],
    #     whistle_image_pie_4d_list[2]))
    # label_train_all0 = label_pie_list[0] + label_pie_list[1] + \
    #                    label_pie_list[2]
    # label_train_all = [
    #     1 if (x == 0) | (x == 1) | (x == 2) | (x == 3) else 0 for x in
    #     label_train_all0]
    #
    # # shuffle
    # from sklearn.utils import shuffle
    # whistle_image_train_fold1_4d, label_train_all = shuffle(whistle_image_train_fold1_4d, label_train_all)
    #
    # # validation data
    # whistle_image_validate_fold1_4d = whistle_image_pie_4d_list[3]
    # # del whistle_image_pie_4d_list[3]
    # label_validate_all = [1 if (x == 0) | (x == 1) | (x == 2) | (x == 3) else 0 for x in label_pie_list[3]]
    # whistle_image_validate_fold1_4d, label_validate_all = shuffle(whistle_image_validate_fold1_4d, label_validate_all)
    #
    # y_pred2, y_pred_prob, best_model_path2 = \
    #     one_fold_validate(model_type, whistle_image_train_fold1_4d,
    #                       label_train_all, whistle_image_validate_fold1_4d,
    #                       label_validate_all, conf_gen)


    # Third case
    model_type = 'resnet18_expt'
    proj_name = model_type + '_alldata_run' + str(rr) + '_f1'
    print(proj_name)
    conf_gen['comment'] = proj_name+'_lr_'+str(conf_gen['learning_rate'])

    # all data
    # train data
    whistle_image_train_all_4d = np.vstack((whistle_image_pie_4d_list[0],
                                              whistle_image_pie_4d_list[1],
                                              whistle_image_pie_4d_list[2],
                                              whistle_image_pie_4d_list[3]))

    label_train_all0 = label_pie_list[0] + label_pie_list[1] + label_pie_list[2] + label_pie_list[3]
    label_train_all = [1 if (x == 0) | (x == 1) | (x == 2) | (x == 3) else 0
                       for x in label_train_all0]

    # shuffle
    from sklearn.utils import shuffle

    whistle_image_train_all_4d, label_train_all = shuffle(
        whistle_image_train_all_4d, label_train_all)

    best_model_path1 = all_data_train_validate(model_type,
                                              whistle_image_train_all_4d,
                                              label_train_all, conf_gen)

    # Fourth case
    # three-quarter train & one-quarter validation
    model_type = 'resnet18_expt'
    proj_name = model_type + '_3quarters_run' + str(rr) + '_f1'
    print(proj_name)
    conf_gen['comment'] = proj_name+'_lr_'+str(conf_gen['learning_rate'])

    # train data
    whistle_image_train_fold1_4d = np.vstack((
        whistle_image_pie_4d_list[0],
        whistle_image_pie_4d_list[1],
        whistle_image_pie_4d_list[2]))
    label_train_all0 = label_pie_list[0] + label_pie_list[1] + \
                       label_pie_list[2]
    label_train_all = [
        1 if (x == 0) | (x == 1) | (x == 2) | (x == 3) else 0 for x in
        label_train_all0]

    # shuffle
    from sklearn.utils import shuffle
    whistle_image_train_fold1_4d, label_train_all = shuffle(whistle_image_train_fold1_4d, label_train_all)

    # validation data
    whistle_image_validate_fold1_4d = whistle_image_pie_4d_list[3]
    # del whistle_image_pie_4d_list[3]
    label_validate_all = [1 if (x == 0) | (x == 1) | (x == 2) | (x == 3) else 0 for x in label_pie_list[3]]
    whistle_image_validate_fold1_4d, label_validate_all = shuffle(whistle_image_validate_fold1_4d, label_validate_all)

    y_pred2, y_pred_prob, best_model_path2 = \
        one_fold_validate(model_type, whistle_image_train_fold1_4d,
                          label_train_all, whistle_image_validate_fold1_4d,
                          label_validate_all, conf_gen)
