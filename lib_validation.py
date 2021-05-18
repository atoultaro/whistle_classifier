#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whislte classification using audio signals
4-fold cross-validation

Created on 12/9/19
@author: atoultaro
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
import gc
import lib_feature
import datetime
import re

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import librosa
from tensorflow.keras import backend, utils
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

from lib_feature import data_generator
# from whistle_classifier.capecod_classifier import metrics_two_fold, find_best_model

from contextlib import redirect_stdout
import itertools
from lib_model import resnet34_expt, resnet18_expt


# data generator
class DataGenerator(utils.Sequence):
    def __init__(self, feature, label, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.X = feature
        self.X_dim = len(feature.shape)
        self.y = to_categorical(label, num_classes)
        self.indices = np.arange(self.y.shape[0])
        self.num_classes = num_classes
        self.shuffle = shuffle

        # self.index = np.arange(len(self.indices))
        # self.df = dataframe
        # self.indices = self.df.index.tolist()
        # self.x_col = x_col
        # self.y_col = y_col

        self.on_epoch_end()

    def __len__(self):
        return int(floor(len(self.indices) / self.batch_size))  # return label.shape[0]

    def __getitem__(self, index):
        # index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        # batch = [self.indices[k] for k in index]
        batch = list(range(index * self.batch_size, (index + 1) * self.batch_size))

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __get_data(self, batch):
        y = np.zeros((self.batch_size, self.y.shape[1]))

        if self.X_dim == 3:
            X = np.zeros((self.batch_size, self.X.shape[1], self.X.shape[2]))
            for i, id in enumerate(batch):
                X[i, :, :] = self.X[id, :, :]  # logic
                y[i, :] = self.y[id, :]  # labels

        elif self.X_dim == 4:
            X = np.zeros((self.batch_size, self.X.shape[1], self.X.shape[2], self.X.shape[3]))
            for i, id in enumerate(batch):
                X[i, :, :, :] = self.X[id, :, :, :]  # logic
                y[i, :] = self.y[id, :]  # labels

        return X, y


def contour_data(file_contour, time_reso):
    print('Retrieving contours...')
    contour_target_ff = []
    len_contour = len(file_contour)
    print('len_contour: '+str(len_contour))
    time_min = 86400.0
    time_max = 0.0
    freq_high = 0.0
    freq_low = 192000.0
    # read contours into the var contour_target_ff
    for cc in range(len_contour):
        time_contour = file_contour[cc]['Time']
        freq_contour = file_contour[cc]['Freq']

        if time_contour.shape[0] > 1:
            # linear interpolation
            # time_contour_interp = np.arange(time_contour[0], time_contour[-1],
            #                                 time_reso)
            new_start_time = round(time_contour[0]/time_reso)*time_reso
            new_step = ceil((time_contour[-1] - time_contour[0])/time_reso)
            time_contour_interp = np.arange(new_start_time, new_start_time+new_step*time_reso, time_reso)

            time_min = np.min((time_contour_interp[0], time_min))
            time_max = np.max((time_contour_interp[-1], time_max))

            freq_contour_interp = np.interp(time_contour_interp, time_contour,
                                            freq_contour)
            freq_high = np.max((np.max(freq_contour_interp), freq_high))
            freq_low = np.min((np.min(freq_contour_interp), freq_low))

            contour_target_ff_cc = dict()
            contour_target_ff_cc['Time'] = time_contour_interp
            contour_target_ff_cc['Freq'] = freq_contour_interp

            contour_target_ff.append(contour_target_ff_cc)

    return contour_target_ff, time_min, time_max, freq_low, freq_high


def fea_ext_dcldc2011(contour_target_list, sound_dir, conf, img_folder, save_file, plot=False):
    '''
    Convert whistle contours into sequences of fixed length
    :param contour_target_list:
    time_reso = 0.1, context_winsize=10.0, ratio_thre=0.02
    :return:
    df_target:
    # , freq_ind_low=64, fmin=4000.0, bins_per_octave=36, n_bins=144
    '''
    freq_low_all = 192000.0  # an realistic upperbound
    freq_high_all = 0.0
    whistle_image_list = []
    label_list = []

    if plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
            for ss in conf['species_name']:
                os.mkdir(os.path.join(img_folder, ss))

    data_count = 1
    for ff in range(len(contour_target_list)):
        filename = contour_target_list[ff][0]
        print('\n'+filename)
        label_contour = contour_target_list[ff][1]
        file_contour = contour_target_list[ff][2]

        contour_target_ff, start_time, end_time, freq_low, freq_high = \
            contour_data(file_contour, conf['time_reso'])
        freq_high_all = np.max((freq_high, freq_high_all))
        freq_low_all = np.min((freq_low, freq_low_all))

        timesteps = ceil((end_time - start_time)/conf['time_reso'])+1
        print("Start time: "+str(start_time))
        print("Stop time: " + str(end_time))

        # spectrogram named whistle_freq for each file
        sound_path = os.path.join(sound_dir, label_contour, filename+'.wav')
        samples, _ = librosa.load(sound_path, sr=conf['sample_rate'], offset=start_time, duration=end_time-start_time+2.*conf['time_reso'])
        whistle_freq = librosa.feature.melspectrogram(samples,
                                                          sr=conf['sample_rate'],
                                                          hop_length=conf['hop_length'],
                                                          power=1)

        whistle_presence = np.zeros((int(timesteps)))
        for cc in contour_target_ff:
            time_ind_start = int(floor((cc['Time'][0]-start_time)/conf['time_reso']))
            for ii in range(cc['Time'].shape[0]):
                whistle_presence[time_ind_start+ii] = 1.0

        # cut whistle_freq into segments for data samples
        size_time = int(conf['context_winsize']/conf['time_reso'])
        size_hop = int(conf['context_hopsize']/conf['time_reso'])
        # freq_high = whistle_freq.shape[0]
        # freq_low = conf['freq_ind_low']
        # freq_low = conf['freq_low']
        for tt in range(floor((whistle_freq.shape[1]-size_time)/size_hop)):
            # whistle_image = whistle_freq[freq_low:freq_high, tt*size_hop:tt*size_hop+size_time]
            whistle_image = whistle_freq[:, tt*size_hop:tt*size_hop+size_time]
            whistle_presence_seg = whistle_presence[tt * size_hop:tt * size_hop + size_time]

            if whistle_presence_seg.sum() >= conf['contour_timethre']:
                # feature extraction
                whistle_image = lib_feature.feature_whistleness(whistle_image)  # magnitude normalization?s
                # whistle_median = species_lib.nopulse_median(whistle_image)
                # whistle_image = (species_lib.avg_sub(whistle_median)).T

                whistle_image_list.append(whistle_image)

                label_list.append(conf['species_id'][label_contour])

                if plot is True:
                    ax.matshow(whistle_image, origin='lower')
                    ax.title.set_text(str(data_count)+': '+label_contour)
                    ax.xaxis.tick_bottom()
                    fig.canvas.draw()
                    plt.savefig(os.path.join(img_folder, label_contour, str(data_count)+'_'+label_contour+'.png'))
                data_count += 1  # ? complex when having noise class
                # print('stop for images')
            elif conf['class_noise'] & (whistle_presence_seg.sum() == 0.0):  # no labels here!

                # feature extraction
                whistle_image = lib_feature.feature_whistleness(whistle_image)
                # whistle_median = species_lib.nopulse_median(whistle_image)
                # whistle_image = (species_lib.avg_sub(whistle_median)).T

                whistle_image_list.append(whistle_image)

                label_list.append(conf['species_id']['noise'])
                data_count += 1
                # noise class image here!
    whistle_image = np.asarray(whistle_image_list)
    np.savez(save_file, whistle_image=whistle_image, label=label_list)

    return whistle_image, label_list, freq_high_all, freq_low_all


# def prepare_data(contour_target_list, conf, img_folder, save_file, plot=False):
#     '''
#     Convert whistle contours into sequences of fixed length
#     :param contour_target_list:
#     time_reso = 0.1, context_winsize=10.0, ratio_thre=0.02
#     :return:
#     df_target:
#     '''
#     freq_low_all = 192000.0
#     freq_high_all = 0.0
#     whistle_image_list = []
#     label_list = []
#
#     if plot:
#         plt.ion()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#
#         if not os.path.exists(img_folder):
#             os.mkdir(img_folder)
#             for ss in conf['species_name']:
#                 os.mkdir(os.path.join(img_folder, ss))
#
#     data_count = 1
#     for ff in range(len(contour_target_list)):
#         filename = contour_target_list[ff][0]
#         print('\n'+filename)
#         label_contour = contour_target_list[ff][1]
#         file_contour = contour_target_list[ff][2]
#
#         contour_target_ff, start_time, end_time, freq_low, freq_high = \
#             contour_data(file_contour, conf['time_reso'])
#         freq_high_all = np.max((freq_high, freq_high_all))
#         freq_low_all = np.min((freq_low, freq_low_all))
#
#         timesteps = ceil((end_time - start_time)/conf['time_reso'])+1
#         print("Start time: "+str(start_time))
#         print("Stop time: " + str(end_time))
#
#         # Binary spectrogram
#         # convert whistle freq into into a 2d feature map: whistle_freq for each file
#         whistle_freq = np.zeros((conf['fft_size'], int(timesteps)))
#         for cc in contour_target_ff:
#             time_ind_start = int(floor((cc['Time'][0]-start_time)/conf['time_reso']))
#             freq_ind = (np.floor(cc['Freq']/conf['sample_rate']*conf['fft_size'])).astype('int')
#             for ii in range(cc['Time'].shape[0]):
#                 try:
#                     whistle_freq[freq_ind[ii], time_ind_start+ii] = 1.0
#                 except:
#                     print('stop!')
#
#         if conf['spectro_dilation']:
#             # whistle_freq_smooth = cv2.GaussianBlur(whistle_freq, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
#             kernel = np.ones((3, 3)).astype(np.uint8)
#             kernel[0, 0] = 0
#             kernel[0, 2] = 0
#             kernel[2, 0] = 0
#             kernel[2, 2] = 0
#             whistle_freq_smooth = cv2.dilate(whistle_freq, kernel, cv2.BORDER_DEFAULT)
#             whistle_freq = whistle_freq_smooth
#
#         # plt.matshow(whistle_freq); plt.show()
#         # print('whistle_freq shape: '+str(whistle_freq.shape[0])+', '+str(whistle_freq.shape[1]) )
#
#         # cut whistle_freq into segments for data samples
#         size_time = int(conf['context_winsize']/conf['time_reso'])
#         size_hop = int(conf['context_hopsize']/conf['time_reso'])
#         freq_high = conf['freq_ind_high']
#         freq_low = conf['freq_ind_low']
#         for tt in range(floor((whistle_freq.shape[1]-size_time)/size_hop)):
#             whistle_image = whistle_freq[freq_low:freq_high, tt*size_hop:tt*size_hop+size_time]
#             # if whistle_image.sum()/whistle_image.shape[0]/whistle_image.shape[1] >= 0.01:
#             # if whistle_image.sum() >= 0.1*conf['img_t']:
#             # print('whistle_image.sum: '+str(whistle_image.sum()))
#             # plt.draw()
#             # plt.pause(0.0001)
#             # plt.clf()
#             # plt.show()
#             if (whistle_image.sum(axis=0) > 0).sum() >= conf['contour_timethre']:
#                 whistle_image_list.append(whistle_image)
#                 label_list.append(conf['species_id'][label_contour])
#
#                 if plot is True:
#                     ax.matshow(whistle_image, origin='lower')
#                     ax.title.set_text(str(data_count)+': '+label_contour)
#                     ax.xaxis.tick_bottom()
#                     fig.canvas.draw()
#                     plt.savefig(os.path.join(img_folder, label_contour, str(data_count)+'_'+label_contour+'.png'))
#                 data_count += 1
#                 # print('stop for images')
#
#     whistle_image_arr = np.asarray(whistle_image_list)
#     # save image array
#     if conf['numpy_data_output']:
#         np.savez(save_file, whistle_image=whistle_image_arr, label=label_list)
#
#     return whistle_image_arr, label_list, freq_high_all, freq_low_all
#
#
# def prepare_data_mask(contour_target_list, sound_dir, conf, img_folder, save_file, plot=False):
#     '''
#     Convert whistle contours into sequences of fixed length
#     :param contour_target_list:
#     time_reso = 0.1, context_winsize=10.0, ratio_thre=0.02
#     :return:
#     df_target:
#     '''
#     freq_low_all = 192000.0  # an realistic upperbound
#     freq_high_all = 0.0
#     whistle_image_list = []
#     label_list = []
#
#     if plot:
#         plt.ion()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#
#         if not os.path.exists(img_folder):
#             os.mkdir(img_folder)
#             for ss in conf['species_name']:
#                 os.mkdir(os.path.join(img_folder, ss))
#
#     data_count = 1
#     for ff in range(len(contour_target_list)):
#         filename = contour_target_list[ff][0]
#         print('\n'+filename)
#         label_contour = contour_target_list[ff][1]
#         file_contour = contour_target_list[ff][2]
#
#         contour_target_ff, start_time, end_time, freq_low, freq_high = \
#             contour_data(file_contour, conf['time_reso'])
#         freq_high_all = np.max((freq_high, freq_high_all))
#         freq_low_all = np.min((freq_low, freq_low_all))
#
#         timesteps = ceil((end_time - start_time)/conf['time_reso'])+1
#         print("Start time: "+str(start_time))
#         print("Stop time: " + str(end_time))
#
#         # spectrogram named whistle_freq for each file
#         sound_path = os.path.join(sound_dir, label_contour, filename+'.wav')
#         samples, _ = librosa.load(sound_path, sr=conf['sample_rate'], offset=start_time, duration=end_time-start_time+2.*conf['time_reso'])
#         whistle_freq0 = np.abs(librosa.pseudo_cqt(samples, sr=conf['sample_rate'],
#                                           hop_length=conf['hop_length'],
#                                           fmin=4000.0, bins_per_octave=36,
#                                           n_bins=144))
#
#         # Masked spectrogram
#         whistle_freq = np.zeros((whistle_freq0.shape[0], whistle_freq0.shape[1]))
#         for cc in contour_target_ff:
#             time_ind_start = int(floor((cc['Time'][0]-start_time)/conf['time_reso']))
#             # freq_ind = (np.floor(cc['Freq']/conf['sample_rate']*conf['fft_size'])).astype('int')
#             freq_ind = (np.log2(cc['Freq']/4000.0)*36.).astype('int')
#             freq_ind[freq_ind<0] = 0.0
#             freq_ind[freq_ind >= 144] = 143
#
#             if False: # width-1 mask
#                 for ii in range(cc['Time'].shape[0]):
#                     whistle_freq[freq_ind[ii]-1:freq_ind[ii]+1+1, time_ind_start+ii] = whistle_freq0[freq_ind[ii]-1:freq_ind[ii]+1+1, time_ind_start+ii]
#
#             # rasterization mask
#             for ii in range(1, cc['Time'].shape[0]):
#                 if freq_ind[ii] > freq_ind[ii-1]:
#                     whistle_freq[freq_ind[ii-1]-1:freq_ind[ii]+1+1, time_ind_start+ii-1:time_ind_start+ii+1] = whistle_freq0[freq_ind[ii-1]-1:freq_ind[ii]+1+1, time_ind_start+ii-1:time_ind_start+ii+1]
#                 elif freq_ind[ii] < freq_ind[ii-1]:
#                     whistle_freq[freq_ind[ii]-1:freq_ind[ii-1]+1+1, time_ind_start+ii-1:time_ind_start+ii+1] = whistle_freq0[freq_ind[ii]-1:freq_ind[ii-1]+1+1, time_ind_start+ii-1:time_ind_start+ii+1]
#                 else:  # freq_ind[ii] == freq_ind[ii-1]
#                     whistle_freq[freq_ind[ii]-1:freq_ind[ii]+1+1, time_ind_start+ii-1:time_ind_start+ii+1] = whistle_freq0[freq_ind[ii]-1:freq_ind[ii]+1+1, time_ind_start+ii-1:time_ind_start+ii+1]
#
#         whistle_presence = np.zeros((int(timesteps)))
#         for cc in contour_target_ff:
#             time_ind_start = int(floor((cc['Time'][0]-start_time)/conf['time_reso']))
#             for ii in range(cc['Time'].shape[0]):
#                 whistle_presence[time_ind_start+ii] = 1.0
#
#         # cut whistle_freq into segments for data samples
#         size_time = int(conf['context_winsize']/conf['time_reso'])
#         size_hop = int(conf['context_hopsize']/conf['time_reso'])
#         freq_high = conf['freq_ind_high']
#         freq_low = conf['freq_ind_low']
#         for tt in range(floor((whistle_freq.shape[1]-size_time)/size_hop)):
#             whistle_image = whistle_freq[freq_low:freq_high, tt*size_hop:tt*size_hop+size_time]
#             whistle_presence_seg = whistle_presence[tt * size_hop:tt * size_hop + size_time]
#             # plt.draw()
#             # plt.pause(0.0001)
#             # plt.clf()
#             # plt.show()
#             # if (whistle_image.sum(axis=0) > 0).sum() >= conf['contour_timethre']:
#             if whistle_presence_seg.sum() >= conf['contour_timethre']:
#                 whistle_image_list.append(whistle_image)
#                 label_list.append(conf['species_id'][label_contour])
#
#                 if plot is True:
#                     ax.matshow(whistle_image, origin='lower')
#                     ax.title.set_text(str(data_count)+': '+label_contour)
#                     ax.xaxis.tick_bottom()
#                     fig.canvas.draw()
#                     plt.savefig(os.path.join(img_folder, label_contour, str(data_count)+'_'+label_contour+'.png'))
#                     # plt.clf()
#                 data_count += 1  # ? complex when having noise class
#                 # print('stop for images')
#             elif conf['class_noise'] & (whistle_presence_seg.sum() == 0.0):  # no labels here!
#                 whistle_image_list.append(whistle_image)
#                 label_list.append(conf['species_id']['noise'])
#                 data_count += 1
#                 # noise class image here!
#     whistle_image = np.asarray(whistle_image_list)
#     # save image array
#     if conf['numpy_data_output']:
#         np.savez(save_file, whistle_image=whistle_image, label=label_list)
#
#     # Shuffle: working on
#     # label_idx = [ii for ii in range(len(label_list))]
#     # random.shuffle(label_idx)
#     # whistle_image_new = [whistle_image[tt] for tt in label_idx]
#     # label_list_new = to_categorical(label_list)[label_idx, :]
#
#     return whistle_image, label_list, freq_high_all, freq_low_all


class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch
    # Arguments
        X_val: The input values
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title
    """
    def __init__(self, X_val, Y_val, classes, normalize=False,
                 cmap=plt.cm.Blues, title='Confusion Matrix'):
        self.X_val = X_val
        self.Y_val = Y_val
        self.title = title
        self.classes = classes
        self.normalize = normalize
        self.cmap = cmap
        plt.ion()
        # plt.show()
        plt.figure()

        plt.title(self.title)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        plt.clf()
        pred = self.model.predict(self.X_val)
        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.Y_val, axis=1)
        cnf_mat = confusion_matrix(max_y, max_pred)

        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:,
                                                np.newaxis]

        thresh = cnf_mat.max() / 2.
        for i, j in itertools.product(range(cnf_mat.shape[0]),
                                      range(cnf_mat.shape[1])):
            plt.text(j, i, cnf_mat[i, j], horizontalalignment="center",
                     color="white" if cnf_mat[i, j] > thresh else "black")

        plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.colorbar()

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.draw()
        plt.show()
        plt.pause(0.001)


def one_fold_validate(model_name, whistle_image_target_4d, label_target,
                  whistle_image_validate_4d, label_validate,
                  conf, fold_id=None):
    label_target_cat = to_categorical(label_target)
    label_validate_cat = to_categorical(label_validate)

    model_name_func = globals()[model_name]
    # model = model_name_func((conf['img_f'], conf['img_t'], 1), depth=20, num_class=4, num_stack=3, num_filters=32)
    model = model_name_func(conf)
    # model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}.hdf5'
    model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}_valacc_{val_accuracy:.4f}.hdf5'

    if conf['comment'] is None:
        conf['comment']=''
    log_dir = make_folder_time_now(folder_out=conf['log_dir'], folder_comment=conf['comment'])

    # if not os.path.exists(log_dir1):
    #     os.mkdir(log_dir1)
    check_path = os.path.join(log_dir, model_name_format)

    if fold_id == 1:
        with open(os.path.join(log_dir, 'architecture.txt'), 'w') as f:
            with redirect_stdout(f):
                # print('')
                for kk in sorted(list(conf.keys())):
                    print(kk + ' ==>> ' + str(conf[kk]))
                model.summary()

    # checkpoint
    checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=1,
                                 save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=conf['patience'])
    if conf['confusion_callback']:
        cm_plot = ConfusionMatrixPlotter(whistle_image_validate_4d,
                                         label_validate_cat, conf['species_name'])
    # model compile
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(lr=conf['learning_rate']),
                  # optimizer=Adam(lr=conf['learning_rate']),
                  metrics=['accuracy'])
    model.summary()

    count_species = label_target_cat.sum(axis=0)+1e-6
    weight_curr = (count_species.max() / count_species).tolist()
    conf["class_weight"] = {0: weight_curr[0], 1: weight_curr[1]}

    if conf['confusion_callback']:
        callback_list = [checkpoint, TensorBoard(log_dir=log_dir), cm_plot,
                         early_stop]
    else:
        callback_list = [checkpoint, TensorBoard(log_dir=log_dir), early_stop]

    model.fit(whistle_image_target_4d, label_target_cat,
              batch_size=conf['batch_size'], epochs=conf['epoch'],
              verbose=1, validation_split=0.2,
              callbacks=callback_list, class_weight=conf["class_weight"])
    # re_model_name_format = 'epoch_\d+_valloss_(\d+.\d{4}).hdf5'
    re_model_name_format = 'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
    best_model_path, _ = find_best_model(log_dir, re_model_name_format,
                                         is_max=False, purge=True)
    conf['best_model'] = best_model_path
    model = load_model(best_model_path)
    y_pred_prob = model.predict(whistle_image_validate_4d)
    y_pred2 = np.argmax(y_pred_prob, axis=1)
    metrics_two_fold(label_validate, y_pred2, log_dir, 'accuracy_fold.txt',
                     conf, mode='fold')

    np.savetxt(os.path.join(log_dir, 'pred_label.txt'), y_pred2, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(log_dir, 'pred_prob.txt'), y_pred_prob, delimiter=',', fmt='%.6f')

    del model
    gc.collect()
    backend.clear_session()

    return y_pred2, y_pred_prob, best_model_path


# def one_fold_validate_fit_only(model_name, whistle_image_target_4d,
#                                label_target, conf, fold_id=1):
#     label_target_cat = to_categorical(label_target)
#     # label_validate_cat = to_categorical(label_validate)
#
#     model_name_func = globals()[model_name]
#     # model = model_name_func((conf['img_f'], conf['img_t'], 1), depth=20, num_class=4, num_stack=3, num_filters=32)
#     model = model_name_func(conf)
#     model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}_valacc_{val_acc:.4f}.hdf5'
#     log_dir1 = os.path.join(conf['log_dir'], 'fold'+str(fold_id))
#     if not os.path.exists(log_dir1):
#         os.mkdir(log_dir1)
#     check_path = os.path.join(log_dir1, model_name_format)
#
#     if fold_id == 1:
#         with open(os.path.join(conf['log_dir'], 'architecture.txt'), 'w') as f:
#             with redirect_stdout(f):
#                 # print('')
#                 for kk in sorted(list(conf.keys())):
#                     print(kk + ' ==>> ' + str(conf[kk]))
#                 model.summary()
#
#     # checkpoint
#     checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=0,
#                                  save_best_only=True)
#     early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
#                                patience=conf['patience'])
#
#     # model compile
#     model.compile(loss=categorical_crossentropy,
#                   optimizer=Adadelta(lr=conf['learning_rate']),
#                   # optimizer=Adam(lr=conf['learning_rate']),
#                   metrics=['accuracy'])
#     model.summary()
#
#     count_species1 = label_target_cat.sum(axis=0).tolist()
#     conf["class_weight"] = (
#                 max(count_species1) / np.array(count_species1)).tolist()
#
#     callback_list = [checkpoint, TensorBoard(log_dir=log_dir1), early_stop]
#
#     model.fit(whistle_image_target_4d, label_target_cat,
#               batch_size=conf['batch_size'], epochs=conf['epoch'],
#               verbose=1, validation_split=0.2,
#               callbacks=callback_list, class_weight=conf["class_weight"])
#     re_model_name_format = 'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
#     best_model_path, _ = find_best_model(log_dir1, re_model_name_format,
#                                          is_max=False, purge=True)
#     conf['best_model'] = best_model_path
#     # model = load_model(best_model_path)
#
#     return best_model_path
#
#
# def one_fold_validate_fit_only_talos(model_name, whistle_image_target_4d,
#                                label_target, conf, params, fold_id=1):
#     label_target_cat = to_categorical(label_target)
#     # label_validate_cat = to_categorical(label_validate)
#
#     model_name_func = globals()[model_name]
#     # model = model_name_func((conf['img_f'], conf['img_t'], 1), depth=20, num_class=4, num_stack=3, num_filters=32)
#     model = model_name_func(conf)
#     model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}_valacc_{val_acc:.4f}.hdf5'
#     log_dir1 = os.path.join(conf['log_dir'], 'fold'+str(fold_id))
#     if not os.path.exists(log_dir1):
#         os.mkdir(log_dir1)
#     check_path = os.path.join(log_dir1, model_name_format)
#
#     if fold_id == 1:
#         with open(os.path.join(conf['log_dir'], 'architecture.txt'), 'w') as f:
#             with redirect_stdout(f):
#                 # print('')
#                 for kk in sorted(list(conf.keys())):
#                     print(kk + ' ==>> ' + str(conf[kk]))
#                 model.summary()
#
#     # checkpoint
#     checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=0,
#                                  save_best_only=True)
#     early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
#                                patience=conf['patience'])
#
#     # model compile
#     model.compile(loss=categorical_crossentropy,
#                   optimizer=Adadelta(lr=conf['learning_rate']),
#                   # optimizer=Adam(lr=conf['learning_rate']),
#                   metrics=['accuracy'])
#     model.summary()
#
#     count_species1 = label_target_cat.sum(axis=0).tolist()
#     conf["class_weight"] = (
#                 max(count_species1) / np.array(count_species1)).tolist()
#
#     callback_list = [checkpoint, TensorBoard(log_dir=log_dir1), early_stop]
#
#     model.fit(whistle_image_target_4d, label_target_cat,
#               batch_size=conf['batch_size'], epochs=conf['epoch'],
#               verbose=1, validation_split=0.2,
#               callbacks=callback_list, class_weight=conf["class_weight"])
#     re_model_name_format = 'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
#     best_model_path, _ = find_best_model(log_dir1, re_model_name_format,
#                                          is_max=False, purge=True)
#     conf['best_model'] = best_model_path
#     # model = load_model(best_model_path)
#
#     return best_model_path


# def validate_on_files(best_model_path_train, fea_out_fold_validate, conf, fold_id):
#     model = load_model(best_model_path_train)
#
#     fea_species_file_list = glob.glob(os.path.join(fea_out_fold_validate, '*.npz'))
#     label_pred = []
#     label_truth = []
#     file_test = []
#     y_pred_prob_tot_list = []
#     for ff in fea_species_file_list:
#         species_filename = os.path.basename(ff)
#         print(species_filename)
#         fea_curr = np.load(ff)
#         # features
#         fea_file_4d = fea_curr['fea_pos']
#         # fea_file_4d = fea_pos_4d[:, conf['img_f']:, :, :] + np.finfo(float).eps
#         # fea_file_4d = species_lib.unit_vector(fea_file_4d)
#
#         # classification
#         if fea_file_4d.shape[0] == 0:
#             continue
#         else:
#             y_pred_prob = model.predict(fea_file_4d)
#             y_pred_prob_tot = y_pred_prob.mean(axis=0)
#             y_pred_prob_tot_list.append(y_pred_prob_tot)
#             y_pred2 = np.argmax(y_pred_prob_tot[:-1])  # predicted species
#             label_pred.append(y_pred2)
#
#             # find the truth label
#             file_test.append(species_filename)
#             species = species_filename.split('_')[0]
#             label_truth.append(conf['species_id'][species])
#
#     log_dir = os.path.join(conf['log_dir'], 'fold' + str(fold_id))
#
#     # make a dataframe by combining file_test, label_truth, label_pred
#     df_validate = pd.DataFrame(list(zip(file_test, label_truth, label_pred)), columns=['sound_file', 'label_truth', 'label_pred'])
#     df_validate.to_csv(os.path.join(log_dir, 'pred.csv'), index=False)
#
#     metrics_two_fold(label_truth, label_pred, log_dir, 'accuracy_fold.txt',
#                      conf, mode='fold')
#     label_pred = np.array(label_pred)
#     np.savetxt(os.path.join(log_dir, 'pred_label.txt'), label_pred, delimiter=',', fmt='%d')
#     y_pred_prob_arr = np.stack(y_pred_prob_tot_list)
#     np.savetxt(os.path.join(log_dir, 'pred_prob.txt'), y_pred_prob_arr, delimiter=',', fmt='%.6f')
#     label_truth = np.array(label_truth)
#
#     return df_validate, label_pred, y_pred_prob_arr, label_truth


def one_fold_validate_generator(model_name, whistle_image_target_4d,
                                label_target, whistle_image_test_4d,
                                label_test, conf, fold_id=1):
    label_target_cat = to_categorical(label_target)
    label_test_cat = to_categorical(label_test)

    whistle_image_train_4d, whistle_image_validate_4d, label_train_cat, \
    label_validate_cat = train_test_split(whistle_image_target_4d,
                                          label_target_cat, test_size=0.2)
    if conf['network_type'] == 'rnn':
        gen_train = data_generator(whistle_image_train_4d, label_train_cat,
                                   batch_size=conf['batch_size'], network_type='rnn')
    elif conf['network_type'] == 'conv2d_lstm':
        gen_train = data_generator(whistle_image_train_4d, label_train_cat,
                                   batch_size=conf['batch_size'], network_type='conv2d_lstm')
    else:  # cnn
        gen_train = data_generator(whistle_image_train_4d, label_train_cat,
                                    batch_size=conf['batch_size'])
        # gen_validate = data_generator(whistle_image_validate_4d, label_validate_cat,
        #                           batch_size=conf['batch_size'])


    model_name_func = globals()[model_name]
    model = model_name_func(conf)
    model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}_valacc_{val_acc:.4f}.hdf5'
    log_dir1 = os.path.join(conf['log_dir'], 'fold'+str(fold_id))
    if not os.path.exists(log_dir1):
        os.mkdir(log_dir1)
    check_path = os.path.join(log_dir1, model_name_format)

    if fold_id == 1:
        with open(os.path.join(conf['log_dir'], 'architecture.txt'), 'w') as f:
            with redirect_stdout(f):
                # print('')
                for kk in sorted(list(conf.keys())):
                    print(kk + ' ==>> ' + str(conf[kk]))
                model.summary()

    # checkpoint
    checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=1,
                                 save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=conf['patience'])
    if conf['confusion_callback']:
        cm_plot = ConfusionMatrixPlotter(whistle_image_validate_4d,
                                         label_validate_cat, conf['species_name'])
    # model compile
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(lr=conf['learning_rate']),
                  # optimizer=Adam(lr=conf['learning_rate']),
                  metrics=['accuracy'])
    model.summary()

    count_species1 = label_target_cat.sum(axis=0).tolist()
    conf["class_weight"] = (
                max(count_species1) / np.array(count_species1)).tolist()

    if conf['confusion_callback']:
        callback_list = [checkpoint, TensorBoard(log_dir=log_dir1), cm_plot,
                         early_stop]
    else:
        callback_list = [checkpoint, TensorBoard(log_dir=log_dir1), early_stop]

    steps = int(floor(whistle_image_train_4d.shape[0]/conf['batch_size']))
    model.fit_generator(gen_train,
                        epochs=conf['epoch'], verbose=1,
                        # samples_per_epoch=conf['batch_size']*steps,
                        # validation_data=(x_test, y_test_onehot),
                        validation_data=(whistle_image_validate_4d, label_validate_cat),
                        # validation_data=gen_validate,
                        # validation_steps=label_validate_cat.shape[0],
                        # validation_steps=steps_validate,
                        steps_per_epoch=steps, callbacks=callback_list,
                        class_weight=conf["class_weight"])

    re_model_name_format = 'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
    best_model_path, _ = find_best_model(log_dir1, re_model_name_format,
                                         is_max=False, purge=True)
    conf['best_model'] = best_model_path
    model = load_model(best_model_path)
    y_pred_prob = model.predict(whistle_image_test_4d)
    y_pred2 = np.argmax(y_pred_prob, axis=1)
    metrics_two_fold(label_test, y_pred2, log_dir1, 'accuracy_fold.txt',
                     conf, mode='fold')

    np.savetxt(os.path.join(log_dir1, 'pred_label.txt'), y_pred2, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(log_dir1, 'pred_prob.txt'), y_pred_prob, delimiter=',', fmt='%.6f')

    del model
    gc.collect()
    backend.clear_session()

    return y_pred2, y_pred_prob, best_model_path


# def four_fold_validate(model_type, whistle_image_pie1_4d,
#                        whistle_image_pie2_4d,
#                        whistle_image_pie3_4d,
#                        whistle_image_pie4_4d, label_pie1, label_pie2,
#                        label_pie3, label_pie4, conf):
#     start_time = timeit.default_timer()
#
#     # fold 1: pie 2, 3, 4 as training and pie 1 as testing
#     whistle_image_train_fold1_4d = np.vstack(
#         (whistle_image_pie2_4d, whistle_image_pie3_4d, whistle_image_pie4_4d))
#     label_train_fold1 = label_pie2 + label_pie3 + label_pie4
#     y_pred1, y_pred_prob1, best_model1 = one_fold_validate(model_type,
#                                              whistle_image_train_fold1_4d,
#                                              label_train_fold1,
#                                              whistle_image_pie1_4d, label_pie1,
#                                              conf, fold_id=1)
#     # fold 2: pie 1, 3, 4 as training and pie 2 as testing
#     whistle_image_train_fold2_4d = np.vstack(
#         (whistle_image_pie1_4d, whistle_image_pie3_4d, whistle_image_pie4_4d))
#     label_train_fold2 = label_pie1 + label_pie3 + label_pie4
#     y_pred2, y_pred_prob2, best_model2 = one_fold_validate(model_type,
#                                              whistle_image_train_fold2_4d,
#                                              label_train_fold2,
#                                              whistle_image_pie2_4d, label_pie2,
#                                              conf, fold_id=2)
#     # fold 3: pie 1, 2, 4 as training and pie 3 as testing
#     whistle_image_train_fold3_4d = np.vstack(
#         (whistle_image_pie1_4d, whistle_image_pie2_4d, whistle_image_pie4_4d))
#     label_train_fold3 = label_pie1 + label_pie2 + label_pie4
#     y_pred3, y_pred_prob3, best_model3 = one_fold_validate(model_type,
#                                              whistle_image_train_fold3_4d,
#                                              label_train_fold3,
#                                              whistle_image_pie3_4d, label_pie3,
#                                              conf, fold_id=3)
#     # fold 4: pie 1, 2, 3 as training and pie 4 as testing
#     whistle_image_train_fold4_4d = np.vstack(
#         (whistle_image_pie1_4d, whistle_image_pie2_4d, whistle_image_pie3_4d))
#     label_train_fold4 = label_pie1 + label_pie2 + label_pie3
#     y_pred4, y_pred_prob4, best_model4 = one_fold_validate(model_type,
#                                              whistle_image_train_fold4_4d,
#                                              label_train_fold4,
#                                              whistle_image_pie4_4d, label_pie4,
#                                              conf, fold_id=4)
#
#     # collect all
#     y_pred_tot = np.concatenate((y_pred1, y_pred2, y_pred3, y_pred4))
#     label_total = label_pie1 + label_pie2 + label_pie3 + label_pie4
#     metrics_two_fold(label_total, y_pred_tot, conf['log_dir'],
#                      'accuracy_total.txt', conf, mode='total')
#
#     stop_time = timeit.default_timer()
#     with open(os.path.join(conf['log_dir'], 'run_time.txt'), 'w') as f:
#         run_time = stop_time-start_time
#         with redirect_stdout(f):
#             print("Run time is: {0:.3f} s.".format(run_time))
#             print("Run time is: {0:.3f} m.".format(run_time/60.0))
#
#     return y_pred1, y_pred2, y_pred3, y_pred4, y_pred_prob1, y_pred_prob2, \
#            y_pred_prob3, y_pred_prob4
#
#
# def four_fold_validate_generator(model_type, whistle_image_pie_4d_list,
#                                  label_pie_list, conf):
#     start_time = timeit.default_timer()
#
#     # fold 1: pie 2, 3, 4 as training and pie 1 as testing
#     whistle_image_train_fold1_4d = np.vstack(
#         (whistle_image_pie_4d_list[1], whistle_image_pie_4d_list[2], whistle_image_pie_4d_list[3]))
#     label_train_fold1 = label_pie_list[1] + label_pie_list[2] + label_pie_list[3]
#     y_pred1, y_pred_prob1, best_model1 = one_fold_validate_generator(model_type,
#                                              whistle_image_train_fold1_4d,
#                                              label_train_fold1,
#                                              whistle_image_pie_4d_list[0], label_pie_list[0],
#                                              conf, fold_id=1)
#     # fold 2: pie 1, 3, 4 as training and pie 2 as testing
#     whistle_image_train_fold2_4d = np.vstack(
#         (whistle_image_pie_4d_list[0], whistle_image_pie_4d_list[2], whistle_image_pie_4d_list[3]))
#     label_train_fold2 = label_pie_list[0] + label_pie_list[2] + label_pie_list[3]
#     y_pred2, y_pred_prob2, best_model2 = one_fold_validate_generator(model_type,
#                                              whistle_image_train_fold2_4d,
#                                              label_train_fold2,
#                                              whistle_image_pie_4d_list[1], label_pie_list[1],
#                                              conf, fold_id=2)
#     # fold 3: pie 1, 2, 4 as training and pie 3 as testing
#     whistle_image_train_fold3_4d = np.vstack(
#         (whistle_image_pie_4d_list[0], whistle_image_pie_4d_list[1], whistle_image_pie_4d_list[3]))
#     label_train_fold3 = label_pie_list[0] + label_pie_list[1] + label_pie_list[3]
#     y_pred3, y_pred_prob3, best_model3 = one_fold_validate_generator(model_type,
#                                              whistle_image_train_fold3_4d,
#                                              label_train_fold3,
#                                              whistle_image_pie_4d_list[2], label_pie_list[2],
#                                              conf, fold_id=3)
#     # fold 4: pie 1, 2, 3 as training and pie 4 as testing
#     whistle_image_train_fold4_4d = np.vstack(
#         (whistle_image_pie_4d_list[0], whistle_image_pie_4d_list[1], whistle_image_pie_4d_list[2]))
#     label_train_fold4 = label_pie_list[0] + label_pie_list[1] + label_pie_list[2]
#     y_pred4, y_pred_prob4, best_model4 = one_fold_validate_generator(model_type,
#                                              whistle_image_train_fold4_4d,
#                                              label_train_fold4,
#                                              whistle_image_pie_4d_list[3], label_pie_list[3],
#                                              conf, fold_id=4)
#
#     # collect all
#     y_pred_tot = np.concatenate((y_pred1, y_pred2, y_pred3, y_pred4))
#     label_total = label_pie_list[0] + label_pie_list[1] + label_pie_list[2] + label_pie_list[3]
#     metrics_two_fold(label_total, y_pred_tot, conf['log_dir'],
#                      'accuracy_total.txt', conf, mode='total')
#
#     stop_time = timeit.default_timer()
#     with open(os.path.join(conf['log_dir'], 'run_time.txt'), 'w') as f:
#         run_time = stop_time-start_time
#         with redirect_stdout(f):
#             print("Run time is: {0:.3f} s.".format(run_time))
#             print("Run time is: {0:.3f} m.".format(run_time/60.0))
#
#     return y_pred1, y_pred2, y_pred3, y_pred4, y_pred_prob1, y_pred_prob2, \
#            y_pred_prob3, y_pred_prob4


def all_data_train_validate(model_name, whistle_image_target_4d, label_target,
                            conf):
    label_target_cat = to_categorical(label_target)

    model_name_func = globals()[model_name]
    model = model_name_func(conf)
    # model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}.hdf5'
    # model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}_valacc_{val_acc:.4f}.hdf5'
    model_name_format = 'epoch_{epoch:02d}_valloss_{val_loss:.4f}_valacc_{val_accuracy:.4f}.hdf5'

    # if conf['comment'] is None:
    #     conf['comment'] = ''
    log_dir = make_folder_time_now(folder_out=conf['log_dir'], folder_comment=conf['comment'])

    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    check_path = os.path.join(log_dir, model_name_format)

    with open(os.path.join(log_dir, 'architecture.txt'), 'w') as f:
        with redirect_stdout(f):
            # print('')
            for kk in sorted(list(conf.keys())):
                print(kk + ' ==>> ' + str(conf[kk]))
            model.summary()

    # checkpoint
    checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=1,
                                 save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=conf['patience'])
    # model compile
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(lr=conf['learning_rate']),
                  # optimizer=Adam(lr=conf['learning_rate']),
                  metrics=['accuracy'])
    model.summary()

    count_species = label_target_cat.sum(axis=0)+1e-6
    weight_curr = (count_species.max() / count_species).tolist()
    conf["class_weight"] = {0: weight_curr[0], 1: weight_curr[1]}

    callback_list = [checkpoint, TensorBoard(log_dir=log_dir), early_stop]

    model.fit(whistle_image_target_4d, label_target_cat,
              batch_size=conf['batch_size'], epochs=conf['epoch'],
              verbose=1, validation_split=0.2,
              callbacks=callback_list, class_weight=conf["class_weight"])
    # re_model_name_format = 'epoch_\d+_valloss_(\d+.\d{4}).hdf5'
    re_model_name_format = 'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
    best_model_path, _ = find_best_model(log_dir, re_model_name_format,
                                         is_max=False, purge=True)
    conf['best_model'] = best_model_path
    model = load_model(best_model_path)
    y_pred_prob = model.predict(whistle_image_target_4d)
    y_pred2 = np.argmax(y_pred_prob, axis=1)
    metrics_two_fold(label_target, y_pred2, log_dir, 'accuracy_fold.txt',
                     conf, mode='fold')

    np.savetxt(os.path.join(log_dir, 'pred_label.txt'), y_pred2, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(log_dir, 'pred_prob.txt'), y_pred_prob, delimiter=',', fmt='%.6f')

    del model
    gc.collect()
    backend.clear_session()

    return best_model_path


def make_folder_time_now(folder_out='./', folder_comment='model_unknown'):
    current = datetime.datetime.now()
    right_now = current.strftime("%Y-%m-%d_%H%M%S")
    print(right_now)

    folder_out_now = os.path.join(folder_out, right_now+'_'+folder_comment)
    if not os.path.exists(folder_out_now):
        os.makedirs(folder_out_now)

    return folder_out_now


# def find_best_model(classifier_path, fmt, is_max=False, purge=True):
#     """
#     Return the path to the model with the best accuracy, given the path to
#     all the trained classifiers
#     Args:
#         classifier_path: path to all the trained classifiers
#         fmt: e.g. "epoch_\d+_[0-1].\d+_(\d+.\d{4}).hdf5"
#         'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
#         is_max: use max; otherwise, min
#         purge: True to purge models files except the best one
#     Return:
#         the path of the model with the best accuracy
#     """
#     # list all files ending with .hdf5
#     day_list = sorted(glob.glob(os.path.join(classifier_path + '/', '*.hdf5')))
#
#     # re the last 4 digits for accuracy
#     hdf5_filename = []
#     hdf5_accu = np.zeros(len(day_list))
#     for dd in range(len(day_list)):
#         filename = os.path.basename(day_list[dd])
#         hdf5_filename.append(filename)
#         # m = re.search("_F1_(0.\d{4}).hdf5", filename)
#         # m = re.search("_([0-1].\d{4}).hdf5", filename)
#         # m = re.search("epoch_\d+_[0-1].\d+_(\d+.\d{4}).hdf5", filename)
#         m = re.search(fmt, filename)
#         try:
#             hdf5_accu[dd] = float(m.groups()[0])
#         except:
#             continue
#
#     # select the laregest one and write to the variable classifier_file
#     if len(hdf5_accu) == 0:
#         best_model_path = ''
#         best_accu = 0
#     else:
#         if is_max is True:
#             ind_max = np.argmax(hdf5_accu)
#         else: # use min instead
#             ind_max = np.argmin(hdf5_accu)
#         best_model_path = day_list[int(ind_max)]
#         best_accu = hdf5_accu[ind_max]
#         # purge all model files except the best_model
#         if purge:
#             for ff in day_list:
#                 if ff != best_model_path:
#                     os.remove(ff)
#     print('Best model:'+str(best_accu))
#     print(best_model_path)
#     return best_model_path, best_accu


# def find_best_model(classifier_path, fmt='epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5', is_max=False, purge=True):
#     """
#     Return the path to the model with the best accuracy, given the path to
#     all the trained classifiers
#     Args:
#         classifier_path: path to all the trained classifiers
#         fmt: e.g. "epoch_\d+_[0-1].\d+_(\d+.\d{4}).hdf5"
#         'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
#         is_max: use max; otherwise, min
#         purge: True to purge models files except the best one
#     Return:
#         the path of the model with the best accuracy
#     """
#     # list all files ending with .hdf5
#     day_list = sorted(glob.glob(os.path.join(classifier_path + '/', '*.hdf5')))
#
#     # re the last 4 digits for accuracy
#     hdf5_filename = []
#     hdf5_accu = np.zeros(len(day_list))
#     for dd in range(len(day_list)):
#         filename = os.path.basename(day_list[dd])
#         hdf5_filename.append(filename)
#         # m = re.search("_F1_(0.\d{4}).hdf5", filename)
#         # m = re.search("_([0-1].\d{4}).hdf5", filename)
#         # m = re.search("epoch_\d+_[0-1].\d+_(\d+.\d{4}).hdf5", filename)
#         m = re.search(fmt, filename)
#         try:
#             hdf5_accu[dd] = float(m.groups()[0])
#         except:
#             continue
#
#     # select the laregest one and write to the variable classifier_file
#     if len(hdf5_accu) == 0:
#         best_model_path = ''
#         best_accu = 0
#     else:
#         if is_max is True:
#             ind_max = np.argmax(hdf5_accu)
#         else: # use min instead
#             ind_max = np.argmin(hdf5_accu)
#         best_model_path = day_list[int(ind_max)]
#         best_accu = hdf5_accu[ind_max]
#         # purge all model files except the best_model
#         if purge:
#             for ff in day_list:
#                 if ff != best_model_path:
#                     os.remove(ff)
#     print('Best model:'+str(best_accu))
#     print(best_model_path)
#     return best_model_path, best_accu


def find_best_model(classifier_path, fmt='epoch_\d+_valloss_(\d+.\d{4})_valacc_(\d+.\d{4}).hdf5', is_max=True, purge=True):
    """
    Return the path to the model with the best accuracy, given the path to
    all the trained classifiers
    Args:
        classifier_path: path to all the trained classifiers
        fmt: e.g. "epoch_\d+_[0-1].\d+_(\d+.\d{4}).hdf5"
        'epoch_\d+_valloss_(\d+.\d{4})_valacc_\d+.\d{4}.hdf5'
        is_max: use max; otherwise, min
        purge: True to purge models files except the best one
    Return:
        the path of the model with the best accuracy
    """
    # list all files ending with .hdf5
    day_list = sorted(glob.glob(os.path.join(classifier_path + '/', '*.hdf5')))

    # re the last 4 digits for accuracy
    hdf5_filename = []
    hdf5_accu = np.zeros(len(day_list))
    for dd in range(len(day_list)):
        filename = os.path.basename(day_list[dd])
        hdf5_filename.append(filename)
        # m = re.search("_F1_(0.\d{4}).hdf5", filename)
        # m = re.search("_([0-1].\d{4}).hdf5", filename)
        # m = re.search("epoch_\d+_[0-1].\d+_(\d+.\d{4}).hdf5", filename)
        m = re.search(fmt, filename)
        try:
            #  hdf5_accu[dd] = float(m.groups()[0])
            hdf5_accu[dd] = float(m.groups()[1])
        except:
            continue

    # select the laregest one and write to the variable classifier_file
    if len(hdf5_accu) == 0:
        best_model_path = ''
        best_accu = 0
    else:
        if is_max is True:
            ind_max = np.argmax(hdf5_accu)
        else: # use min instead
            ind_max = np.argmin(hdf5_accu)
        best_model_path = day_list[int(ind_max)]
        best_accu = hdf5_accu[ind_max]
        # purge all model files except the best_model
        if purge:
            for ff in day_list:
                if ff != best_model_path:
                    os.remove(ff)
    print('Best model:'+str(best_accu))
    print(best_model_path)
    return best_model_path, best_accu


def metrics_two_fold(y_test, y_pred, log_dir, filename, conf, mode=None):
    """
    :param y_test:
    :param y_pred:
    :param log_dir:
    :param filename:
    :param model:
    :param conf:
    :param mode: "total" or "fold"
    :return:
    """
    # confusion matrix
    confu_mat = confusion_matrix(y_test, y_pred)
    # balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    # classification report
    class_report = classification_report(y_test, y_pred, digits=3)
    # f1-score
    f1_class_avg = f1_score(y_test, y_pred, average='macro')

    with open(os.path.join(log_dir, filename), 'w') as f:
        with redirect_stdout(f):
            print(conf['species_name'])

            print("\nConfusion matrix:\n")
            print(confu_mat)
            print("\nBalanced accuracy:")
            print(balanced_accuracy)
            print("\nClassification_report:")
            print(class_report)
            print("\nf1 score:")
            print(f1_class_avg)

            if mode == "total":
                print("\nHyper-parameters:")
                # model.summary()
                print("\nBatch size: " + str(conf['batch_size']))
                print("Epoch: " + str(conf['epoch']))
                print("\nLearning rate: " + str(conf['learning_rate']))
                # print("L2_regularization: " + str(conf['l2_regu']))
                # print("Dropout: " + str(conf['dropout']))
                # print("Recurrent dropout: " + str(conf['recurrent_dropout']))
                # print("Duration threshold: " + str(conf['duration_thre']))
                # print("Time resolution: " + str(conf['time_reso']))
                # print("Bidirectional mode: " + str(conf['bi_mod']))
                # print('Window look back: '+str(conf['win_back']))
                # print("Train data shuffle: " + str(conf['shuffle']))
            elif mode == "fold":
                print("Class weight: " + str(conf['class_weight']))
                print("Best model: " + str(conf['best_model']))



# def main_old():
#     # bin files for training & testing
#     bin_dir = '/home/ys587/__Data/__whistle/tonals_20190210/label_bin_files/cv4'
#     # bin_dir_train = os.path.join(bin_dir, 'cv2/__first_pie')
#     # bin_dir_test = os.path.join(bin_dir, 'cv2/__second_pie')
#     bin_dir_fold = dict()
#     for pp in range(4):  # 'bin_dir_fold['pie1']'
#         bin_dir_fold['pie'+str(pp+1)] = os.path.join(bin_dir, 'pie'+str(pp+1))
#
#     sound_dir = '/home/ys587/__Data/__whistle/__sound_species/'
#     species_name = ['bottlenose', 'common', 'spinner', 'melon-headed']
#     species_id = {'bottlenose': 0, 'common': 1, 'spinner': 2, 'melon-headed': 3}
#
#     conf_gen = dict()
#     conf_gen['log_dir'] = "/home/ys587/__Data/__whistle/__log_dir_context/contour_temp_audio"
#     conf_gen['save_dir'] = "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store_temp"
#     conf_gen['data_store'] = "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store/__four_class"
#     conf_gen['bin_dir'] = bin_dir
#
#     conf_gen['species_name'] = species_name
#     conf_gen['species_id'] = species_id
#     # conf_gen['time_reso'] = 0.01  # 10 ms
#     # conf_gen['time_reso'] = 0.05  # 50 ms
#     conf_gen['time_reso'] = 0.02  # 20 ms
#
#     # cepstral coefficient
#     conf_gen['sample_rate'] = 192000
#     conf_gen["num_class"] = len(species_name)
#
#     conf_gen['context_winsize'] = 1.0  # sec
#     conf_gen['context_hopsize'] = 0.5  # sec
#     conf_gen['contour_timethre'] = 10  # 0.2 s
#
#     conf_gen['fft_size'] = 4096
#     conf_gen['hop_length'] = int(conf_gen['time_reso']*conf_gen['sample_rate'])
#     # audio
#     conf_gen['freq_ind_low'] = 0
#     # conf_gen['freq_ind_low'] = 20
#     conf_gen['freq_ind_high'] = 144
#
#     conf_gen['img_t'] = int(floor((conf_gen['context_winsize'] / conf_gen['time_reso'])))
#     # conf_gen['img_f'] = conf_gen['freq_ind_high'] - conf_gen['freq_ind_low']
#     conf_gen['img_f'] = 64
#     conf_gen['input_shape'] = (conf_gen['img_f'], conf_gen['img_t'], 1)
#
#     conf_gen['l2_regu'] = 0.01
#     # conf_gen['l2_regu'] = 0.001
#     # conf_gen['l2_regu'] = 0.2
#     conf_gen['dropout'] = 0.1
#     # conf_gen['batch_size'] = 128  # lstm_2lay
#     # conf_gen['batch_size'] = 32  # resnet 18, 34
#     conf_gen['batch_size'] = 64
#     conf_gen['epoch'] = 200
#     # conf_gen['epoch'] = 1  # debug
#     conf_gen['learning_rate'] = 1.0
#     pie_num = 4
#     conf_gen['pie_num'] = pie_num
#
#     conf_gen['confusion_callback'] = False
#     conf_gen['spectro_dilation'] = False
#
#     # conf_gen['numpy_data_output'] = False  # !!
#     # conf_gen['numpy_data_use'] = not conf_gen['numpy_data_output']
#     conf_gen['numpy_data_use'] = True
#     conf_gen['img_data_output'] = False  # output image of spectrogram data
#
#     # add one more class 'noise': 4 species class + 1 noise class
#     conf_gen['class_noise'] = True  # add the fifth class: noise
#
#     # Use masked spectrogram of whistle contours, instead of original spectrograms
#     # conf_gen['mask_spec_contour'] = True
#     conf_gen['mask_spec_contour'] = False
#
#     for pp in range(pie_num):  # 'pie1_data.npz'
#         conf_gen['save_file_pie'+str(pp+1)] = os.path.join(conf_gen['save_dir'],
#                                                    'pie'+str(pp+1)+'_data.npz')
#     for pp in range(pie_num):
#         conf_gen['image_pie' + str(pp + 1)] = os.path.join(conf_gen['save_dir'],
#                                                                'pie' + str(pp + 1))
#
#     if not os.path.exists(conf_gen['log_dir']):
#         os.mkdir(conf_gen['log_dir'])
#
#     # RNN
#     # conf_gen['network_type'] = 'conv2d_lstm'  # rnn, cnn or conv2d_lstm
#     # conf_gen['network_type'] = 'rnn'
#     conf_gen['network_type'] = 'cnn'
#     # conf_gen['recurrent_dropout'] = 0.1
#     conf_gen['recurrent_dropout'] = 0.01
#     conf_gen['dense_size'] = 128
#
#     # Read species labels, filenames & extract time and frequency sequences
#     contour_pie_list = []
#     bin_wav_pair_pie_list = []
#     for pp in range(pie_num):
#         contour_pie_curr, bin_wav_pair_pie_curr = bin_extract(
#             bin_dir_fold['pie'+str(pp+1)], sound_dir, species_name)
#         contour_pie_list.append(contour_pie_curr)
#         bin_wav_pair_pie_list.append(bin_wav_pair_pie_curr)
#
#     # read contours from bin files
#     contour_pie_list_alllist =  []
#     for pp in range(pie_num):
#         contour_pie_list_curr = contour_target_retrieve(contour_pie_list[pp],
#                                                     bin_dir_fold['pie'+str(pp+1)], conf_gen['time_reso'])
#         contour_pie_list_alllist.append(contour_pie_list_curr)
#
#     # prepare training & testing data
#     if conf_gen['class_noise']:
#         conf_gen['species_name'].append('noise')
#         conf_gen['species_id'].update({'noise': 4})
#         conf_gen["num_class"] += 1
#         if not conf_gen['mask_spec_contour']:
#             # expt
#             # conf_gen['data_store'] = "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store/__five_class"
#             # 1-s win
#             # conf_gen['data_store'] = "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store/__1s_win/__five_class"
#             # 1-s win, 0.5-s overlap
#             # conf_gen['data_store'] = "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store/__1s_win/__five_class_win_1s_overlap_p5s_contour_p2s_timereso_p02"
#             conf_gen[
#                 'data_store'] = "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store/__pcen_nopulse/__1s_win_p2s_hop_p4s_contour_48k_samplerate"
#             # attention
#             # conf_gen['data_store'] = "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store/__five_class_attention"
#         else:  # Masked-spectrogram-contour
#             conf_gen['data_store'] = \
#                 "/home/ys587/__Data/__whistle/__log_dir_context/audio_data_store/__five_class_mask_spec_pasterization"
#
#     # Pie 1 - 4
#     whistle_image_pie_list = []
#     label_pie_list = []
#     # freq_high_pie_list = []
#     # freq_low_pie_list = []
#     for pp in range(pie_num):
#         pie_curr_data_path = os.path.join(conf_gen['data_store'], 'pie'+str(pp+1)+'_data.npz')
#         if os.path.exists(pie_curr_data_path) & conf_gen['numpy_data_use']:
#             print('Loading pie data...')
#             data_temp = np.load(pie_curr_data_path)
#             whistle_image_pie_curr = data_temp['whistle_image']
#             label_pie_curr = data_temp['label'].tolist()
#         elif conf_gen['mask_spec_contour']:
#             whistle_image_pie_curr, label_pie_curr, _, _ = prepare_data_mask(
#                 contour_pie_list_alllist[pp], sound_dir, conf_gen, conf_gen['image_pie'+str(pp+1)],
#                 conf_gen['save_file_pie'+str(pp+1)], plot=conf_gen['img_data_output'])
#         else:
#             whistle_image_pie_curr, label_pie_curr, _, _ = prepare_data_audio(
#                 contour_pie_list_alllist[pp], sound_dir, conf_gen, conf_gen['image_pie'+str(pp+1)],
#                 conf_gen['save_file_pie'+str(pp+1)], plot=conf_gen['img_data_output'])
#         whistle_image_pie_list.append(whistle_image_pie_curr)
#         label_pie_list.append(label_pie_curr)
#         # freq_high_pie_list.append(freq_high_pie_curr)
#         # freq_low_pie_list.append(freq_low_pie_curr)
#
#     # Change the dimensions
#     if conf_gen['network_type'] == 'cnn':
#         whistle_image_pie_4d_list = []
#         for pp in range(pie_num):
#             whistle_image_pie_curr_4d = np.expand_dims(whistle_image_pie_list[pp], axis=3)
#             whistle_image_pie_4d_list.append(whistle_image_pie_curr_4d)
#     elif conf_gen['network_type'] == 'rnn':
#         whistle_image_pie_4d_list = []
#         for pp in range(pie_num):
#             whistle_image_pie_curr_4d = np.transpose(whistle_image_pie_list[pp], (0, 2, 1))
#             whistle_image_pie_4d_list.append(whistle_image_pie_curr_4d)
#     elif conf_gen['network_type'] == 'conv2d_lstm':
#         whistle_image_pie_4d_list = []
#         for pp in range(pie_num):
#             whistle_image_pie_curr_0 = np.expand_dims(whistle_image_pie_list[pp],
#                                                   axis=(3, 4))
#             whistle_image_pie_curr_4d = np.transpose(whistle_image_pie_curr_0,
#                                                  (0, 2, 1, 3, 4))
#             whistle_image_pie_4d_list.append(whistle_image_pie_curr_4d)
#
#     log_dir = '/home/ys587/__Data/__whistle/__log_dir_context/__new_results'
#
#     learning_rate_list = []
#     le_regu_list = []
#     for ii_lr in [1.0, 0.1]:
#         for ii_l2 in [0.1, 0.01]:
#             learning_rate_list.append(ii_lr)
#             le_regu_list.append(ii_l2)
#
#     for rr in range(len(le_regu_list)):
#         # for rr in range(5):
#         conf_gen['learning_rate'] = learning_rate_list[rr]
#         conf_gen['l2_regu'] = le_regu_list[rr]
#
#         model_type = 'resnet34_expt'
#         proj_name = model_type + '_run' + str(rr) + '_f1'
#         print(proj_name)
#
#         conf_gen['log_dir'] = os.path.join(log_dir, proj_name)
#         if not os.path.exists(conf_gen['log_dir']):
#             os.mkdir(conf_gen['log_dir'])
#         conf_gen['num_filters'] = 16
#         label_pred1, label_pred2, label_pred3, label_pred4, pred_prob1, \
#         pred_prob2, pred_prob3, pred_prob4 = four_fold_validate_generator(
#             model_type, whistle_image_pie_4d_list, label_pie_list, conf_gen)
#
#
