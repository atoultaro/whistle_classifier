#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 2/4/21
@author: atoultaro
"""
import os
import random
from math import ceil, log
import multiprocessing as mp
from itertools import repeat

import numpy as np
import pandas as pd
import librosa

import lib_feature

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# # from tensorflow.contrib.image import sparse_image_warp
# from tensorflow_addons.image import sparse_image_warp
# from tensorflow.python.framework import constant_op

from PIL import Image


def wavname_to_meta(this_basename, dataset_name):
    """
    Translate the filename into meta data such as species, dataset & id
    Args:
        this_basename:
        dataset_name:

    Returns:

    """
    split_filename = (os.path.splitext(this_basename)[0]).split('_')
    species_this = split_filename[0]
    clipid_this = int(split_filename[-1])

    # add the column 'deployment'
    if dataset_name == 'oswald':
        deploy_this = split_filename[1]
    elif dataset_name == 'gillispie':
        # deploy_this = split_filename[1]
        if split_filename[0] != 'NO':
            deploy_this = split_filename[1]
        else:
            deploy_this = split_filename[2]
    elif dataset_name == 'dclde2011':
        deploy_this = 'dclde2011'
    elif dataset_name == 'watkin':
        if split_filename[0] != 'NO':
            deploy_this = split_filename[1]+split_filename[2]
        else:
            deploy_this = split_filename[2] + split_filename[3]
    else:
        deploy_this = 'NA'

    return species_this, clipid_this, deploy_this


def wavname_to_meta_oswald(this_basename, dataset_name):
    """
    Translate the filename into meta data such as species, dataset & id
    this oswald only version adds acoustic encounters.
    Args:
        this_basename:
        dataset_name:

    Returns:

    """
    split_filename = (os.path.splitext(this_basename)[0]).split('_')
    species_this = split_filename[0]
    deploy_this = split_filename[1]
    encounter_this = split_filename[2]
    filename_this = this_basename

    return species_this, deploy_this, encounter_this, filename_this


def sample_length_normal(samples, clip_length=96000):
    """
    Normalize the sound samples' length
    Args:
        samples:
        clip_length:

    Returns:

    """
    if samples.shape[0] <= clip_length:
        samples_temp = np.zeros(clip_length)
        samples_temp[0:samples.shape[0]] = samples
        return samples_temp
    elif samples.shape[0] > clip_length:
        return samples[0:clip_length]
    else:
        return samples


def load_and_normalize(sound_path, sr=48000, clip_length=96000):
    """
    Load sound samples from the path, normalize the length & remove DC values
    Returns:

    """
    samples, _ = librosa.load(sound_path, sr=sr)
    samples = sample_length_normal(samples, clip_length)
    samples = samples - samples.mean()

    return samples


def dataset_fea_augment_parallel(df_curr_species, df_curr_noise, dataset_name, dataset_path, fs=48000, copies_of_aug=3,
                                 clip_length=96000, hop_length=960, shift_time_max=25, shift_freq_max=5,
                                 added_noise=False, noise_type='cross'):
    """
    Run data augmentation / feature extraction on one dataset
    Args:
        df_curr_species:
        df_curr_noise:
        dataset_name:
        dataset_path:
        fs:
        clip_length:
        hop_length:
        shift_time_max:
        shift_freq_max:

    Returns:

    """
    cpu_count = os.cpu_count() - 1
    pool_fea = mp.Pool(processes=cpu_count)
    spec_feas_orig_list = []
    labels_orig_list = []
    spec_feas_aug_list = []
    labels_aug_list = []

    # debug_n = 100  # DEBUG

    row_list = []
    for index, row in df_curr_species.iterrows():
        # for index, row in df_curr_species.sample(n=debug_n).iterrows():
        row_list.append(row)

    row_noise_list = []
    if (df_curr_noise is None) | (df_curr_noise.shape[0] == 0):
        for ii in range(df_curr_species.shape[0]):
            # for ii in range(debug_n):
            row_noise_list.append(pd.Series())
    elif noise_type == 'cross':
        for index, row_noise in df_curr_noise.sample(n=df_curr_species.shape[0], replace=True).iterrows():
            # for index, row_noise in df_curr_noise.sample(n=debug_n, replace=True).iterrows():
            row_noise_list.append(row_noise)
    elif noise_type == 'single':
        for index, row_species in df_curr_species.iterrows():
            row_noise = df_curr_noise[df_curr_noise['deployment'] == row_species['deployment']].sample(n=1, replace=True).iloc[0]
            row_noise_list.append(row_noise)

        # for index, row_noise in df_curr_noise.sample(n=df_curr_species.shape[0], replace=True).iterrows():
        #     # for index, row_noise in df_curr_noise.sample(n=debug_n, replace=True).iterrows():
        #     row_noise_list.append(row_noise)

    # copies_of_aug: a dictionary from species to class weight   <<== for what?
    # species_counts = df_curr_species['species'].value_counts()
    # weight_dict = dict()
    # for ss, count in species_counts.iteritems():
    #     weight_dict[ss] = count
    # count_max = max(list(weight_dict.values()))
    # for kk in weight_dict.keys():
    #     # weight_dict[kk] = int(ceil(count_max / weight_dict[kk]))
    #     weight_dict[kk] = int(ceil(log(count_max / weight_dict[kk])))*2 + copies_of_aug
    #
    # copies_of_aug_list = []
    # for index, row in df_curr_species.iterrows():
    #     # for index, row in df_curr_species.sample(n=debug_n).iterrows():
    #     copies_of_aug_list.append(weight_dict[row['species']])

    for spec_feas_orig_each, labels_orig_each, spec_feas_aug_each, labels_aug_each in pool_fea.starmap(
            fea_augment_parallel, zip(row_list, row_noise_list, repeat(dataset_path), repeat(fs),
                                      repeat(copies_of_aug), repeat(clip_length), repeat(hop_length),
                                      repeat(shift_time_max), repeat(shift_freq_max), repeat(added_noise))
    ):
        spec_feas_orig_list.append(spec_feas_orig_each)
        labels_orig_list.append(labels_orig_each)
        spec_feas_aug_list.append(spec_feas_aug_each)
        labels_aug_list.append(labels_aug_each)

    # for spec_feas_orig_each, labels_orig_each in pool_fea.starmap(
    #         fea_augment_parallel, zip(row_list, row_noise_list, repeat(dataset_path), repeat(fs),
    #                                   repeat(clip_length), repeat(hop_length), repeat(shift_time_max),
    #                                   repeat(shift_freq_max))
    # ):
    #     spec_feas_orig_list.append(spec_feas_orig_each)
    #     labels_orig_list.append(labels_orig_each)

    pool_fea.close()
    pool_fea.terminate()
    pool_fea.join()

    feas_orig = np.concatenate(spec_feas_orig_list)
    labels_orig = np.concatenate(labels_orig_list)
    feas_aug = np.concatenate(spec_feas_aug_list)
    labels_aug = np.concatenate(labels_aug_list)

    # combine features & labels
    # feas_orig = np.stack(spec_feas_orig)
    np.savez(os.path.join(dataset_path, dataset_name+'_orig'), feas_orig=feas_orig, labels_orig=labels_orig)
    # feas_aug = np.stack(spec_feas_aug)
    np.savez(os.path.join(dataset_path, dataset_name+'_aug'), feas_aug=feas_aug, labels_aug=labels_aug)

    return feas_orig, labels_orig, feas_aug, labels_aug


def fea_augment_parallel(row, row_noise, dataset_path, fs=48000, copies_of_aug=3, clip_length=96000, hop_length=960,
                         shift_time_max=25, shift_freq_max=5, added_noise=False):
    spec_feas_orig = []
    labels_orig = []
    # filenames_orig = []

    spec_feas_aug = []
    labels_aug = []
    # filenames_aug = []

    # original sound
    curr_clip_path = os.path.join(dataset_path, '__' + row['dataset'], '__sound_clips', row['filename'])
    samples = load_and_normalize(curr_clip_path, sr=fs, clip_length=clip_length)
    spectro = librosa.feature.melspectrogram(samples, sr=fs, hop_length=hop_length, power=1)

    spec_feas_orig.append(lib_feature.feature_whistleness(spectro, unit_vec=False))
    # spec_feas_orig.append(lib_feature.feature_whistleness(spectro, unit_vec=True))
    labels_orig.append(row['species'])

    # augmented sound
    aug_count = 0
    while aug_count < copies_of_aug:
        # (1) adding noises from another noise clips0
        # if row_noise.shape[0] != 0:
        if added_noise:
            curr_noise_path = os.path.join(dataset_path, '__' + row['dataset'], '__sound_clips', row_noise['filename'])
            spectro_aug = add_noise_to_signal(curr_noise_path, samples, fs=fs, clip_length=clip_length,
                                              hop_length=hop_length)
        else:
            spectro_aug = librosa.feature.melspectrogram(samples, sr=fs, hop_length=hop_length, power=1)
        # (2) time & freq shifting
        spectro_aug = time_freq_shifting(spectro_aug, shift_time_max, shift_freq_max)
        # (3) warping & masking through SpecAugment
        spectro_aug = spec_augment(spectro_aug, time_warping_para=40, frequency_masking_para=5, time_masking_para=40,
                                   num_mask=1)
        spectro_aug = spectro_aug*(spectro_aug >= 0)
        spec_feas_aug.append(lib_feature.feature_whistleness(spectro_aug, unit_vec=False))
        # spec_feas_aug.append(lib_feature.feature_whistleness(spectro_aug, unit_vec=True))
        labels_aug.append(row['species'])
        del spectro_aug

        aug_count += 1

    # # (1) time & freq shifting
    # spectro_shift = time_freq_shifting(spectro, shift_time_max, shift_freq_max)
    #
    # spec_feas_aug.append(lib_feature.feature_whistleness(spectro_shift))
    # labels_aug.append(row['species'])
    # del spectro_shift
    #
    # # (2) warping & masking through SpecAugment
    # spectro_warp = spec_augment(spectro, time_warping_para=80, frequency_masking_para=5, time_masking_para=20,
    #                             num_mask=1)
    # spec_feas_aug.append(lib_feature.feature_whistleness(spectro_warp))
    # labels_aug.append(row['species'])
    # del spectro_warp
    #
    # # (3) adding noises from another noise clips
    # if row_noise.shape[0] != 0:
    #     curr_noise_path = os.path.join(dataset_path, '__' + row['dataset'], '__sound_clips', row_noise['filename'])
    #     spectro_noisy = add_noise_to_signal(curr_noise_path, samples, fs=fs, clip_length=clip_length, hop_length=hop_length)
    #
    #     spec_feas_aug.append(lib_feature.feature_whistleness(spectro_noisy))
    #     labels_aug.append(row['species'])
    #     del spectro_noisy

    return spec_feas_orig, labels_orig, spec_feas_aug, labels_aug


def time_freq_shifting(spectro, shift_time_max, shift_freq_max):
    """
    augmentation: shift the spectrogrom up & down, left & right
    Args:
        spectro:
        shift_time_max:
        shift_freq_max:

    Returns:

    """
    shift_time = random.randrange(-shift_time_max, shift_time_max+1)
    shift_freq = random.randrange(-shift_freq_max, shift_freq_max+1)
    spectro_shift = np.zeros(spectro.shape)
    # noise_low, noise_high = np.percentile(spectro, [10, 50])
    noise_floor = np.percentile(spectro, [30], axis=1).reshape(-1)
    f_max = spectro_shift.shape[0]
    t_max = spectro_shift.shape[1]
    for i in range(f_max):
        for j in range(t_max):
            if (i - shift_freq >= 0) & (i - shift_freq < f_max) & (j - shift_time >= 0) & (j - shift_time < t_max):
                spectro_shift[i, j] = spectro[i - shift_freq, j - shift_time]
            else:
                if i - shift_freq < 0:
                    i_new = 0
                elif i - shift_freq >= f_max:
                    i_new = f_max - 1
                else:
                    i_new = i - shift_freq
                noise_ij = random.gauss(noise_floor[i_new], noise_floor[i_new] * 0.2)
                spectro_shift[i, j] = noise_ij * (noise_ij >= 0.0)

    return spectro_shift


def add_noise_to_signal(noise_path, samples, fs=48000, clip_length=96000, hop_length=960):  # row_noise = df_curr_noise.sample(n=1).iloc[0]
    noise = load_and_normalize(noise_path, sr=fs, clip_length=clip_length)

    alpha = random.uniform(0.7, 0.95)
    samples_noisy = alpha*samples + (1-alpha)*noise
    spectro_noisy = librosa.feature.melspectrogram(samples_noisy, sr=fs, hop_length=hop_length, power=1)

    return spectro_noisy


def spec_augment(mel_spectrogram, time_warping_para=40, frequency_masking_para=20, time_masking_para=20,
                 num_mask=1):
    # Step 1 : Time warping
    tau = mel_spectrogram.shape[1]

    # Image warping control point setting
    # control_point_locations = np.asarray([[64, 64], [64, 80]])
    # control_point_locations = constant_op.constant(
    #     np.float32(np.expand_dims(control_point_locations, 0)))
    #
    # control_point_displacements = np.ones(
    #     control_point_locations.shape.as_list())
    # control_point_displacements = constant_op.constant(
    #     np.float32(control_point_displacements))

    # Input: mel_spectrogram; output: warped_mel_spectrogram
    # mel spectrogram data type convert to tensor constant for sparse_image_warp
    # mel_spectrogram = mel_spectrogram.reshape([1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1])
    # mel_spectrogram_op = constant_op.constant(np.float32(mel_spectrogram))
    # w = random.randint(0, time_warping_para)
    #
    # warped_mel_spectrogram_op, _ = sparse_image_warp(mel_spectrogram_op,
    #                                                  source_control_point_locations=control_point_locations,
    #                                                  dest_control_point_locations=control_point_locations + control_point_displacements,
    #                                                  interpolation_order=2,
    #                                                  regularization_weight=0,
    #                                                  num_boundary_points=0
    #                                                  )
    #
    # # Change data type of warp result to numpy array for masking step
    # with tf.Session() as sess:
    #     warped_mel_spectrogram = sess.run(warped_mel_spectrogram_op)
    #
    # warped_mel_spectrogram = warped_mel_spectrogram.reshape([warped_mel_spectrogram.shape[1],
    #                                                          warped_mel_spectrogram.shape[2]])

    warped_mel_spectrogram = time_warp(mel_spectrogram, max_time_warp=time_warping_para, inplace=False)

    # loop Masking line number
    for i in range(num_mask):
        # Step 2 : Frequency masking
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        v = 128  # Now hard coding but I will improve soon.
        f0 = random.randint(0, v - f)
        warped_mel_spectrogram[f0:f0 + f, :] = 0

        # Step 3 : Time masking
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        warped_mel_spectrogram[:, t0:t0 + t] = 0

    return warped_mel_spectrogram


def time_warp(x, max_time_warp=40, inplace=False):
    """time warp for spec augment

    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp"
        (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    """
    window = max_time_warp

    t = x.shape[0]
    if t - window <= window:
        return x
    # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
    center = random.randrange(window, t - window)
    warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

    left = Image.fromarray(x[:center]).resize((x.shape[1], warped), Image.BICUBIC)
    right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), Image.BICUBIC)
    if inplace:
        x[:warped] = left
        x[warped:] = right
        return x

    return np.concatenate((left, right), 0)
