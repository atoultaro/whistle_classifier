#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 2/4/21
@author: atoultaro
"""
import os
import numpy as np
import random

import librosa


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
        deploy_this = split_filename[1]
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


