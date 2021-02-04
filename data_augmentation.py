#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data augmentation

Created on 2/3/21
@author: atoultaro
"""
import os
import glob
# import random

# import numpy as np
import pandas as pd
# import librosa
from specAugment.spec_augment_tensorflow import *

import lib_feature


def wavname_to_meta(this_basename, dataset_name):
    split_filename = (os.path.splitext(this_basename)[0]).split('_')
    species_this = split_filename[0]
    clipid_this = int(split_filename[-1])

    # add the column 'deployment'
    if dataset_name == 'oswald':
        deploy_this =  split_filename[1]
    elif dataset_name == 'gillispie':
        deploy_this = split_filename[1]
    elif dataset_name == 'dclde2011':
        deploy_this = 'dclde2011'
    elif dataset_name == 'watkin':
        deploy_this = split_filename[1]+split_filename[2]
    else:
        deploy_this = 'NA'

    return species_this, clipid_this, deploy_this


def sample_length_normal(samples):
    if samples.shape[0] <= clip_length:
        samples_temp = np.zeros(clip_length)
        samples_temp[0:samples.shape[0]] = samples
        return samples_temp
    elif samples.shape[0] > clip_length:
        return samples[0:clip_length]
    else:
        return samples


# priori knowledge
species_dict = {'NO': 0, 'BD': 1, 'MH': 2, 'CD': 3, 'STR': 4, 'SPT': 5, 'SPIN': 6, 'PLT': 7, 'RD': 8, 'RT': 9,
                'WSD': 10, 'FKW': 11, 'BEL': 12, 'KW': 13, 'WBD': 14, 'DUSK': 15, 'FRA': 16, 'PKW': 17, 'LPLT': 18,
                'NAR': 19, 'CLY': 20, 'SPE': 21, 'ASP': 22}
fs = 48000
time_reso = 0.02
hop_length = int(time_reso*fs)

clip_length = 2*fs  # 96,000 samples
freq_low = 50  # mel-scale =~ 2 kHz
shift_time_max = int(0.5/time_reso)
shift_freq_max = 10

random.seed(0)

# where the sound clips are
dataset_path = '/home/ys587/__Data/__whistle/__whislte_30_species/__dataset'
datasets = ['oswald', 'gillispie', 'dclde2011', 'watkin']
clip_paths = dict()
for dd in datasets:
    clip_paths.update({dd: os.path.join(dataset_path, '__'+dd, '__sound_clips')})

# statistics of sound clips
# build labels, dataframe from the filenames
dataset_df = dict()
for dd in datasets:
    # for dd in [datasets[0]]:
    print('..'+dd)
    # search by the species names in the filenames
    wav_list = glob.glob(clip_paths[dd]+'/*.wav')
    wav_list.sort()
    wav_basename = [os.path.basename(ww) for ww in wav_list]

    species_list = []
    clip_id = []
    deployment_list = []
    for ww in range(len(wav_basename)):
        species_file, clipid_file, deploy_file = wavname_to_meta(wav_basename[ww], dd)
        species_list.append(species_file)
        clip_id.append(clipid_file)
        deployment_list.append(deploy_file)
        # split_filename = (os.path.splitext(wav_basename[ww])[0]).split('_')
        # species_list.append(split_filename[0])
        # clip_id.append(int(split_filename[-1]))
        #
        # # add the column 'deployment'
        # if dd == 'oswald':
        #     deployment_list.append(split_filename[1])
        # elif dd == 'gillispie':
        #     deployment_list.append(split_filename[1])
        # elif dd == 'dclde2011':
        #     deployment_list.append('dclde2011')
        # elif dd == 'watkin':
        #     deployment_list.append(split_filename[1]+split_filename[2])
        # else:
        #     continue

    # build dataframe: dataset, base filename, species, clip id
    dataset_df.update({dd: pd.DataFrame(list(zip([dd]*len(wav_basename), wav_basename, species_list, clip_id, deployment_list)),
                                        columns=['dataset', 'filename', 'species', 'id', 'deployment'])})
    dataset_df[dd].to_csv(os.path.join(dataset_path, dd+'.csv'), index=False)

# all four datasets
df_total = pd.concat(list(dataset_df.values()), axis=0)
df_total.to_csv(os.path.join(dataset_path, 'four_datasets.csv'), index=False)

# stats: how many clips for species, for noise, for each dataset & total?
# Under construction

# train: three datasets except gillispie / test dataset
df_total_except_oswald = pd.concat([dataset_df['gillispie'], dataset_df['dclde2011'], dataset_df['watkin']], axis=0)
df_total_except_oswald.to_csv(os.path.join(dataset_path, 'four_except_oswald.csv'), index=False)
df_total_except_gillispie = pd.concat([dataset_df['oswald'], dataset_df['dclde2011'], dataset_df['watkin']], axis=0)
df_total_except_gillispie.to_csv(os.path.join(dataset_path, 'four_except_gillispie.csv'), index=False)
df_total_except_dclde2011 = pd.concat([dataset_df['oswald'], dataset_df['gillispie'], dataset_df['watkin']], axis=0)
df_total_except_dclde2011.to_csv(os.path.join(dataset_path, 'four_except_dclde2011.csv'), index=False)
df_total_except_watkin = pd.concat([dataset_df['oswald'], dataset_df['dclde2011'], dataset_df['gillispie']], axis=0)
df_total_except_watkin.to_csv(os.path.join(dataset_path, 'four_except_watkin.csv'), index=False)

# data augmentation: time & freq shift, warping, add noise, cutoff
for dd in datasets:
    # for dd in [datasets[0]]:
    df_curr = dataset_df[dd]
    # split into NO vs all species
    df_curr_noise = df_curr[df_curr['species'] == 'NO']
    df_curr_species = df_curr[df_curr['species'] != 'NO']

    spec_feas_orig = []
    labels_orig = []
    filenames_orig = []
    spec_feas_aug = []
    labels_aug = []
    filenames_aug = []

    # augmentation on species data & extract features
    for index, row in df_curr_species.iterrows():
        # for index, row in df_curr_species.sample(n=5).iterrows():
        print(row['filename'])
        curr_species = species_dict[row['species']]

        curr_clip_path = os.path.join(dataset_path, '__'+dd, '__sound_clips', row['filename'])
        samples, _ = librosa.load(curr_clip_path, sr=fs)
        samples = sample_length_normal(samples)
        samples = samples - samples.mean()
        spectro = librosa.feature.melspectrogram(samples, sr=fs, hop_length=hop_length, power=1)

        # original features
        spec_feas_orig.append(lib_feature.feature_whistleness(spectro))
        labels_orig.append(curr_species)
        filenames_orig.append(row['filename'])

        # time & freq shifting
        shift_time = random.randrange(-shift_time_max, shift_time_max+1)
        shift_freq = random.randrange(-shift_freq_max, shift_freq_max+1)
        spectro_shift = np.zeros(spectro.shape)
        noise_low, noise_high = np.percentile(spectro, [0, 20])
        f_max = spectro_shift.shape[0]
        t_max = spectro_shift.shape[1]
        for i in range(f_max):
            for j in range(t_max):
                if (i-shift_freq >= 0) & (i-shift_freq < f_max) & (j-shift_time >= 0) & (j-shift_time < t_max):
                    spectro_shift[i, j] = spectro[i-shift_freq, j-shift_time]
                else:
                    spectro_shift[i, j] = random.uniform(noise_low, noise_high)
        spec_feas_aug.append(lib_feature.feature_whistleness(spectro_shift))
        labels_aug.append(curr_species)
        filenames_aug.append(row['filename'])
        # para_aug_shift_time.append(shift_time)  # keep track parameters for augmentation
        # para_aug_shift_freq.append(shift_freq)

        # warping & masking through SpecAugment
        spectro_warp = spec_augment(spectro, time_warping_para=40, frequency_masking_para=5, time_masking_para=20)
        spec_feas_aug.append(lib_feature.feature_whistleness(spectro_warp))
        labels_aug.append(curr_species)
        filenames_aug.append(row['filename'])

        # adding noises from another noise clips
        row_noise = df_curr_noise.sample(n=1).iloc[0]
        curr_noise_path = os.path.join(dataset_path, '__'+dd, '__sound_clips', row_noise['filename'])
        noise, _ = librosa.load(curr_noise_path, sr=fs)
        noise = sample_length_normal(noise)
        noise = noise - noise.mean()
        alpha = random.uniform(0.7, 0.95)
        samples_noisy = alpha*samples + (1-alpha)*noise

        spectro_noisy = librosa.feature.melspectrogram(samples_noisy, sr=fs, hop_length=hop_length, power=1)
        spec_feas_aug.append(lib_feature.feature_whistleness(spectro_noisy))
        labels_aug.append(curr_species)
        filenames_aug.append(row['filename'])

        # change contrast: tone down the high magnitudes whereas amplify the low magnitudes
        # Later

    # combine features & labels
    spec_orig_dataset = np.stack(spec_feas_orig)
    np.savez(os.path.join(dataset_path, dd+'_orig'), spec_orig_dataset=spec_orig_dataset, labels_orig=labels_orig)
    spec_aug_dataset = np.stack(spec_feas_aug)
    np.savez(os.path.join(dataset_path, dd+'_aug'), spec_aug_dataset=spec_aug_dataset, labels_auc=labels_aug)

    # generate dataframe for augmented data & save to csv: need augmentation parameters
    # for ff in filenames_aug:
    #     wavname_to_meta(ff, dd)




