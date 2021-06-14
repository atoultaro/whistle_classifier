#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data augmentation

Created on 2/3/21
@author: atoultaro
"""
# import os
import glob
# import random

# import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import repeat
# import librosa
# from specAugment.spec_augment_tensorflow import *

# import lib_feature
from lib_augment import *


# priori knowledge
# species_dict = {'BD': 0, 'MH': 1, 'CD': 2, 'STR': 3, 'SPT': 4, 'SPIN': 5, 'PLT': 6, 'RD': 7, 'RT': 8,
#                 'WSD': 9, 'FKW': 10, 'BEL': 11, 'KW': 12, 'WBD': 13, 'DUSK': 14, 'FRA': 15, 'PKW': 16, 'LPLT': 17, }
# remove 'SPE', 'CLY', 'ASP'
species_dict = {'BD': 1, 'MH': 2, 'CD': 3, 'STR': 4, 'SPT': 5, 'SPIN': 6, 'PLT': 7, 'RD': 8, 'RT': 9,
                'WSD': 10, 'FKW': 11, 'BEL': 12, 'KW': 13, 'WBD': 14, 'DUSK': 15, 'FRA': 16, 'PKW': 17, 'LPLT': 18,
                'NAR': 19, 'CLY': 20, 'SPE': 21, 'ASP': 22}
# species_all = list(species_dict.keys())

fs = 48000
time_reso = 0.02
hop_length = int(time_reso*fs)  #

copies_of_aug = 10
clip_length = 2*fs  # 96,000 samples
freq_low = 50  # mel-scale =~ 2 kHz
shift_time_max = int(0.5/time_reso)  # 25
shift_freq_max = 5

random.seed(0)

# where the sound clips are
dataset_path = '/home/ys587/__Data/__whistle/__whistle_30_species/__dataset'
datasets = ['oswald', 'gillispie', 'dclde2011', 'watkin']
clip_paths = dict()
for dd in datasets:
    clip_paths.update({dd: os.path.join(dataset_path, '__'+dd, '__sound_clips')})

# statistics of sound clips
print('Build labels & dataframe from the filenames')
dataset_df = dict()
for dd in [datasets[0]]:
    # for dd in [datasets[0]]:
    print('=='+dd)
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
# df_total_except_oswald = pd.concat([dataset_df['gillispie'], dataset_df['dclde2011'], dataset_df['watkin']], axis=0)
# df_total_except_oswald.to_csv(os.path.join(dataset_path, 'four_except_oswald.csv'), index=False)
# df_total_except_gillispie = pd.concat([dataset_df['oswald'], dataset_df['dclde2011'], dataset_df['watkin']], axis=0)
# df_total_except_gillispie.to_csv(os.path.join(dataset_path, 'four_except_gillispie.csv'), index=False)
# df_total_except_dclde2011 = pd.concat([dataset_df['oswald'], dataset_df['gillispie'], dataset_df['watkin']], axis=0)
# df_total_except_dclde2011.to_csv(os.path.join(dataset_path, 'four_except_dclde2011.csv'), index=False)
# df_total_except_watkin = pd.concat([dataset_df['oswald'], dataset_df['dclde2011'], dataset_df['gillispie']], axis=0)
# df_total_except_watkin.to_csv(os.path.join(dataset_path, 'four_except_watkin.csv'), index=False)

# data augmentation: time & freq shift, warping, add noise, cutoff
print('Data augmentation: time/freq shift, warping, & adding noise.')
for dd in datasets:
# for dd in [datasets[0]]:
    print('=='+dd)
    df_curr = dataset_df[dd]
    # split into NO vs all species
    df_curr_species = df_curr[df_curr['species'] != 'NO']
    print('====Sound clips: '+str(df_curr_species.shape[0]))
    df_curr_noise = df_curr[df_curr['species'] == 'NO']
    print('====Noise clips: ' + str(df_curr_noise.shape[0]))

    if dd == 'oswald':
        # split the data into two parts:
        # df_curr_species_1 = df_curr_species[(df_curr_species['deployment'] == 'PICEAS2005') |
        #                                     (df_curr_species['deployment'] == 'STAR2000')]
        # df_curr_noise_1 = df_curr_noise[(df_curr_noise['deployment'] == 'PICEAS2005') |
        #                                 (df_curr_noise['deployment'] == 'STAR2000')]
        # dataset_fea_augment_parallel(df_curr_species_1, df_curr_noise_1, dd+'_part1', dataset_path, fs=fs, clip_length=clip_length,
        #                              hop_length=hop_length, shift_time_max=shift_time_max,
        #                              shift_freq_max=shift_freq_max)
        #
        # df_curr_species_2 = df_curr_species[(df_curr_species['deployment'] == 'HICEAS2002') |
        #                                     (df_curr_species['deployment'] == 'STAR2003') |
        #                                     (df_curr_species['deployment'] == 'STAR2006')]
        # df_curr_noise_2 = df_curr_noise[(df_curr_noise['deployment'] == 'PICEAS2002') |
        #                                     (df_curr_noise['deployment'] == 'STAR2003') |
        #                                     (df_curr_noise['deployment'] == 'STAR2006')]
        # dataset_fea_augment_parallel(df_curr_species_2, df_curr_noise_2, dd+'_part2', dataset_path, fs=fs, clip_length=clip_length,
        #                              hop_length=hop_length, shift_time_max=shift_time_max,
        #                              shift_freq_max=shift_freq_max)
        # # save dataframes into csv files
        # df_curr_species_1.to_csv(os.path.join(dataset_path, dd+'_part1.csv'), index=False)
        # df_curr_species_2.to_csv(os.path.join(dataset_path, dd+'_part2.csv'), index=False)
        # del df_curr_species_1, df_curr_noise_1
        # del df_curr_species_2, df_curr_noise_2
        for ee in ['STAR2000', 'STAR2003', 'STAR2006', 'HICEAS2002', 'PICEAS2005']:
            df_curr_species_1 = df_curr_species[(df_curr_species['deployment'] == ee)]
            # df_curr_noise_1 = df_curr_noise[(df_curr_noise['deployment'] == ee)]
            dataset_fea_augment_parallel(df_curr_species_1, df_curr_noise, dd+'_'+ee, dataset_path, fs=fs,
                                         copies_of_aug=copies_of_aug, clip_length=clip_length,
                                         hop_length=hop_length, shift_time_max=shift_time_max,
                                         shift_freq_max=shift_freq_max)
            df_curr_species_1.to_csv(os.path.join(dataset_path, dd + '_'+ee+'.csv'), index=False)
            del df_curr_species_1

    elif dd == 'gillispie':
        # split the data into two parts:
        df_curr_species_1 = df_curr_species[(df_curr_species['deployment'] == '48kHz')]
        # df_curr_noise_1 = df_curr_noise[(df_curr_noise['deployment'] == '48kHz')]
        dataset_fea_augment_parallel(df_curr_species_1, df_curr_noise, dd + '_48kHz', dataset_path, fs=fs,
                                     clip_length=clip_length, hop_length=hop_length, shift_time_max=shift_time_max,
                                     shift_freq_max=shift_freq_max)
        df_curr_species_2 = df_curr_species[(df_curr_species['deployment'] == '96kHz')]
        # df_curr_noise_2 = df_curr_noise[(df_curr_noise['deployment'] == '96kHz')]
        dataset_fea_augment_parallel(df_curr_species_2, df_curr_noise, dd + '_96kHz', dataset_path, fs=fs,
                                     clip_length=clip_length, hop_length=hop_length, shift_time_max=shift_time_max,
                                     shift_freq_max=shift_freq_max)
        # save dataframes into csv files
        df_curr_species_1.to_csv(os.path.join(dataset_path, dd+'_48kHz.csv'), index=False)
        df_curr_species_2.to_csv(os.path.join(dataset_path, dd+'_96kHz.csv'), index=False)
        del df_curr_species_1
        del df_curr_species_2
    else:
        dataset_fea_augment_parallel(df_curr_species, df_curr_noise, dd, dataset_path, fs=fs, clip_length=clip_length,
                                 hop_length=hop_length, shift_time_max=shift_time_max, shift_freq_max=shift_freq_max)

    # pool_fea = mp.Pool(processes=cpu_count)
    # spec_feas_orig_list = []
    # labels_orig_list = []
    # spec_feas_aug_list = []
    # labels_aug_list = []
    #
    # debug_n = 1000
    #
    # row_list = []
    # row_noise_list = []
    # # for index, row in df_curr_species.iterrows():
    # for index, row in df_curr_species.sample(n=debug_n).iterrows():
    #     row_list.append(row)
    #
    # for index, row_noise in df_curr_noise.sample(n=debug_n, replace=True).iterrows():
    #     # for index, row_noise in df_curr_noise.sample(n=df_curr_species.shape[0], replace=True).iterrows():
    #     row_noise_list.append(row_noise)
    #
    # for spec_feas_orig_each, labels_orig_each, spec_feas_aug_each, labels_aug_each in pool_fea.starmap(
    #         fea_augment_parallel, zip(row_list, row_noise_list, repeat(dataset_path), repeat(fs),
    #                                   repeat(clip_length), repeat(hop_length), repeat(shift_time_max),
    #                                   repeat(shift_freq_max))
    # ):
    #     spec_feas_orig_list.append(spec_feas_orig_each)
    #     labels_orig_list.append(labels_orig_each)
    #     spec_feas_aug_list.append(spec_feas_aug_each)
    #     labels_aug_list.append(labels_aug_each)
    #
    # # for spec_feas_orig_each, labels_orig_each in pool_fea.starmap(
    # #         fea_augment_parallel, zip(row_list, row_noise_list, repeat(dataset_path), repeat(fs),
    # #                                   repeat(clip_length), repeat(hop_length), repeat(shift_time_max),
    # #                                   repeat(shift_freq_max))
    # # ):
    # #     spec_feas_orig_list.append(spec_feas_orig_each)
    # #     labels_orig_list.append(labels_orig_each)
    #
    # pool_fea.close()
    # pool_fea.terminate()
    # pool_fea.join()
    #
    # feas_orig = np.concatenate(spec_feas_orig_list)
    # labels_orig = np.concatenate(labels_orig_list)
    # feas_aug = np.concatenate(spec_feas_aug_list)
    # labels_aug = np.concatenate(labels_aug_list)
    #
    # # combine features & labels
    # # feas_orig = np.stack(spec_feas_orig)
    # np.savez(os.path.join(dataset_path, dd+'_orig'), feas_orig=feas_orig, labels_orig=labels_orig)
    # # feas_aug = np.stack(spec_feas_aug)
    # np.savez(os.path.join(dataset_path, dd+'_aug'), feas_aug=feas_aug, labels_aug=labels_aug)






    # filenames_orig = []
    # filenames_aug = []

    # augmentation on species data & extract features
    # for index, row in df_curr_species.iterrows():
    # for index, row in df_curr_species.sample(n=1000).iterrows():
    #     print(row['filename'])
    #     # curr_species = species_dict[row['species']]
    #     curr_species = row['species']

        # # original sound
        # curr_clip_path = os.path.join(dataset_path, '__'+dd, '__sound_clips', row['filename'])
        # samples = load_and_normalize(curr_clip_path, sr=fs, clip_length=clip_length)
        # spectro = librosa.feature.melspectrogram(samples, sr=fs, hop_length=hop_length, power=1)
        #
        # spec_feas_orig.append(lib_feature.feature_whistleness(spectro))
        # labels_orig.append(curr_species)
        # filenames_orig.append(row['filename'])
        #
        # # time & freq shifting
        # spectro_shift = time_freq_shifting(spectro, shift_time_max, shift_freq_max)
        #
        # spec_feas_aug.append(lib_feature.feature_whistleness(spectro_shift))
        # labels_aug.append(curr_species)
        # filenames_aug.append(row['filename'])
        # del spectro_shift
        #
        # # warping & masking through SpecAugment
        # spectro_warp = spec_augment(spectro, time_warping_para=80, frequency_masking_para=5, time_masking_para=20,
        #                             num_mask=1)
        # spec_feas_aug.append(lib_feature.feature_whistleness(spectro_warp))
        # labels_aug.append(curr_species)
        # filenames_aug.append(row['filename'])
        # del spectro_warp
        #
        # # adding noises from another noise clips
        # row_noise = df_curr_noise.sample(n=1).iloc[0]  # <<<===
        # curr_noise_path = os.path.join(dataset_path, '__'+dd, '__sound_clips', row_noise['filename'])
        # spectro_noisy = add_noise_to_signal(curr_noise_path, samples, fs=fs,
        #                                     clip_length=clip_length, hop_length=hop_length)
        #
        # spec_feas_aug.append(lib_feature.feature_whistleness(spectro_noisy))
        # labels_aug.append(curr_species)
        # filenames_aug.append(row['filename'])
        # del spectro_noisy
        # # Later: change contrast: tone down the high magnitudes whereas amplify the low magnitudes

    # combine features & labels
    # spec_orig_dataset = np.stack(spec_feas_orig)
    # np.savez(os.path.join(dataset_path, dd+'_orig'), spec_orig_dataset=spec_orig_dataset, labels_orig=labels_orig)
    # spec_aug_dataset = np.stack(spec_feas_aug)
    # np.savez(os.path.join(dataset_path, dd+'_aug'), spec_aug_dataset=spec_aug_dataset, labels_auc=labels_aug)

    # Later: generate dataframe for augmented data & save to csv: need augmentation parameters




