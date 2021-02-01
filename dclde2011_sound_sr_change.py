#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 2/1/21
@author: atoultaro
"""
import os
import glob
import librosa
import soundfile as sf
import sox
import pandas as pd
import sys

# proj = '96kHz'
# proj_path = os.path.dirname(__file__)
proj_path_orig = '/home/ys587/__Data/__whistle/__whistle_dclde2011'
sample_rate_target = 48000

# proj_path_orig = os.path.join(proj_path, proj)
proj_path_new = proj_path_orig+'_48kHz'
if not os.path.exists(proj_path_new):
    os.makedirs(proj_path_new)

# collect sound info
species_list = os.listdir(proj_path_orig)

# conventional approach: read by the target sampling rate in librosa and save the first channel
species_folder = os.listdir(proj_path_orig)
species_folder.sort()
for ss in species_folder:
    print(ss)
    file_list = glob.glob(os.path.join(proj_path_orig, ss, '*.wav'))
    file_list.sort()

    output_folder = os.path.join(proj_path_new, ss)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for ff in file_list:
        print(ff)
        samples, _ = librosa.load(ff, sr=sample_rate_target)
        if samples.ndim == 2:
            samples = samples[0]
        ff2 = os.path.join(proj_path_new, ss, os.path.basename(ff))
        sf.write(ff2, samples, sample_rate_target, subtype='PCM_16')

# transformer approach: abandoned because it cannot offer selecting a specific channel of a multiple-channel sounds instead of merging them.
if False:
    # change sampling rate to the target sampling rate
    tfm = sox.Transformer()
    tfm.set_output_format(rate=sample_rate_target, channels=1, bits=16)

    species_id_list = []
    file_list = []
    sr_list = []
    duration_list = []
    chan_list = []
    # for ss in species_list[:2]:
    for ss in species_list:
        print(ss)

        species_dir = os.path.join(proj_path_new, ss)
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)

        wav_list = glob.glob(os.path.join(proj_path_orig, ss)+'/*.wav')
        print(len(wav_list))
        for ww in wav_list:
            info = sox.file_info.info(ww)
            # print('')

            species_id_list.append(ss)
            file_list.append(os.path.basename(ww))
            sr_list.append(int(info['sample_rate']))
            duration_list.append(float(info['duration']))
            chan_list.append(int(info['channels']))

            tfm.build_file(ww, os.path.join(species_dir, os.path.basename(ww)))

    df_sound_info = pd.DataFrame({'file': file_list,
                                  'species': species_id_list,
                                  'sample_rate': sr_list,
                                  'duration': duration_list,
                                  'chan_num': chan_list,
                                  })
    df_sound_info.to_csv(os.path.join(proj_path, proj+'.csv'), index=False)
