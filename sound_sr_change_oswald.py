#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling rate change for Oswald data
Created on 2/2/21
@author: atoultaro
"""
import os
import glob
import librosa
import soundfile as sf

# proj_path = os.path.dirname(__file__)
proj_path_root = '/home/ys587/__Data/__whistle/__whistle_oswald'
sample_rate_target = 48000
deployment = ['HICEAS2002', 'PICEAS2005', 'STAR2000', 'STAR2003', 'STAR2006']

for dd in deployment:
    proj_path_orig = os.path.join(proj_path_root, dd)
    proj_path_new = proj_path_orig+'_48kHz'
    if not os.path.exists(proj_path_new):
        os.makedirs(proj_path_new)

    # conventional approach: read by the target sampling rate in librosa and save the first channel
    encounter_folder = os.listdir(proj_path_orig)
    encounter_folder.sort()
    for ss in encounter_folder:
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
