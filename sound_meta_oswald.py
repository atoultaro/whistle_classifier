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
proj_path_root = '/home/ys587/__Data/__whistle/__whistle_30_species/__sound_48k/__whistle_oswald'
sample_rate_target = 48000
deployment = ['HICEAS2002', 'PICEAS2005', 'STAR2000', 'STAR2003', 'STAR2006']

species_dict = {'BD': 0, 'CD': 1, 'STR': 2, 'SPT': 3, 'SPIN': 4, 'PLT': 5, 'RT': 6,  'FKW': 7}
num_species = len(species_dict)
species_list = list(species_dict.keys())

species_to_code = {'bottlenose': 'BD', 'longbeaked_common': 'CD', 'shortbeaked_common': 'CD', 'common': 'CD',
                   'striped': 'STR', 'spotted': 'SPT', 'spinner': 'SPIN', 'pilot': 'PLT', 'roughtoothed': 'RT',
                   'false_killer': 'FKW'}

duration = dict()
encounter_num = dict()
for dd in deployment:
    proj_path_orig = os.path.join(proj_path_root, dd)

    # conventional approach: read by the target sampling rate in librosa and save the first channel
    encounter_folder = os.listdir(proj_path_orig)
    encounter_folder.sort()

    duration[dd] = dict()
    encounter_num[dd] = dict()

    # duration[dd]['total'] = 0
    for ee in species_list:
        duration[dd][ee] = 0.0
    for ee in species_list:
        encounter_num[dd][ee] = 0

    for ss in encounter_folder:
        print(ss)
        file_list = glob.glob(os.path.join(proj_path_orig, ss, '*.wav'))
        file_list.sort()

        species_curr = species_to_code[ss.split(' ')[0]]
        for ff in file_list:
            print(ff)
            sound_meta = sf.info(ff)
            duration[dd][species_curr] += sound_meta.duration
            # duration[dd]['total'] += sound_meta.duration
            print()
        encounter_num[dd][species_curr] += 1
        print()
    print()




print()

