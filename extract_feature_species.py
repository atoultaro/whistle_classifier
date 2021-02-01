#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 1/20/21
@author: atoultaro
"""
import os
import numpy as np
from whistle_classifier import lib_feature

data_source = ['oswald', 'gillispie', 'dclde2011', 'watkin']
# data_source = ['dclde2011']
# data_source = ['gillispie']
# data_source = ['watkin']
model_whistleness = '/home/ys587/__Data/__whistle/__whislte_30_species/__fit_result_whistleness/__fea_mel_pcen_p4s_unit_contour_no_pulses/2021-01-31_204503_resnet18_expt_alldata_run1_f1_lr_0.00333/epoch_49_valloss_0.2673_valacc_0.9425.hdf5'  # use_pcen=True, remove_pulse=True
species_dict = {'NO': 0, 'BD': 1, 'MH': 2, 'CD': 3, 'STR': 4, 'SPT': 5, 'SPIN': 6, 'PLT': 7, 'RD': 8, 'RT': 9,
                'WSD': 10, 'FKW': 11, 'BEL': 12, 'KW': 13, 'WBD': 14, 'DUSK': 15, 'FRA': 16, 'PKW': 17, 'LPLT': 18,
                'NAR': 19, 'CLY': 20, 'SPE': 21, 'ASP': 22}

feature_list = []
for dd in data_source:
    if dd == 'oswald':
        # setting
        work_path = '/home/ys587/__Data/__whistle/__whislte_30_species/__oswald'
        whistle_data_oswald = '/home/ys587/__Data/__whistle/__whistle_oswald'  # sound
        fea_out = os.path.join(work_path, '__fea_files')
        if not os.path.exists(fea_out):
            os.makedirs(fea_out)
        seltab_out = os.path.join(work_path, '__sound_seltab')
        if not os.path.exists(seltab_out):
            os.makedirs(seltab_out)
        deployment = ['HICEAS2002', 'PICEAS2005', 'STAR2000', 'STAR2003', 'STAR2006']
        species_to_code = {'bottlenose': 'BD', 'longbeaked_common': 'CD', 'shortbeaked_common': 'CD', 'common': 'CD',
                           'striped': 'STR', 'spotted': 'SPT', 'spinner': 'SPIN', 'pilot': 'PLT', 'roughtoothed': 'RT',
                           'false_killer': 'FKW'}

        # extract features and save them into folders
        path_sound_info = os.path.join(work_path, '__sound_info')
        if not os.path.exists(path_sound_info):
            os.makedirs(path_sound_info)
        csv_oswald_sound = os.path.join(work_path, '__sound_info', 'oswald_encounter.csv')
        csv_oswald_info = os.path.join(work_path, '__sound_info', 'oswald_soundinfo.csv')
        df_sound_oswald, df_info_oswald = lib_feature.df_sound_info_oswald(csv_oswald_sound, csv_oswald_info,
                                                                            species_to_code, whistle_data_oswald,
                                                                            deployment)
        lib_feature.extract_fea_oswald(df_sound_oswald, model_whistleness, fea_out, seltab_out, use_pcen=True, remove_pulse=True)
        # combine features and save them into numpy arrays
        # print('Reading features...')
        species_fea = lib_feature.read_features_from_files(fea_out, list(species_dict.keys()))
        if len(species_fea) == 0:
            raise ('No feature individual files are present. Need to compute features.')
        fea_oswald_4d, label_oswald = lib_feature.combine_features_from_dict(species_fea, work_path, 'fea_label_oswald.npz', species_dict)
        # fea_oswald_out = os.path.join(work_path, 'fea_label_oswald.npz')
        # np.savez(fea_oswald_out, fea_part_4d=fea_oswald_4d, label_part=label_oswald)

    elif dd == 'gillispie':
        # setting
        work_path = '/home/ys587/__Data/__whistle/__whislte_30_species/__gillispie'
        whistle_data_gillispie = '/home/ys587/__Data/__whistle/__whistle_gillispie'  # sound
        fea_out = os.path.join(work_path, '__fea_files')
        if not os.path.exists(fea_out):
            os.makedirs(fea_out)
        seltab_out = os.path.join(work_path, '__sound_seltab')
        if not os.path.exists(seltab_out):
            os.makedirs(seltab_out)

        # extract features and save them into folders
        path_sound_info = os.path.join(work_path, '__sound_info')
        if not os.path.exists(path_sound_info):
            os.makedirs(path_sound_info)
        csv_gillispie_sound = os.path.join(work_path, '__sound_info', 'gillispie_encounter.csv')
        csv_gillispie_info = os.path.join(work_path, '__sound_info', 'gillispie_soundinfo.csv')
        df_sound_gillispie, df_info_gillispie = lib_feature.df_sound_info_gillispie(csv_gillispie_sound, csv_gillispie_info,
                                                                           whistle_data_gillispie)

        lib_feature.extract_fea_gillispie(df_sound_gillispie, model_whistleness, fea_out, seltab_out, use_pcen=True, remove_pulse=True)

        species_fea = lib_feature.read_features_from_files(fea_out, list(species_dict.keys()))
        if len(species_fea) == 0:
            raise ('No feature individual files are present. Need to compute features.')
        fea_gillispie_4d, label_gillispie = lib_feature.combine_features_from_dict(species_fea, work_path, 'fea_label_gillispie.npz', species_dict)
        # fea_gillispie_out = os.path.join(work_path, 'fea_label_gillispie.npz')
        # np.savez(fea_gillispie_out, fea_part_4d=fea_gillispie_4d, label_part=label_gillispie)

    elif dd == 'dclde2011':
        work_path = '/home/ys587/__Data/__whistle/__whislte_30_species/__dclde2011'
        whistle_data_dclde2011 = '/home/ys587/__Data/__whistle/__whistle_dclde2011'  # sound
        fea_out = os.path.join(work_path, '__fea_files')
        if not os.path.exists(fea_out):
            os.makedirs(fea_out)
        seltab_out = os.path.join(work_path, '__sound_seltab')
        if not os.path.exists(seltab_out):
            os.makedirs(seltab_out)

        # extract features and save them into folders
        path_sound_info = os.path.join(work_path, '__sound_info')
        if not os.path.exists(path_sound_info):
            os.makedirs(path_sound_info)
        csv_dclde2011_sound = os.path.join(work_path, '__sound_info', 'dclde2011_encounter.csv')
        csv_dclde2011_info = os.path.join(work_path, '__sound_info', 'dclde2011_soundinfo.csv')
        df_sound_dclde2011, df_info_dclde2011 = lib_feature.df_sound_info_dclde2011(csv_dclde2011_sound, csv_dclde2011_info,
                                                                           whistle_data_dclde2011)

        lib_feature.extract_fea_dclde2011(df_sound_dclde2011, model_whistleness, fea_out, seltab_out, use_pcen=True, remove_pulse=True)
        species_fea = lib_feature.read_features_from_files(fea_out, list(species_dict.keys()))
        if len(species_fea) == 0:
            raise ('No feature individual files are present. Need to compute features.')
        fea_dclde2011_4d, label_dclde2011 = lib_feature.combine_features_from_dict(species_fea, work_path, 'fea_label_dclde2011.npz', species_dict)

    elif dd == 'watkin':
        # setting
        work_path = '/home/ys587/__Data/__whistle/__whislte_30_species/__watkin'
        whistle_data_watkin = '/home/ys587/__Data/__whistle/__watkin_sounds'  # sound
        fea_out = os.path.join(work_path, '__fea_files')
        if not os.path.exists(fea_out):
            os.makedirs(fea_out)
        seltab_out = os.path.join(work_path, '__sound_seltab')
        if not os.path.exists(seltab_out):
            os.makedirs(seltab_out)

        # extract features and save them into folders
        path_sound_info = os.path.join(work_path, '__sound_info')
        if not os.path.exists(path_sound_info):
            os.makedirs(path_sound_info)
        csv_watkin_sound = os.path.join(work_path, '__sound_info', 'watkin_encounter.csv')
        csv_watkin_info = os.path.join(work_path, '__sound_info', 'watkin_soundinfo.csv')
        df_sound_watkin, df_info_watkin = lib_feature.df_sound_info_watkin(csv_watkin_sound, csv_watkin_info,
                                                                           whistle_data_watkin)

        lib_feature.extract_fea_watkin(df_sound_watkin, model_whistleness, fea_out, seltab_out, use_pcen=True, remove_pulse=True)
        species_fea = lib_feature.read_features_from_files(fea_out, list(species_dict.keys()))
        if len(species_fea) == 0:
            raise ('No feature individual files are present. Need to compute features.')
        fea_watkin_4d, label_watkin = lib_feature.combine_features_from_dict(species_fea, work_path, 'fea_label_watkin.npz', species_dict)

