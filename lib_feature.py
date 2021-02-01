#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for species recognition
Created on 6/16/20
@author: atoultaro
"""
import os
import sys
import pandas as pd
import glob
from math import floor, ceil
import numpy as np
from scipy.ndimage.filters import median_filter

import cv2
import soundfile as sf
# from keras.models import load_model
from tensorflow.keras.models import load_model
import librosa


def make_sound_sel_table(seltab_output_path, begin_time, end_time, begin_path,
                         file_offset, score_arr, score_thre, chan=None,
                         class_id=None):
    assert(begin_time.shape[0] == score_arr.shape[0])
    event_num = score_arr.shape[0]

    data_dict = {
        'Selection': [ii+1 for ii in range(event_num)],
        'View': ['Spectrogram 1']*event_num,

        'Begin Time (s)': np.around(begin_time, decimals=2),
        'End Time (s)': np.around(end_time, decimals=2),
        'Low Freq (Hz)': [3000.0]*event_num,
        'High Freq (Hz)': [22000.0]*event_num,
        'Score': np.around(score_arr, decimals=4),
        'Score Thre': np.repeat(np.around(score_thre, decimals=3), event_num),
        'Begin Path': begin_path,
        'File Offset': file_offset
    }
    if chan is None:
        data_dict.update({'Channel': [1] * event_num})
    else:
        data_dict.update({'Channel': chan})
    if class_id is not None:
        data_dict.update({'Class_id': class_id})
        df_seltab0 = pd.DataFrame.from_dict(data_dict)
        col_name = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                    'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Score',
                    'Score Thre', 'Begin Path', 'File Offset', 'Class_id']
    else:
        df_seltab0 = pd.DataFrame.from_dict(data_dict)
        col_name = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                    'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Score',
                    'Score Thre', 'Begin Path', 'File Offset']
    # sort columns by the order of Raven's selection table
    df_seltab = df_seltab0[col_name]
    # sort rows by first, Begin Path and then, Begin
    # df_seltab = df_seltab.sort_values(by=['Begin Path', 'File Offset'])
    df_seltab = df_seltab.sort_values(by=['File Offset'])
    # df_seltab.update(pd.Series([ii+1 for ii in range(event_num)], name='Selection'))


    # write out selection table
    df_seltab.to_csv(seltab_output_path, sep='\t', mode='a', index=False)


def sound_file_info(df_target):
    ''' Get sound information
    '''
    print('Retrieving sound information...')
    sound_file = []
    sound_samplerate = []
    sound_duration = []
    sound_channels = []
    sound_format = []
    sound_subtype = []
    for index, row in df_target.iterrows():
        print(row['path'])
        filelist_curr = glob.glob(row['path']+'/*.wav')
        try:
            for ff in filelist_curr:
                sound_file.append(ff)
                soundinfo = sf.info(ff)
                sound_samplerate.append(soundinfo.samplerate)
                sound_duration.append(soundinfo.duration)
                sound_channels.append(soundinfo.channels)
                sound_format.append(soundinfo.format)
                sound_subtype.append(soundinfo.subtype)
        except IOError:
            sys.exit('File '+filelist_curr[0]+' does not exist.')
        except IndexError:
            print(row['path']+' is empty.')
            continue

    df_target_info = pd.DataFrame(list(zip(sound_file, sound_samplerate,
                                   sound_duration,
                                   sound_channels,
                                   sound_format,
                                   sound_subtype)), columns=['file',
                                                            'samplerate',
                                                            'duration',
                                                            'channels',
                                                            'format',
                                                            'subtype'])

    return df_target_info


def make_sound_sel_table(seltab_output_path, begin_time, end_time, begin_path,
                         file_offset, score_arr, score_thre,
                         chan=None, class_id=None):
    assert(begin_time.shape[0] == score_arr.shape[0])
    event_num = score_arr.shape[0]

    data_dict = {
        'Selection': [ii+1 for ii in range(event_num)],
        'View': ['Spectrogram 1']*event_num,

        'Begin Time (s)': np.around(begin_time, decimals=3),
        'End Time (s)': np.around(end_time, decimals=3),
        'Low Freq (Hz)': [3000.0]*event_num,
        'High Freq (Hz)': [22000.0]*event_num,
        'Score': np.around(score_arr, decimals=4),
        'Score Thre': np.repeat(np.around(score_thre, decimals=3), event_num),
        'Begin Path': begin_path,
        'File Offset': file_offset
    }
    if chan is None:
        data_dict.update({'Channel': [1] * event_num})
    else:
        data_dict.update({'Channel': chan})
    if class_id is not None:
        data_dict.update({'Class_id': class_id})
        df_seltab0 = pd.DataFrame.from_dict(data_dict)
        col_name = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                    'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Score',
                    'Score Thre', 'Begin Path', 'File Offset', 'Class_id']
    else:
        df_seltab0 = pd.DataFrame.from_dict(data_dict)
        col_name = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                    'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Score',
                    'Score Thre', 'Begin Path', 'File Offset']
    # sort columns by the order of Raven's selection table
    df_seltab = df_seltab0[col_name]
    # sort rows by first, Begin Path and then, Begin
    # df_seltab = df_seltab.sort_values(by=['Begin Path', 'File Offset'])
    # df_seltab = df_seltab.sort_values(by=['File Offset'])
    # df_seltab.update(pd.Series([ii+1 for ii in range(event_num)], name='Selection'))

    # write out selection table
    df_seltab.to_csv(seltab_output_path, sep='\t', mode='a', index=False)


def make_sound_sel_table_dclde2020(seltab_output_path, begin_time, end_time,
                                   begin_path, file_offset, score_arr,
                                   score_thre, chan, class_id, score_max=None,
                                   score_no=None,
                                   score_bd=None,
                                   score_mh=None,
                                   score_cd=None,
                                   score_str=None,
                                   score_spt=None,
                                   score_spin=None,
                                   score_plt=None,
                                   score_rd=None,
                                   score_rt=None,
                                   score_wsd=None,
                                   score_fkw=None,
                                   score_bel=None,
                                   score_kw=None,
                                   score_wbd=None,
                                   score_dusk=None,
                                   score_fra=None,
                                   score_pkw=None,
                                   score_lplt=None,
                                   score_nar=None,
                                   score_cly=None,
                                   score_spe=None,
                                   score_asp=None
                                   ):
    assert(begin_time.shape[0] == score_arr.shape[0])
    event_num = score_arr.shape[0]

    data_dict = {
        'Selection': [ii+1 for ii in range(event_num)],
        'View': ['Spectrogram 1']*event_num,

        'Begin Time (s)': np.around(begin_time, decimals=3),
        'End Time (s)': np.around(end_time, decimals=3),
        'Low Freq (Hz)': [3000.0]*event_num,
        'High Freq (Hz)': [22000.0]*event_num,
        'Score': np.around(score_arr, decimals=4),
        'Score Thre': np.repeat(np.around(score_thre, decimals=3), event_num),
        'Begin Path': begin_path,
        'File Offset': file_offset,
        'Channel': chan,
        'Class_id': class_id,
        'Score Max': score_max,
        'NO': score_no,
        'BD': score_bd,
        'MH': score_mh,
        'CD': score_cd,
        'STR': score_str,
        'SPT': score_spt,
        'SPIN': score_spin,
        'PLT': score_plt,
        'RD': score_rd,
        'RT': score_rt,
        'WSD': score_wsd,
        'FKW': score_fkw,
        'BEL': score_bel,
        'KW': score_kw,
        'WBD': score_wbd,
        'DUSK': score_dusk,
        'FRA': score_fra,
        'PKW': score_pkw,
        'LPLT': score_lplt,
        'NAR': score_nar,
        'CLY': score_cly,
        'SPE': score_spe,
        'ASP': score_asp,

    }
    # data_dict.update({'Channel': [1] * event_num})
    # data_dict.update({'Class_id': class_id})
    df_seltab0 = pd.DataFrame.from_dict(data_dict)
    col_name = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Score',
                'Score Thre', 'Begin Path', 'File Offset', 'Class_id',
                'Score Max', 'NO', 'BD', 'MH', 'CD', 'STR', 'SPT', 'SPIN', 'PLT', 'RD', 'RT',
                'WSD', 'FKW', 'BEL', 'KW', 'WBD', 'DUSK', 'FRA', 'PKW', 'LPLT',
                'NAR', 'CLY', 'SPE', 'ASP']

    # sort columns by the order of Raven's selection table
    df_seltab = df_seltab0[col_name]
    # sort rows by first, Begin Path and then, Begin
    # df_seltab = df_seltab.sort_values(by=['Begin Path', 'File Offset'])
    # df_seltab = df_seltab.sort_values(by=['File Offset'])
    # df_seltab.update(pd.Series([ii+1 for ii in range(event_num)], name='Selection'))

    # write out selection table
    if os.path.exists(seltab_output_path):
        os.remove(seltab_output_path)
    df_seltab.to_csv(seltab_output_path, sep='\t', mode='a', index=False,
                     float_format='%.6f')


def make_sound_sel_table_empty_dclde2020(seltab_output_path):
    col_name = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Score',
                'Score Thre', 'Begin Path', 'File Offset', 'Class_id', 'NO', 'BD', 'MH', 'CD', 'STR', 'SPT', 'SPIN', 'PLT', 'RD', 'RT',
                'WSD', 'FKW', 'BEL', 'KW', 'WBD', 'DUSK', 'FRA', 'PKW', 'LPLT',
                'NAR', 'CLY', 'SPE', 'ASP']
    df_seltab = pd.DataFrame(columns=col_name)
    df_seltab.to_csv(seltab_output_path, mode='a', index=False)


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


def df_sound_info_oswald(csv_oswald_sound, csv_oswald_info, species_to_code, whistle_data=None,
                         deployment=None):
    if whistle_data is None:
        whistle_data = '/home/ys587/__Data/__whistle/__whistle_oswald'
    if deployment is None:
        deployment = ['HICEAS2002', 'PICEAS2005', 'STAR2000', 'STAR2003',
                      'STAR2006']

    data_raw = []
    for dd in deployment:
        print(dd)
        deploy_folder = os.path.join(whistle_data, dd)
        folder_namelist = os.listdir(deploy_folder)
        for ff in folder_namelist:
            print(ff)
            data_raw.append([dd, ff, os.path.join(whistle_data, dd, ff)])
    df_sound_oswald = pd.DataFrame(data_raw, columns=['deployment', 'folder', 'path'])
    # get species name
    df_sound_oswald['species_name'] = df_sound_oswald['folder'].str.extract(r'([a-zA-Z_]*)\s')
    # get encounter name
    df_sound_oswald['encounter'] = df_sound_oswald['folder'].str.extract(r'\s([a-zA-Z]\d+)')
    # species to code
    df_sound_oswald['species'] = df_sound_oswald['species_name'].apply(lambda x: species_to_code[x])

    df_info_oswald = sound_file_info(df_sound_oswald)

    df_sound_oswald.to_csv(csv_oswald_sound, index=False)
    df_info_oswald.to_csv(csv_oswald_info, index=False)

    return df_sound_oswald, df_info_oswald


def df_sound_info_gillispie(csv_gillispie_sound, csv_gillispie_info, whistle_data=None,
                            deployment=None):
    if whistle_data is None:
        whistle_data = '/home/ys587/__Data/__whistle/__whistle_gillispie'
    if deployment is None:
        deployment = ['48kHz_to_48kHz', '96kHz_to_48kHz']
        # deployment = ['48kHz_small']  # test

    if os.path.exists(csv_gillispie_sound) & os.path.exists(csv_gillispie_info):
        df_sound_gillispie = pd.read_csv(csv_gillispie_sound)
        df_info_gillispie = pd.read_csv(csv_gillispie_info)
    else:
        data_raw = []
        for dd in deployment:
            folder_namelist = os.listdir(os.path.join(whistle_data, dd))
            for ff in folder_namelist:
                data_raw.append([dd, ff, os.path.join(whistle_data, dd, ff)])

        df_sound_gillispie = pd.DataFrame(data_raw, columns=['deployment', 'folder', 'path'])
        df_sound_gillispie['species'] = df_sound_gillispie['folder']

        df_info_gillispie = sound_file_info(df_sound_gillispie)

        df_sound_gillispie.to_csv(csv_gillispie_sound, index=False)
        df_info_gillispie.to_csv(csv_gillispie_info, index=False)

    return df_sound_gillispie, df_info_gillispie


def df_sound_info_dclde2011(csv_dclde2011_sound, csv_dclde2011_info, whistle_data=None):
    if whistle_data is None:
        whistle_data = '/home/ys587/__Data/__whistle/__sound_species'

    species_name = {'bottlenose': 'BD', 'common': 'CD', 'melon-headed': 'MH', 'spinner': 'SPIN'}

    if os.path.exists(csv_dclde2011_sound) & os.path.exists(csv_dclde2011_info):
        df_sound_dclde2011 = pd.read_csv(csv_dclde2011_sound)
        df_info_dclde2011 = pd.read_csv(csv_dclde2011_info)
    else:
        data_raw = []
        folder_namelist = os.listdir(whistle_data)
        for ff in folder_namelist:
            data_raw.append([ff, os.path.join(whistle_data, ff), species_name[ff]])

        df_sound_dclde2011 = pd.DataFrame(data_raw, columns=['folder', 'path', 'species'])
        # df_sound_dclde2011['species'] = species_name[df_sound_dclde2011['folder']]

        df_info_dclde2011 = sound_file_info(df_sound_dclde2011)

        df_sound_dclde2011.to_csv(csv_dclde2011_sound, index=False)
        df_info_dclde2011.to_csv(csv_dclde2011_info, index=False)

    return df_sound_dclde2011, df_info_dclde2011


def df_sound_info_watkin(csv_watkin_sound, csv_watkin_info, whistle_data=None,
                            deployment=None):
    if whistle_data is None:
        whistle_data = '/home/ys587/__Data/__whistle/__whistle_watkin'
    if deployment is None:
        deployment = ['all_cuts_48k', 'best_of_cuts_48k']

    species_name = {'BD19D': 'BD', 'BD10A': 'MH', 'BD3A': 'CD', 'BD15C': 'STR', 'BD15A': 'SPT', 'BD15L': 'SPIN',
                    'BE3D': 'PLT', 'BD4A': 'RD', 'BD17A': 'RT', 'BD6A': 'WSD', 'BE9A': 'FKW', 'BB1A': 'BEL',
                    'BE7A': 'KW', 'BD6B': 'WBD', 'BD6H': 'DUSK', 'BD5A': 'FRA', 'BZZZ': 'PKW', 'BE3C': 'LPLT',
                    'BB2A': 'NAR', 'BD15B': 'CLY', 'BA2A': 'SPE', 'BD15F': 'ASP'}

    if os.path.exists(csv_watkin_sound) & os.path.exists(csv_watkin_info):
        df_sound_watkin = pd.read_csv(csv_watkin_sound)
        df_info_watkin = pd.read_csv(csv_watkin_info)
    else:
        data_raw = []
        for dd in deployment:
            folder_namelist = os.listdir(os.path.join(whistle_data, dd))
            for ff in folder_namelist:
                data_raw.append([dd, ff, os.path.join(whistle_data, dd, ff), species_name[ff]])

        df_sound_watkin = pd.DataFrame(data_raw, columns=['deployment', 'folder', 'path', 'species'])
        # df_sound_watkin['species'] = df_sound_watkin['folder']

        df_info_watkin = sound_file_info(df_sound_watkin)

        df_sound_watkin.to_csv(csv_watkin_sound, index=False)
        df_info_watkin.to_csv(csv_watkin_info, index=False)

    return df_sound_watkin, df_info_watkin


def read_features_from_files(npz_path, species_list):
    species_fea_part = dict()
    for ss in species_list:
        print('Reading features from files ' + ss + ':')
        fea_species_part_list = glob.glob(os.path.join(npz_path, ss+'*.npz'))
        fea_pos_part_list = []
        fea_neg_part_list = []
        for ff in fea_species_part_list:
            fea_curr = np.load(ff)
            fea_pos_part_list.append(fea_curr['fea_pos'])
            fea_neg_part_list.append(fea_curr['fea_neg'])
            fea_curr_tmp = fea_curr['fea_pos']
            if np.isnan(fea_curr_tmp).sum() > 0:
                print('gotcha')

        if len(fea_pos_part_list) >= 1:
            fea_pos_part = np.concatenate(fea_pos_part_list, axis=0)
        elif len(fea_pos_part_list) == 0:
            continue
        else:
            fea_pos_part = fea_pos_part_list[0]

        if len(fea_neg_part_list) >= 1:
            # fea_neg_part = np.concatenate(fea_neg_part_list, axis=0)
            fea_neg_part = np.concatenate(fea_neg_part_list[::2], axis=0)  # every other ones
        elif len(fea_neg_part_list) == 0:
            continue

        species_fea_part.update({ss: {'fea_pos': fea_pos_part}})
        species_fea_part.update({'NO': {'fea_pos': fea_neg_part}})

    return species_fea_part


def combine_features_from_dict(species_fea_part, fea_part_out, fea_out_filename, species_id):
    print("combine features into part data sets")
    fea_part_list = []
    label_part_list = []
    for key, value in species_fea_part.items():
        print('Combining features from ' + key + ':')
    
        # fea_species = value['fea_pos']
        fea_species = value['fea_pos'].astype('float32')  # not yet test.
        fea_part_list.append(fea_species)
        label = [species_id[key]] * fea_species.shape[0]
        label_part_list.append(label)
    fea_part_4d = np.concatenate(fea_part_list, axis=0)
    label_part = np.concatenate(label_part_list, axis=0)
    del species_fea_part
    del fea_part_list
    del label_part_list
    try:
        np.savez(os.path.join(fea_part_out, fea_out_filename), fea_part_4d=fea_part_4d, label_part=label_part)
    except OSError:
        print('Cannnot find the output folder')

    return fea_part_4d, label_part


# def unit_vector(fea_4d):
#     for ii in range(fea_4d.shape[0]):
#         fea_sum = np.abs(fea_4d[ii, :, :, :]).sum()
#         if fea_sum:
#             fea_4d[ii, :, :, :] = fea_4d[ii, :, :, :]/fea_sum
#         else:
#             fea_4d[ii, :, :, :] = np.zeros((fea_4d.shape[1], fea_4d.shape[2], 1))
#
#     return fea_4d


# def powerlaw(spectro_mat, nu1=1., nu2=2., gamma=1.):
#     dim_f, dim_t = spectro_mat.shape
#
#     mu_k = [powelaw_find_mu(spectro_mat[ff, :]) for ff in range(dim_f)]
#     mat0 = spectro_mat ** gamma - np.array(mu_k).reshape(dim_f, 1) * np.ones(
#         (1, dim_t))
#     mat_a_denom = [(np.sum(mat0[:, tt] ** 2.)) ** .5 for tt in range(dim_t)]
#     mat_a = mat0 / (np.ones((dim_f, 1)) * np.array(mat_a_denom).reshape(1, dim_t))
#
#     mat_b_denom = [(np.sum(mat0[ff, :] ** 2.)) ** .5 for ff in range(dim_f)]
#     mat_b = mat0 / (np.array(mat_b_denom).reshape(dim_f, 1) * np.ones((1, dim_t)))
#
#     mat_a = mat_a * (mat_a > 0)  # set negative values into zero
#     mat_b = mat_b * (mat_b > 0)
#
#     whistle_powerlaw = (mat_a ** (2.0 * nu1)) * (mat_b ** (2.0 * nu2))
#
#     return whistle_powerlaw


# def powelaw_find_mu(time_f):
#     time_f_sorted = np.sort(time_f)
#     spec_half_len = int(np.floor(time_f_sorted.shape[0] * .5))
#     ind_j = np.argmin(
#         time_f_sorted[spec_half_len:spec_half_len * 2] - time_f_sorted[0:spec_half_len])
#     mu = np.mean(time_f_sorted[ind_j:ind_j + spec_half_len])
#
#     return mu
#
#
# def powerlawsym(spectro_mat, nu1=2., nu2=2., gamma=1.):
#     dim_f, dim_t = spectro_mat.shape
#
#     mu_k = [powelaw_find_mu(spectro_mat[ff, :]) for ff in range(dim_f)]
#     mat0 = spectro_mat ** gamma - np.array(mu_k).reshape(dim_f, 1) * np.ones(
#         (1, dim_t))
#
#     mat_a_denom = [(np.sum(mat0[:, tt] ** 2.)) ** .5 for tt in range(dim_t)]
#     mat_a = mat0 / (
#                 np.ones((dim_f, 1)) * np.array(mat_a_denom).reshape(1, dim_t))
#
#     mu_t = [powelaw_find_mu(spectro_mat[:, tt].T) for tt in range(dim_t)]
#     mat1 = spectro_mat ** gamma - np.ones((dim_f, 1))*np.array(mu_t).reshape(1, dim_t)
#     mat_b_denom = [(np.sum(mat1[ff, :] ** 2.)) ** .5 for ff in range(dim_f)]
#     mat_b = mat1 / (np.array(mat_b_denom).reshape(dim_f, 1) * np.ones((1, dim_t)))
#
#     mat_a = mat_a * (mat_a > 0)  # set negative values into zero
#     mat_b = mat_b * (mat_b > 0)
#
#     whistle_powerlaw = (mat_a ** (2.0 * nu1)) * (mat_b ** (2.0 * nu2))
#
#     return whistle_powerlaw


# def nopulse_separation(spectro_mat, harm_dim=(15, 1), per_dim=(1, 15)):
#     harmonic_filter = np.asarray(harm_dim, dtype=int)
#     percussion_filter = np.asarray(per_dim, dtype=int)
#     harmonic_slice = median_filter(spectro_mat, harmonic_filter)
#     percussion_slice = median_filter(spectro_mat, percussion_filter)
#     # harmonic_mask = harmonic_slice > percussion_slice  # binary
#     p_mask = 2.0
#     harmonic_slice_ = harmonic_slice**p_mask
#     percussion_slice_ = percussion_slice**p_mask
#     slice_sum = harmonic_slice_ + percussion_slice_
#
#     # spectro_mat_nopulse = spectro_mat * (harmonic_slice_ / slice_sum)
#     spectro_mat_nopulse = spectro_mat * (percussion_slice_ / slice_sum)
#     return spectro_mat_nopulse


# def feature_extract(spectro, freq_low=0):
#     spectro_median = nopulse_median(spectro[freq_low:, :])
#     spectro_fea = (avg_sub(spectro_median)).T
#
#     return spectro_fea


def feature_whistleness(spectro, use_pcen=True, remove_pulse=True, unit_vec=True, freq_low=0):
    if use_pcen:
        spectro = librosa.pcen(spectro * (2 ** 31))  # apply pcen
    if remove_pulse:
        spectro = nopulse_median(spectro[freq_low:, :])  # remove the pulsive noise / click
    spectro_fea = (avg_sub(spectro)).T  # remove the tonal noise

    if unit_vec:
        # vec_len = np.sqrt(np.sum(spectro_fea ** 2.))  # length of vector "spectro"
        spectro_fea = spectro_fea - spectro_fea.mean()
        vec_len = np.sum(np.abs(spectro_fea))
        spectro_fea = spectro_fea/vec_len if vec_len else np.zeros(spectro_fea.shape)

    return spectro_fea


# def fea_pcen_nopulse(samples, conf_samplerate, conf_hop_length, freq_min=64, freq_max=128):
#     mel_spectrogram = librosa.feature.melspectrogram(
#         samples, sr=conf_samplerate, hop_length=conf_hop_length, power=1)
#
#     whistle_pcen_no_pulse = fea_pcen_nopulse_from_mel(mel_spectrogram,
#                                                       freq_min=freq_min,
#                                                       freq_max=freq_max)
#
#     return whistle_pcen_no_pulse
#
#
# def fea_pcen_nopulse_from_mel(melspectro, freq_min=64, freq_max=None):
#     if freq_max is None:
#         freq_max = melspectro.shape[0]
#     whistle_freq = librosa.pcen(melspectro * (2 ** 31))
#     whistle_freq = nopulse_separation(whistle_freq)
#     whistle_freq = whistle_freq[freq_min:freq_max, :] + np.finfo(float).eps
#
#     fea_sum = np.abs(whistle_freq).sum()
#     if fea_sum > 0.0:
#         whistle_freq = whistle_freq / fea_sum
#     else:
#         whistle_freq = np.zeros(
#             (whistle_freq.shape[0], whistle_freq.shape[1]))
#
#     return whistle_freq
#
#
# def fea_powerlaw(samples, conf_samplerate, conf_hop_length):
#     mel_spectrogram = librosa.feature.melspectrogram(
#         samples, sr=conf_samplerate, hop_length=conf_hop_length, power=1)
#     whistle_powerlaw = powerlaw(mel_spectrogram)
#
#     return whistle_powerlaw
#
#
# def fea_pcen(samples, conf_samplerate, conf_hop_length):
#     mel_spectrogram = librosa.feature.melspectrogram(
#         samples, sr=conf_samplerate, hop_length=conf_hop_length, power=1)
#     whistle_pcen = librosa.pcen(mel_spectrogram * (2 ** 31))
#
#     return whistle_pcen
#
#
# def fea_powerlawsym(samples, conf_samplerate, conf_hop_length):
#     mel_spectrogram = librosa.feature.melspectrogram(
#         samples, sr=conf_samplerate, hop_length=conf_hop_length, power=1)
#     whistle_powerlawsym = powerlawsym(mel_spectrogram)
#
#     return whistle_powerlawsym


def extract_fea_oswald(df_sound_oswald, model_name, fea_out, seltab_out,
                       use_pcen=True,
                       remove_pulse=True,
                       conf_samplerate=48000,
                       conf_win_size=1.,
                       conf_hop_size=0.5,
                       conf_hop_length=int(0.02*48000),  # int(conf['time_reso']*conf['sample_rate'])
                       conf_time_multi=floor(1./0.02),  # floor(conf['win_size'] / conf['time_reso'])
                       conf_time_multi_hop=floor(0.5/0.02),  # floor(conf['hop_size'] / conf['time_reso'])
                       conf_whistle_thre_pos=0.9,
                       conf_whistle_thre_neg=0.3,
                       ):
    classifier_model = load_model(model_name)

    for index, row in df_sound_oswald.iterrows():
        print('Acoustic encounter ' + str(index + 1) + '/' + str(
            len(df_sound_oswald)) + ': ' + row['folder'])

        wav_list = glob.glob(row['path'] + '/*.wav')
        wav_list.sort()
        whistle_time_start_pos = []
        whistle_time_end_pos = []
        whistle_score_pos = []
        begin_path_pos = []
        file_offset_pos = []
        whistle_time_start_neg = []
        whistle_time_end_neg = []
        whistle_score_neg = []
        begin_path_neg = []
        file_offset_neg = []
        whistle_image_4d_pos_list = []
        whistle_image_4d_neg_list = []

        for ww in wav_list:
            print(os.path.basename(ww))
            # ww = os.path.join(row['path'], ww0)
            samples, _ = librosa.load(ww, sr=conf_samplerate)
            if np.ndim(samples) > 1:
                samples = samples[0]

            whistle_freq = librosa.feature.melspectrogram(samples,
                                                          sr=conf_samplerate,
                                                          hop_length=conf_hop_length,
                                                          power=1)
            whistle_freq_list = []
            win_num = floor((whistle_freq.shape[1] - conf_time_multi) / conf_time_multi_hop) + 1  # 0.5s hop

            if win_num > 0:
                for nn in range(win_num):
                    whistle_freq_curr = whistle_freq[:,
                                        nn * conf_time_multi_hop:
                                        nn * conf_time_multi_hop + conf_time_multi]
                    whistle_freq_curr = feature_whistleness(whistle_freq_curr, use_pcen, remove_pulse)
                    whistle_freq_list.append(whistle_freq_curr)

                if len(whistle_freq_list) >= 2:
                    whistle_image = np.stack(whistle_freq_list)
                else:
                    whistle_image = np.expand_dims(whistle_freq_list[0], axis=0)
                whistle_image_4d = np.expand_dims(whistle_image, axis=3)

                predictions = classifier_model.predict(whistle_image_4d)
            # 3 sec!

            # extract features here for both positive & negative classes
            whistle_win_ind_pos = np.where(predictions[:, 1] > conf_whistle_thre_pos)[0]
            whistle_win_ind_neg = np.where(predictions[:, 1] < conf_whistle_thre_neg)[0]

            whistle_image_4d_pos_list.append(
                whistle_image_4d[whistle_win_ind_pos, :, :, :])
            whistle_image_4d_neg_list.append(
                whistle_image_4d[whistle_win_ind_neg, :, :, :])

            if whistle_win_ind_pos.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_pos * conf_hop_size
                whistle_time_start_pos.append(whistle_time_start_curr)
                whistle_time_end_pos.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_pos.append(predictions[:, 1][whistle_win_ind_pos])
                begin_path_pos.append([ww] * whistle_win_ind_pos.shape[0])
                file_offset_pos.append(whistle_win_ind_pos * conf_hop_size)

            if whistle_win_ind_neg.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_neg * conf_hop_size
                whistle_time_start_neg.append(whistle_time_start_curr)
                whistle_time_end_neg.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_neg.append(predictions[:, 1][whistle_win_ind_neg])
                begin_path_neg.append([ww] * whistle_win_ind_neg.shape[0])
                file_offset_neg.append(whistle_win_ind_neg * conf_hop_size)

        # if fea_type is not None:
        whistle_image_4d_pos = np.concatenate(whistle_image_4d_pos_list)
        whistle_image_4d_neg = np.concatenate(whistle_image_4d_neg_list)
        fea_out_file = os.path.join(fea_out, row['species'] + '_' + row[
            'deployment'] + '_' + row['encounter'] + '.npz')
        np.savez(fea_out_file, fea_pos=whistle_image_4d_pos, fea_neg=whistle_image_4d_neg)

        # make sound selection table
        if len(whistle_time_start_pos) >= 1:
            if len(whistle_time_start_pos) >= 2:
                whistle_time_start_pos = np.concatenate(whistle_time_start_pos)
                whistle_time_end_pos = np.concatenate(whistle_time_end_pos)
                whistle_score_pos = np.concatenate(whistle_score_pos)
                begin_path_pos = np.concatenate(begin_path_pos)
                file_offset_pos = np.concatenate(file_offset_pos)
            else:  # == 1
                whistle_time_start_pos = whistle_time_start_pos[0]
                whistle_time_end_pos = whistle_time_end_pos[0]
                whistle_score_pos = whistle_score_pos[0]
                begin_path_pos = begin_path_pos[0]
                file_offset_pos = file_offset_pos[0]
            seltab_out_file_pos = os.path.join(seltab_out,
                                           row['species'] + '_' + row[
                                               'deployment'] + '_' + row[
                                               'encounter'] + '_pos.txt')
            make_sound_sel_table(seltab_out_file_pos,
                                             whistle_time_start_pos,
                                             whistle_time_end_pos, begin_path_pos,
                                             file_offset_pos, whistle_score_pos,
                                             conf_whistle_thre_pos)
        if len(whistle_time_start_neg) >= 1:
            if len(whistle_time_start_neg) >= 2:
                whistle_time_start_neg = np.concatenate(
                    whistle_time_start_neg)
                whistle_time_end_neg = np.concatenate(whistle_time_end_neg)
                whistle_score_neg = np.concatenate(whistle_score_neg)
                begin_path_neg = np.concatenate(begin_path_neg)
                file_offset_neg = np.concatenate(file_offset_neg)
            else:  # == 1
                whistle_time_start_neg = whistle_time_start_neg[0]
                whistle_time_end_neg = whistle_time_end_neg[0]
                whistle_score_neg = whistle_score_neg[0]
                begin_path_neg = begin_path_neg[0]
                file_offset_neg = file_offset_neg[0]
            seltab_out_file_neg = os.path.join(seltab_out,
                                               row['species'] + '_' + row[
                                                   'deployment'] + '_' + row[
                                                   'encounter'] + '_neg.txt')
            make_sound_sel_table(seltab_out_file_neg, whistle_time_start_neg,
                                 whistle_time_end_neg, begin_path_neg,
                                 file_offset_neg, whistle_score_neg,
                                 conf_whistle_thre_neg)
    return None


def extract_fea_gillispie(df_sound_gillispie, model_name, fea_out, seltab_out,
                          use_pcen=True,
                          remove_pulse=True,
                          conf_samplerate=48000,
                          # conf_time_reso=0.02,
                          conf_win_size=1.,
                          conf_hop_size=0.5,
                          conf_hop_length=int(0.02*48000),  # int(conf['time_reso']*conf['sample_rate'])
                          conf_time_multi=floor(1./0.02),  # floor(conf['win_size'] / conf['time_reso'])
                          conf_time_multi_hop=floor(0.5/0.02),  # floor(conf['hop_size'] / conf['time_reso'])
                          conf_whistle_thre_pos=0.9,
                          conf_whistle_thre_neg=0.3,
                          ):
    classifier_model = load_model(model_name)
    for index, row in df_sound_gillispie.iterrows():
    # for index, row in df_sound_gillispie[:1].iterrows():  # DEBUG
        print('Species ' + str(index + 1) + '/' + str(len(df_sound_gillispie)) + ': ' + row['folder'])

        wav_list = glob.glob(row['path'] + '/*.wav')
        wav_list.sort()
        whistle_time_start_pos = []
        whistle_time_end_pos = []
        whistle_score_pos = []
        begin_path_pos = []
        file_offset_pos = []
        whistle_time_start_neg = []
        whistle_time_end_neg = []
        whistle_score_neg = []
        begin_path_neg = []
        file_offset_neg = []

        for ww in wav_list:
            print(os.path.basename(ww))
            samples, _ = librosa.load(ww, sr=conf_samplerate)
            if np.ndim(samples) > 1:
                samples = samples[0]

            whistle_freq = librosa.feature.melspectrogram(samples,
                                                          sr=conf_samplerate,
                                                          hop_length=conf_hop_length,
                                                          power=1)

            whistle_freq_list = []
            win_num = floor(
                (whistle_freq.shape[1] - conf_time_multi) / conf_time_multi_hop) + 1  # 0.5s hop

            if win_num > 0:
                for nn in range(win_num):
                    whistle_freq_curr = whistle_freq[:,
                                        nn * conf_time_multi_hop:
                                        nn * conf_time_multi_hop + conf_time_multi]

                    # whistle_freq_curr = fea_pcen_nopulse_from_mel(whistle_freq_curr)
                    whistle_freq_curr = feature_whistleness(whistle_freq_curr, use_pcen, remove_pulse)
                    whistle_freq_list.append(whistle_freq_curr)

                if len(whistle_freq_list) >= 2:
                    whistle_image = np.stack(whistle_freq_list)
                else:
                    whistle_image = np.expand_dims(whistle_freq_list[0],
                                                   axis=0)
                whistle_image_4d = np.expand_dims(whistle_image, axis=3)

                predictions = classifier_model.predict(whistle_image_4d)

            # extract features here for both positive & negative classes
            whistle_win_ind_pos = np.where(predictions[:, 1] > conf_whistle_thre_pos)[0]
            whistle_win_ind_neg = np.where(predictions[:, 1] < conf_whistle_thre_neg)[0]
            fea_out_file = os.path.join(fea_out, row['species'] + '_' + row[
                'deployment'] + '_' + os.path.splitext(os.path.basename(ww))[
                                            0] + '.npz')
            # if fea_type == 'pcen_nopulse':
            np.savez(fea_out_file,
                     fea_pos=whistle_image_4d[whistle_win_ind_pos, :, :, :],
                     fea_neg=whistle_image_4d[whistle_win_ind_neg, :, :, :])

            if whistle_win_ind_pos.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_pos * conf_hop_size
                whistle_time_start_pos.append(whistle_time_start_curr)
                whistle_time_end_pos.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_pos.append(
                    predictions[:, 1][whistle_win_ind_pos])
                begin_path_pos.append([ww] * whistle_win_ind_pos.shape[0])
                file_offset_pos.append(whistle_win_ind_pos * conf_hop_size)
            if whistle_win_ind_neg.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_neg * conf_hop_size
                whistle_time_start_neg.append(whistle_time_start_curr)
                whistle_time_end_neg.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_neg.append(
                    predictions[:, 1][whistle_win_ind_neg])
                begin_path_neg.append([ww] * whistle_win_ind_neg.shape[0])
                file_offset_neg.append(whistle_win_ind_neg * conf_hop_size)

        # make sound selection table
        if len(whistle_time_start_pos) >= 1:
            if len(whistle_time_start_pos) >= 2:
                whistle_time_start_pos = np.concatenate(whistle_time_start_pos)
                whistle_time_end_pos = np.concatenate(whistle_time_end_pos)
                whistle_score_pos = np.concatenate(whistle_score_pos)
                begin_path_pos = np.concatenate(begin_path_pos)
                file_offset_pos = np.concatenate(file_offset_pos)
            else:  # == 1
                whistle_time_start_pos = whistle_time_start_pos[0]
                whistle_time_end_pos = whistle_time_end_pos[0]
                whistle_score_pos = whistle_score_pos[0]
                begin_path_pos = begin_path_pos[0]
                file_offset_pos = file_offset_pos[0]
            seltab_out_file = os.path.join(seltab_out,
                                           row['species'] + '_' + row[
                                               'deployment'] + '_pos.txt')
            make_sound_sel_table(seltab_out_file,
                                             whistle_time_start_pos,
                                             whistle_time_end_pos, begin_path_pos,
                                             file_offset_pos, whistle_score_pos,
                                             conf_whistle_thre_pos)

        if len(whistle_time_start_neg) >= 1:
            if len(whistle_time_start_neg) >= 2:
                whistle_time_start_neg = np.concatenate(whistle_time_start_neg)
                whistle_time_end_neg = np.concatenate(whistle_time_end_neg)
                whistle_score_neg = np.concatenate(whistle_score_neg)
                begin_path_neg = np.concatenate(begin_path_neg)
                file_offset_neg = np.concatenate(file_offset_neg)
            else:  # == 1
                whistle_time_start_neg = whistle_time_start_neg[0]
                whistle_time_end_neg = whistle_time_end_neg[0]
                whistle_score_neg = whistle_score_neg[0]
                begin_path_neg = begin_path_neg[0]
                file_offset_neg = file_offset_neg[0]
            seltab_out_file = os.path.join(seltab_out,
                                           row['species'] + '_' + row[
                                               'deployment'] + '_neg.txt')
            make_sound_sel_table(seltab_out_file,
                                             whistle_time_start_neg,
                                             whistle_time_end_neg, begin_path_neg,
                                             file_offset_neg, whistle_score_neg,
                                             conf_whistle_thre_neg)


def extract_fea_dclde2011(df_sound_dclde2011, model_name, fea_out, seltab_out,
                          use_pcen=True,
                          remove_pulse=True,
                          conf_samplerate=48000,
                          # conf_time_reso=0.02,
                          conf_win_size=1.,
                          conf_hop_size=0.5,
                          conf_hop_length=int(0.02*48000),  # int(conf['time_reso']*conf['sample_rate'])
                          conf_time_multi=floor(1./0.02),  # floor(conf['win_size'] / conf['time_reso'])
                          conf_time_multi_hop=floor(0.5/0.02),  # floor(conf['hop_size'] / conf['time_reso'])
                          conf_whistle_thre_pos=0.9,
                          conf_whistle_thre_neg=0.3,
                          ):
    classifier_model = load_model(model_name)
    for index, row in df_sound_dclde2011.iterrows():
        print('Species ' + str(index + 1) + '/' + str(len(df_sound_dclde2011)) + ': ' + row['folder'])

        wav_list = glob.glob(row['path'] + '/*.wav')
        wav_list.sort()
        whistle_time_start_pos = []
        whistle_time_end_pos = []
        whistle_score_pos = []
        begin_path_pos = []
        file_offset_pos = []
        whistle_time_start_neg = []
        whistle_time_end_neg = []
        whistle_score_neg = []
        begin_path_neg = []
        file_offset_neg = []

        for ww in wav_list:
            print(os.path.basename(ww))
            samples, _ = librosa.load(ww, sr=conf_samplerate)
            if np.ndim(samples) > 1:
                samples = samples[0]

            whistle_freq = librosa.feature.melspectrogram(samples,
                                                          sr=conf_samplerate,
                                                          hop_length=conf_hop_length,
                                                          power=1)

            whistle_freq_list = []
            win_num = floor(
                (whistle_freq.shape[1] - conf_time_multi) / conf_time_multi_hop) + 1  # 0.5s hop

            if win_num > 0:
                for nn in range(win_num):
                    whistle_freq_curr = whistle_freq[:,
                                        nn * conf_time_multi_hop:
                                        nn * conf_time_multi_hop + conf_time_multi]

                    # whistle_freq_curr = fea_pcen_nopulse_from_mel(whistle_freq_curr)
                    whistle_freq_curr = feature_whistleness(whistle_freq_curr, use_pcen, remove_pulse)
                    whistle_freq_list.append(whistle_freq_curr)

                if len(whistle_freq_list) >= 2:
                    whistle_image = np.stack(whistle_freq_list)
                else:
                    whistle_image = np.expand_dims(whistle_freq_list[0],
                                                   axis=0)
                whistle_image_4d = np.expand_dims(whistle_image, axis=3)

                predictions = classifier_model.predict(whistle_image_4d)

            # extract features here for both positive & negative classes
            whistle_win_ind_pos = np.where(predictions[:, 1] > conf_whistle_thre_pos)[0]
            whistle_win_ind_neg = np.where(predictions[:, 1] < conf_whistle_thre_neg)[0]
            fea_out_file = os.path.join(fea_out, row['species'] + '_' + os.path.splitext(os.path.basename(ww))[
                                            0] + '.npz')
            np.savez(fea_out_file,
                     fea_pos=whistle_image_4d[whistle_win_ind_pos, :, :, :],
                     fea_neg=whistle_image_4d[whistle_win_ind_neg, :, :, :])

            if whistle_win_ind_pos.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_pos * conf_hop_size
                whistle_time_start_pos.append(whistle_time_start_curr)
                whistle_time_end_pos.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_pos.append(
                    predictions[:, 1][whistle_win_ind_pos])
                begin_path_pos.append([ww] * whistle_win_ind_pos.shape[0])
                file_offset_pos.append(whistle_win_ind_pos * conf_hop_size)
            if whistle_win_ind_neg.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_neg * conf_hop_size
                whistle_time_start_neg.append(whistle_time_start_curr)
                whistle_time_end_neg.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_neg.append(
                    predictions[:, 1][whistle_win_ind_neg])
                begin_path_neg.append([ww] * whistle_win_ind_neg.shape[0])
                file_offset_neg.append(whistle_win_ind_neg * conf_hop_size)

        # make sound selection table
        if len(whistle_time_start_pos) >= 1:
            if len(whistle_time_start_pos) >= 2:
                whistle_time_start_pos = np.concatenate(whistle_time_start_pos)
                whistle_time_end_pos = np.concatenate(whistle_time_end_pos)
                whistle_score_pos = np.concatenate(whistle_score_pos)
                begin_path_pos = np.concatenate(begin_path_pos)
                file_offset_pos = np.concatenate(file_offset_pos)
            else:  # == 1
                whistle_time_start_pos = whistle_time_start_pos[0]
                whistle_time_end_pos = whistle_time_end_pos[0]
                whistle_score_pos = whistle_score_pos[0]
                begin_path_pos = begin_path_pos[0]
                file_offset_pos = file_offset_pos[0]
            seltab_out_file = os.path.join(seltab_out,
                                           row['species'] + '_pos.txt')
            make_sound_sel_table(seltab_out_file,
                                             whistle_time_start_pos,
                                             whistle_time_end_pos, begin_path_pos,
                                             file_offset_pos, whistle_score_pos,
                                             conf_whistle_thre_pos)

        if len(whistle_time_start_neg) >= 1:
            if len(whistle_time_start_neg) >= 2:
                whistle_time_start_neg = np.concatenate(whistle_time_start_neg)
                whistle_time_end_neg = np.concatenate(whistle_time_end_neg)
                whistle_score_neg = np.concatenate(whistle_score_neg)
                begin_path_neg = np.concatenate(begin_path_neg)
                file_offset_neg = np.concatenate(file_offset_neg)
            else:  # == 1
                whistle_time_start_neg = whistle_time_start_neg[0]
                whistle_time_end_neg = whistle_time_end_neg[0]
                whistle_score_neg = whistle_score_neg[0]
                begin_path_neg = begin_path_neg[0]
                file_offset_neg = file_offset_neg[0]
            seltab_out_file = os.path.join(seltab_out,
                                           row['species'] + '_neg.txt')
            make_sound_sel_table(seltab_out_file,
                                             whistle_time_start_neg,
                                             whistle_time_end_neg, begin_path_neg,
                                             file_offset_neg, whistle_score_neg,
                                             conf_whistle_thre_neg)


def extract_fea_watkin(df_sound_watkin, model_name, fea_out, seltab_out,
                          use_pcen=True,
                          remove_pulse=True,
                          conf_samplerate=48000,
                          # conf_time_reso=0.02,
                          conf_win_size=1.,
                          conf_hop_size=0.5,
                          conf_hop_length=int(0.02*48000),  # int(conf['time_reso']*conf['sample_rate'])
                          conf_time_multi=floor(1./0.02),  # floor(conf['win_size'] / conf['time_reso'])
                          conf_time_multi_hop=floor(0.5/0.02),  # floor(conf['hop_size'] / conf['time_reso'])
                          conf_whistle_thre_pos=0.9,
                          conf_whistle_thre_neg=0.3,
                          ):
    classifier_model = load_model(model_name)
    for index, row in df_sound_watkin.iterrows():
        print('Species ' + str(index + 1) + '/' + str(len(df_sound_watkin)) + ': ' + row['folder'])

        wav_list = glob.glob(row['path'] + '/*.wav')
        wav_list.sort()
        whistle_time_start_pos = []
        whistle_time_end_pos = []
        whistle_score_pos = []
        begin_path_pos = []
        file_offset_pos = []
        whistle_time_start_neg = []
        whistle_time_end_neg = []
        whistle_score_neg = []
        begin_path_neg = []
        file_offset_neg = []

        for ww in wav_list:
            print(os.path.basename(ww))
            samples, _ = librosa.load(ww, sr=conf_samplerate)
            if np.ndim(samples) > 1:
                samples = samples[0]

            if samples.shape[0] < conf_samplerate:
                samples_new = np.zeros(conf_samplerate)
                samples_new[0:samples.shape[0]] = samples
                samples = samples_new

            whistle_freq = librosa.feature.melspectrogram(samples,
                                                          sr=conf_samplerate,
                                                          hop_length=conf_hop_length,
                                                          power=1)

            whistle_freq_list = []
            win_num = floor(
                (whistle_freq.shape[1] - conf_time_multi) / conf_time_multi_hop) + 1  # 0.5s hop

            if win_num > 0:
                for nn in range(win_num):
                    whistle_freq_curr = whistle_freq[:,
                                        nn * conf_time_multi_hop:
                                        nn * conf_time_multi_hop + conf_time_multi]

                    # whistle_freq_curr = fea_pcen_nopulse_from_mel(whistle_freq_curr)
                    whistle_freq_curr = feature_whistleness(whistle_freq_curr, use_pcen, remove_pulse)
                    whistle_freq_list.append(whistle_freq_curr)

                if len(whistle_freq_list) >= 2:
                    whistle_image = np.stack(whistle_freq_list)
                else:
                    whistle_image = np.expand_dims(whistle_freq_list[0],
                                                   axis=0)
                whistle_image_4d = np.expand_dims(whistle_image, axis=3)

                predictions = classifier_model.predict(whistle_image_4d)

            # extract features here for both positive & negative classes
            whistle_win_ind_pos = np.where(predictions[:, 1] > conf_whistle_thre_pos)[0]
            whistle_win_ind_neg = np.where(predictions[:, 1] < conf_whistle_thre_neg)[0]
            fea_out_file = os.path.join(fea_out, row['species'] + '_' + os.path.splitext(os.path.basename(ww))[
                                            0] + '.npz')
            np.savez(fea_out_file,
                     fea_pos=whistle_image_4d[whistle_win_ind_pos, :, :, :],
                     fea_neg=whistle_image_4d[whistle_win_ind_neg, :, :, :])

            if whistle_win_ind_pos.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_pos * conf_hop_size
                whistle_time_start_pos.append(whistle_time_start_curr)
                whistle_time_end_pos.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_pos.append(
                    predictions[:, 1][whistle_win_ind_pos])
                begin_path_pos.append([ww] * whistle_win_ind_pos.shape[0])
                file_offset_pos.append(whistle_win_ind_pos * conf_hop_size)
            if whistle_win_ind_neg.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = whistle_win_ind_neg * conf_hop_size
                whistle_time_start_neg.append(whistle_time_start_curr)
                whistle_time_end_neg.append(
                    whistle_time_start_curr + conf_win_size)
                # detected whistle score
                whistle_score_neg.append(
                    predictions[:, 1][whistle_win_ind_neg])
                begin_path_neg.append([ww] * whistle_win_ind_neg.shape[0])
                file_offset_neg.append(whistle_win_ind_neg * conf_hop_size)

        # make sound selection table
        if len(whistle_time_start_pos) >= 1:
            if len(whistle_time_start_pos) >= 2:
                whistle_time_start_pos = np.concatenate(whistle_time_start_pos)
                whistle_time_end_pos = np.concatenate(whistle_time_end_pos)
                whistle_score_pos = np.concatenate(whistle_score_pos)
                begin_path_pos = np.concatenate(begin_path_pos)
                file_offset_pos = np.concatenate(file_offset_pos)
            else:  # == 1
                whistle_time_start_pos = whistle_time_start_pos[0]
                whistle_time_end_pos = whistle_time_end_pos[0]
                whistle_score_pos = whistle_score_pos[0]
                begin_path_pos = begin_path_pos[0]
                file_offset_pos = file_offset_pos[0]
            seltab_out_file = os.path.join(seltab_out, row['species']+'_'+row['deployment']+'_pos.txt')
            make_sound_sel_table(seltab_out_file,
                                             whistle_time_start_pos,
                                             whistle_time_end_pos, begin_path_pos,
                                             file_offset_pos, whistle_score_pos,
                                             conf_whistle_thre_pos)

        if len(whistle_time_start_neg) >= 1:
            if len(whistle_time_start_neg) >= 2:
                whistle_time_start_neg = np.concatenate(whistle_time_start_neg)
                whistle_time_end_neg = np.concatenate(whistle_time_end_neg)
                whistle_score_neg = np.concatenate(whistle_score_neg)
                begin_path_neg = np.concatenate(begin_path_neg)
                file_offset_neg = np.concatenate(file_offset_neg)
            else:  # == 1
                whistle_time_start_neg = whistle_time_start_neg[0]
                whistle_time_end_neg = whistle_time_end_neg[0]
                whistle_score_neg = whistle_score_neg[0]
                begin_path_neg = begin_path_neg[0]
                file_offset_neg = file_offset_neg[0]
            seltab_out_file = os.path.join(seltab_out,
                                           row['species']+'_'+row['deployment']+'_neg.txt')
            make_sound_sel_table(seltab_out_file,
                                             whistle_time_start_neg,
                                             whistle_time_end_neg, begin_path_neg,
                                             file_offset_neg, whistle_score_neg,
                                             conf_whistle_thre_neg)


def nopulse_median(spectro_mat, per_dim=(15, 1)):
    """
    Gillispie's median filter
    """
    percussion_filter = np.asarray(per_dim, dtype=int)
    spectro_mat += 1. / 32767
    # spectro_mat = 10*np.log10(spectro_mat)
    percussion_slice = median_filter(spectro_mat, percussion_filter)
    spectro_mat_nopulse = spectro_mat - percussion_slice

    return spectro_mat_nopulse


def spectro_median(fea_img):
    """
    Apply nopulse_median on the whole spectrogram
    :param fea_img:
    :return:
    """
    fea_nopulse_list = []
    for ii in range(fea_img.shape[0]):
        fea_nopulse_curr = nopulse_median(fea_img[ii, :, :])
        fea_nopulse_list.append(fea_nopulse_curr)
    fea_nopulse = np.stack(fea_nopulse_list)

    return fea_nopulse


# average subtraction: input=fea_median, output=fea_noiseless
def avg_sub(fea_img, alpha=0.003787050936477576):
    """
    alpha = 1.0-np.exp(np.log(0.15)*.02/10. = 0.003787050936477576
    :param fea_img:
    :param alpha:
    :return:
    """
    bg_noise = np.zeros(fea_img.shape)
    bg_noise[:, 0] = fea_img[:, 0]

    for tt in range(1, fea_img.shape[1]):
        bg_noise[:, tt] = alpha * fea_img[:, tt] + (1.0 - alpha) * bg_noise[
                                                                     :, tt - 1]

    return fea_img - bg_noise


def spectro_nonoise(fea_img, alpha=0.003787050936477576):
    """
    alpha = 1.0-np.exp(np.log(0.15)*.02/10. = 0.003787050936477576
    :param fea_input:
    :param alpha:
    :return:
    """
    fea_nopulse_list = []
    for ii in range(fea_img.shape[0]):
        fea_nopulse_curr = avg_sub(fea_img[ii, :, :], alpha)
        fea_nopulse_list.append(fea_nopulse_curr)
    fea_nopulse = np.stack(fea_nopulse_list)

    return fea_nopulse


# Gaussian smoothing
def gaussian_smoothing(spectro_mat):
    """
    Gaussian smoothing
    """
    spectro_smooth = cv2.GaussianBlur(spectro_mat, (3, 3), cv2.BORDER_DEFAULT)

    return spectro_smooth


def spectro_smooth(fea_img):
    fea_smooth_list = []
    for ii in range(fea_img.shape[0]):
        fea_smooth_curr = gaussian_smoothing(fea_img[ii, :, :])
        fea_smooth_list.append(fea_smooth_curr)
    fea_smooth = np.stack(fea_smooth_list)

    return fea_smooth


def data_generator(whistle_image_target_4d, label_target_cat, batch_size, network_type='cnn'):
    num_samples = label_target_cat.shape[0]
    num_batch = int(floor(num_samples/batch_size))
    if network_type == 'cnn':  # cnn
        whistle_image_target_4d =  whistle_image_target_4d[:num_batch*batch_size,:,:,:]
        while 1:
            for i in range(num_batch):
                # 1875 * 32 = 60000 -> # of training samples
                yield whistle_image_target_4d[
                      i * batch_size:(i + 1) * batch_size, :, :, :], \
                      label_target_cat[i * batch_size:(i + 1) * batch_size,:]
    elif network_type == 'rnn':
        whistle_image_target_4d = whistle_image_target_4d[:num_batch * batch_size, :, :]
        while 1:
            for i in range(num_batch):
                # print(i)
                yield whistle_image_target_4d[
                      i * batch_size:(i + 1) * batch_size, :, :], \
                      label_target_cat[i * batch_size:(i + 1) * batch_size,:]
    else:  # conv2d_lstm
        whistle_image_target_4d = whistle_image_target_4d[:num_batch * batch_size, :, :, :, :]
        while 1:
            for i in range(num_batch):
                # print(i)
                yield whistle_image_target_4d[
                      i * batch_size:(i + 1) * batch_size, :, :], \
                      label_target_cat[i * batch_size:(i + 1) * batch_size,:]

