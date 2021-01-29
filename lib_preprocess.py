"""
Preprocess module:
Utility functions of sound processing & feature preprocessing

Author: Yu Shiu
Date: May 24, 2019
"""
import numpy as np
import os
import sys
import glob
# import pickle
# import random
# from math import floor

# import pandas as pd
# import warnings
# import librosa
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences

from peak_picker.TonalClass import tonal
# from utilities.features import ztransform, non_ztransform, rocca
# from scipy.signal import butter, lfilter
# import multiprocessing as mp
# from itertools import repeat
#
# from species_classifier.load_feature_model import load_fea_model

# warnings.filterwarnings("error")


def bin_extract(bin_dir, sound_dir, species_name):
    """
    Extract whistle contour information from bin files
    :param bin_dir: the folder that has the bin files
    :param sound_dir: the folder that has the sound files
    :param species_name: a dict that has the list of species names
    :return file_contour_pair: a dict  with filename as the key. The value
    consists of (i) species class label; (ii) list of [time & freq sequence]
    :return bin_wav_pair: a dict with filename as the key to the sound files.
    """
    bin_wav_pair = dict()
    file_contour_pair = dict()
    # for species in [species_name[0]]:
    for species in species_name:
        print(species)
        bin_file_list = glob.glob(os.path.join(bin_dir, species, '*.bin'))
        bin_file_list.sort()
        if len(bin_file_list) == 0:
            print('Tonal files were not found. ')
            sys.exit()
        for bb in bin_file_list:
            file_base = os.path.splitext(os.path.basename(bb))[0]
            file_name = file_base + '.wav'
            file_path = os.path.join(sound_dir, species, file_name)
            if os.path.isfile(file_path):  # bin & wav match!
                print('Process ' + file_name)
                bin_wav_pair[bb] = file_path
                # Extract contours
                tonal0 = tonal(bb)
                contour_list = []
                while True:
                    try:
                        contour_list.append(tonal0.__next__())
                    except StopIteration:
                        print("End of the bin file of " + file_base + ".bin")
                        break
                file_contour_pair[file_base] = [species, contour_list]
    return file_contour_pair, bin_wav_pair


def contour_target_retrieve(contour_target, bin_dir_target, time_reso):
    '''
    Retrieve whistle contours from .bin files
    :param contour_target:
    :param bin_dir_target:
    :param conf:
    :return:
    '''
    # time_reso = conf['time_reso']
    # duration_thre = conf["duration_thre"]
    # transform = conf['transform']
    # species_id = conf['species_id']

    duration_max = 0
    count_all = 0
    count_long = 0
    data_list = []
    df_ff_fea_list = []
    df_ff_list = []
    contour_target_list = []
    # iterate over files
    for ff in sorted(list(contour_target.keys())):
        print(ff)
        file_contour = contour_target[ff][1]
        label_contour = contour_target[ff][0]
        len_contour = len(file_contour)

        if len_contour >= 1:
            # Retrieve contours
            contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, time_reso)
            contour_target_list.append([ff, label_contour, contour_target_ff])
    return contour_target_list


def contour_retrieve(file_contour, count_all, duration_max, time_reso):
    print('Retrieving contours...')
    contour_target_ff = []
    contour_dur = np.zeros(len(file_contour))
    len_contour = len(file_contour)
    # read contours into the var contour_target_ff
    for cc in range(len_contour):
        count_all += 1
        time_contour = file_contour[cc]['Time']
        freq_contour = file_contour[cc]['Freq']

        duration = time_contour[-1] - time_contour[0]
        contour_dur[cc] = duration
        duration_max = max([duration_max, duration])

        # linear interpolation
        time_contour_interp = np.arange(time_contour[0], time_contour[-1],
                                        time_reso)
        freq_contour_interp = np.interp(time_contour_interp, time_contour,
                                        freq_contour)

        contour_target_ff_cc = dict()
        contour_target_ff_cc['Time'] = time_contour_interp
        contour_target_ff_cc['Freq'] = freq_contour_interp

        contour_target_ff.append(contour_target_ff_cc)

    return contour_target_ff, count_all, duration_max


# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
#
#
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=10):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
#
#
# def calc_cepstrum(samples, sample_rate, win_size, step_size, fft_size, filt_num=64, low_freq=5000, high_freq=23500):
#     num_frame = int(np.floor((samples.shape[0]-win_size)/step_size))
#     fea_cepstrum_list = []
#     for ii in range(num_frame):
#         samp_frame = samples[ii*step_size:ii*step_size+win_size]  # windowing
#         samp_fft = np.zeros(fft_size)
#         samp_fft[:win_size] = samp_frame  # zero-padding
#
#         # samp_fft2 = butter_bandpass_filter(samp_fft, 5000, 23500, sample_rate, order=64)
#         # ceps = real_cepstrum(samp_fft, fft_size)
#         ceps = filtered_cepstrum(samp_fft, fft_size, sr=sample_rate,
#                                  num_fil=filt_num, low_freq=low_freq, high_freq=high_freq)
#         fea_cepstrum_list.append(ceps)
#     fea_cepstrum = np.vstack(fea_cepstrum_list)
#
#     return fea_cepstrum  # array (num_sample, dim_fea)
#
#
# def calc_cepstrum_parallel(samples, config):
#     sample_rate = config['sample_rate']
#     win_size = config['input_size']
#     step_size = int(win_size/2)
#     fft_size = win_size
#     filt_num = config['filt_num']  # 64
#     low_freq = config['low_freq']  # 5000
#     high_freq = config['high_freq']  # 23500
#
#     fea_cep_raw = calc_cepstrum(samples, sample_rate, win_size, step_size,
#                                  fft_size, filt_num=64, low_freq=5000,
#                                  high_freq=23500)
#     mid_point = int(fea_cep_raw.shape[0] / 2)
#     fea_cep_0 = fea_cep_raw[
#                 mid_point - 5:mid_point + 5]  # center of the contour
#
#     return fea_cep_0  # array (num_sample, dim_fea)
#
#
# def calc_energy(samples, sample_rate, win_size, step_size, fft_size, filt_num=64, low_freq=5000, high_freq=23500):
#     num_frame = int(np.floor((samples.shape[0]-win_size)/step_size))
#     fea_cepstrum_list = []
#     for ii in range(num_frame):
#         samp_frame = samples[ii*step_size:ii*step_size+win_size]  # windowing
#         samp_fft = np.zeros(fft_size)
#         samp_fft[:win_size] = samp_frame  # zero-padding
#
#         # samp_fft2 = butter_bandpass_filter(samp_fft, 5000, 23500, sample_rate, order=64)
#         # ceps = real_cepstrum(samp_fft, fft_size)
#         # ceps = filtered_cepstrum(samp_fft, fft_size, sr=sample_rate,
#         #                          num_fil=filt_num, low_freq=low_freq, high_freq=high_freq)
#         ceps = filtered_energy(samp_fft, fft_size, sr=sample_rate,
#                                  num_fil=filt_num, low_freq=low_freq, high_freq=high_freq)
#         fea_cepstrum_list.append(ceps)
#     fea_cepstrum = np.vstack(fea_cepstrum_list)
#
#     return fea_cepstrum  # array (num_sample, dim_fea)
#
#
#
#
# def real_cepstrum(x, fft_size):
#     r"""Compute the real cepstrum of a real sequence.
#     x : ndarray
#         Real sequence to compute real cepstrum of.
#     n : {None, int}, optional
#         Length of the Fourier transform.
#     Returns
#     """
#     if x.shape[0] != fft_size:
#         raise Exception('Sample length needs to be the same as FFT size.')
#     spectrum = np.fft.fft(x, n=fft_size)
#     ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
#
#     return ceps
#
#
# def filtered_cepstrum(x, fft_size, sr, num_fil, low_freq, high_freq):
#     r"""Compute the real cepstrum of a real sequence.
#     x : ndarray
#         Real sequence to compute real cepstrum of.
#     n : {None, int}, optional
#         Length of the Fourier transform.
#     Returns
#     """
#     if x.shape[0] != fft_size:
#         raise Exception('Sample length needs to be the same as FFT size.')
#     spectrum = np.fft.fft(x, n=fft_size)
#
#     win_half_size = (high_freq-low_freq)*1.0 / (num_fil+1)/sr*fft_size
#     win_full_size = int(2.0*win_half_size)
#     if win_full_size % 2 == 1:  # check if odd
#         win_full_size -= 1  # make it even
#
#     win_half_size_rounded = win_full_size/2
#     win_half = np.arange(win_half_size_rounded)/win_half_size_rounded
#     win_full = np.concatenate((win_half, 1.0-win_half))
#
#     ind_start = (win_half_size*np.arange(num_fil*1.0)).astype(int)
#     filtered_sepctrum = np.array([(np.sqrt((np.abs(spectrum[ii:ii+win_full_size])**2)*win_full)).sum() for ii in ind_start])
#     # ceps = np.fft.ifft(np.log(np.abs(filtered_sepctrum))).real  # original one
#     ceps = np.fft.ifft(np.log(filtered_sepctrum**2.)).real
#
#     return ceps
#
#
# def filtered_energy(x, fft_size, sr, num_fil, low_freq, high_freq):
#     r"""Compute the real cepstrum of a real sequence.
#     x : ndarray
#         Real sequence to compute real cepstrum of.
#     n : {None, int}, optional
#         Length of the Fourier transform.
#     Returns
#     """
#     if x.shape[0] != fft_size:
#         raise Exception('Sample length needs to be the same as FFT size.')
#     spectrum = np.fft.fft(x, n=fft_size)
#
#     win_half_size = (high_freq-low_freq)*1.0 / (num_fil+1)/sr*fft_size
#     win_full_size = int(2.0*win_half_size)
#     if win_full_size % 2 == 1:  # check if odd
#         win_full_size -= 1  # make it even
#
#     win_half_size_rounded = win_full_size/2
#     win_half = np.arange(win_half_size_rounded)/win_half_size_rounded
#     win_full = np.concatenate((win_half, 1.0-win_half))
#
#     ind_start = (win_half_size*np.arange(num_fil*1.0)).astype(int)
#     filtered_sepctrum = np.array([(np.sqrt((np.abs(spectrum[ii:ii+win_full_size])**2)*win_full)).sum() for ii in ind_start])
#     # ceps = np.fft.ifft(np.log(np.abs(filtered_sepctrum))).real
#
#     return filtered_sepctrum
#
#
# def power_law_calc(spectro_mat, nu1=2.0, nu2=1.0, gamma=2.0):
#     """
#     The preprocessing part of Generalized power-law (GPL) energy detector. It
#     preprocesses the spectrogram so that the tonal sounds are emphasized
#     whereas short-duration time impulse and narrow-band noises are diminished.
#
#     :param spectro_mat: spectrogram matrix that GPL is applied on
#     :param nu1: Weight parameter to the power of freq
#     :param nu2: Weight parameter to the power of time
#     :param gamma:
#     :return power_law_mat: spectrogram preprocessed with GPL
#     """
#     dim_f, dim_t = spectro_mat.shape
#     mu_k = [power_law_find_mu(spectro_mat[ff, :]) for ff in range(dim_f)]
#
#     mat0 = spectro_mat ** gamma - np.array(mu_k).reshape(dim_f, 1) * np.ones(
#         (1, dim_t))
#     mat_a_denom = [(np.sum(mat0[:, tt] ** 2.)) ** .5 for tt in range(dim_t)]
#     mat_a = mat0 / (np.ones((dim_f, 1)) * np.array(mat_a_denom).reshape(1, dim_t))
#     mat_b_denom = [(np.sum(mat0[ff, :] ** 2.)) ** .5 for ff in range(dim_f)]
#     mat_b = mat0 / (np.array(mat_b_denom).reshape(dim_f, 1) * np.ones((1, dim_t)))
#
#     mat_a = mat_a * (mat_a > 0)  # set negative values into zero
#     mat_b = mat_b * (mat_b > 0)
#     power_law_mat = (mat_a**nu1)*(mat_b**nu2)
#     # power_law_mat = (mat_a ** (2.0 * nu1)) * (mat_b ** (2.0 * nu2))
#     # PowerLawTFunc = np.sum((mat_a**nu1)*(mat_b**nu2), axis=0)
#
#     return power_law_mat
#
#
# def power_law_find_mu(spectro_target):
#     """
#     Function used in Generalized Power Law (GPL) to find the mu for each
#     frequency bin
#
#     :param spectro_target: input spectrogram
#     :return an array of mu where one for each frequency:
#     """
#     spec_sorted = np.sort(spectro_target)
#     spec_half_len = int(np.floor(spec_sorted.shape[0]*.5))
#     ind_j = np.argmin(spec_sorted[spec_half_len:spec_half_len*2] - spec_sorted[0:spec_half_len])
#     mu = np.mean(spec_sorted[ind_j:ind_j+spec_half_len])
#
#     return mu
#
#
# def bin_extract(bin_dir, sound_dir, species_name):
#     """
#     Extract whistle contour information from bin files
#     :param bin_dir: the folder that has the bin files
#     :param sound_dir: the folder that has the sound files
#     :param species_name: a dict that has the list of species names
#     :return file_contour_pair: a dict  with filename as the key. The value
#     consists of (i) species class label; (ii) list of [time & freq sequence]
#     :return bin_wav_pair: a dict with filename as the key to the sound files.
#     """
#     bin_wav_pair = dict()
#     file_contour_pair = dict()
#     # for species in [species_name[0]]:
#     for species in species_name:
#         print(species)
#         bin_file_list = glob.glob(os.path.join(bin_dir, species, '*.bin'))
#         bin_file_list.sort()
#         if len(bin_file_list) == 0:
#             print('Tonal files were not found. ')
#             sys.exit()
#         for bb in bin_file_list:
#             file_base = os.path.splitext(os.path.basename(bb))[0]
#             file_name = file_base + '.wav'
#             file_path = os.path.join(sound_dir, species, file_name)
#             if os.path.isfile(file_path):  # bin & wav match!
#                 print('Process ' + file_name)
#                 bin_wav_pair[bb] = file_path
#                 # Extract contours
#                 tonal0 = tonal(bb)
#                 contour_list = []
#                 while True:
#                     try:
#                         contour_list.append(tonal0.__next__())
#                     except StopIteration:
#                         print("End of the bin file of " + file_base + ".bin")
#                         break
#                 file_contour_pair[file_base] = [species, contour_list]
#     return file_contour_pair, bin_wav_pair
#
#
# def timestep_info(contour_all, timestep_file, percentile_list):
#     """
#     Measure the information of time steps in each whistle contours
#     :param contour_all: output from bin_extraction. It has all the info
#     contained in bin files
#     :param timestep_file: the output file that time step info is written to
#     :param percentile_list: the levels of percentiles we target at. E.g.
#     [0, 25, 50, 75, 100] has the quartiles as well as min and max.
#     """
#     contour_count = 0
#     contour_fixed_step = 0
#     f = open(timestep_file, "w")
#     for ff in contour_all.keys():
#         # print(ff)
#         f.write(ff+"\n")
#         file_contour = contour_all[ff][1]
#         for cc in range(len(file_contour)):
#             # print('Contour: '+str(cc))
#             f.write('Contour: '+str(cc)+' Length: '+str(len(file_contour[cc]['Time']))+'\t')
#             timestep_prctile = np.percentile(np.diff(file_contour[cc]['Time']),
#                                                      percentile_list)
#             contour_count += 1
#             if timestep_prctile[0] == timestep_prctile[4]: # max - min
#                 contour_fixed_step += 1
#             for ii in timestep_prctile.tolist():
#                 f.write(str(ii)+"\t")
#             f.write("\n")
#     f.close()
#     print("contour_count: "+str(contour_count))
#     print("contour_fixed_step: "+str(contour_fixed_step))
#
#
# def fea_cqt_powerlaw(samples, samplerate):
#     nu_1 = 3.0
#     nu_2 = 1.0
#     gamma = 1.0
#
#     # cqt
#     spectro_cqt = np.abs( librosa.cqt(samples, sr=samplerate, hop_length=1000,
#                                       n_bins=12 * 3 * 4,
#                                       bins_per_octave=12 * 4,
#                                       fmin=6000))
#     spectro_cqt_ud = np.flipud(spectro_cqt)
#     spectro_cqt_pl = power_law_calc(spectro_cqt_ud, nu_1, nu_2, gamma)
#     fea = ((spectro_cqt_pl - spectro_cqt_pl.min()) / (spectro_cqt_pl.max())).T
#
#     return fea
#
#
# def contour_repeat_score(contour_target_ff, len_contour, time_reso,
#                          overlap_time_thre=0.05, std_norm_thre=0.05):
#     contour_count = 0
#     contour_ind = np.ones(len_contour)
#     for cc1 in range(len_contour):
#         # print("cc1: " + str(cc1) + " length " + str(
#         #    contour_target_ff[cc1]['Time'].shape[0]))
#         for cc2 in range(cc1 + 1, len_contour):
#             a1 = contour_target_ff[cc1]['Time'][0]
#             b1 = contour_target_ff[cc1]['Time'][-1]
#             a2 = contour_target_ff[cc2]['Time'][0]
#             b2 = contour_target_ff[cc2]['Time'][-1]
#             if (b1 - a2 > 0) & (b2 - a1 > 0):  # there is overlap greater than 0.1 sec
#                 if a1 <= a2:
#                     time_start = a2
#                     seq_ind = 2
#                 else:
#                     time_start = a1
#                     seq_ind = 1
#                 if b1 <= b2:
#                     time_end = b1
#                 else:
#                     time_end = b2
#
#                 if time_end - time_start >= overlap_time_thre:
#                     if seq_ind == 1:
#                         ind_list = list(range(0, int(
#                             round((time_end - a1) / time_reso))))
#                         freq_seq_1 = [contour_target_ff[cc1]['Freq'][tt] for
#                                       tt in ind_list]
#                         freq_seq_2 = [contour_target_ff[cc2]['Freq'][
#                                           tt + int(round((time_start - a2) / time_reso))]
#                                       for tt in ind_list]
#                     else:  # seq_ind 2
#                         ind_list = list(range(0, int(
#                             round((time_end - a2) / time_reso))))
#                         freq_seq_2 = [contour_target_ff[cc2]['Freq'][tt] for
#                                       tt in ind_list]
#                         freq_seq_1 = [contour_target_ff[cc1]['Freq'][tt + int(round((time_start - a1) / time_reso))] for tt in ind_list]
#
#                     ratio_seq = np.array(freq_seq_1) / np.array(freq_seq_2)
#                     try:
#                         std_norm = ratio_seq.std() / ratio_seq.mean()
#                     except RuntimeWarning:
#                         print()
#
#                     if std_norm <= std_norm_thre:  # small freq one will lose a point
#                         if np.array(freq_seq_1).mean() <= np.array(
#                                 freq_seq_2).mean():
#                             contour_ind[cc2] -= 1
#                         else:
#                             contour_ind[cc1] -= 1
#
#                         # print('Duration steps: ' + str(int(
#                         #     round((time_end - time_start) / time_reso + 1.0))))
#                         # print('')
#                         contour_count += 1
#     print('Remaining contours: ' + str((contour_ind >= 1.0).sum())+' / '+str(len_contour))
#     return contour_ind, contour_count
#
#
# def fea_freq_label_extract(contour_target_ff, contour_ind, fea_list, label_list, species_label, count_long, duration_thre, transform):
#     contour_ind_list = np.where(contour_ind >= 1.0)[0].tolist()
#     fea_dict = dict()
#     for cc in contour_ind_list:
#         freq_contour_interp = contour_target_ff[cc]['Freq']
#         time_contour_interp = contour_target_ff[cc]['Time']
#         duration = time_contour_interp[-1] - time_contour_interp[0]
#
#         if duration >= duration_thre:
#             count_long += 1
#             if transform == "zscorederiv":
#                 fea_contour = ztransform(freq_contour_interp)
#             elif transform == "non_z":
#                 fea_contour = non_ztransform(freq_contour_interp)
#             elif transform == "rocca":
#                 fea_dict = rocca(freq_contour_interp, time_contour_interp)
#                 # fea_contour = np.array(list(fea_dict.values()))
#                 fea_contour = np.array([fea_dict[kk] for kk in sorted(fea_dict.keys())])
#             else:  # pass freq contour directly
#                 fea_contour = freq_contour_interp
#             fea_list.append(fea_contour)
#             label_list.append(species_label)
#
#     if transform == "rocca":
#         fea_name = sorted(fea_dict.keys())
#     else:
#         fea_name = None
#     return fea_list, label_list, count_long, fea_name
#
#
# def fea_context_base_extract(contour_target_ff, contour_ind, ff, species_name_this, species_label, bin_wav_pair,
#                                bin_dir_target, duration_thre, conf, fea='rocca_cep'):
#     '''
#     Extract features for each file contour_target_ff
#     :param contour_target_ff:
#     :param contour_ind:
#     :param ff:
#     :param species_name_this:
#     :param species_label:
#     :param bin_wav_pair:
#     :param bin_dir_target:
#     :param duration_thre:
#     :param conf:
#     :param fea:
#     :return:
#     '''
#     contour_ind_list = np.where(contour_ind >= 1.0)[0].tolist()
#     fea_dict = dict()
#     fea_rocca_list = []
#     fea_cep_list = []
#     label_list = []
#     count_long = 0
#     if (fea == 'cep') or (fea == 'rocca_cep'):
#         model_fea = load_fea_model(model_dir)
#     sample_list = []
#     sound_file = bin_wav_pair[os.path.join(bin_dir_target, species_name_this,
#                                            ff + '.bin')]
#     # # extract time of contours
#     # time_start = []
#     # time_stop = []
#     # for cc in contour_ind_list:
#     #     time_contour_interp = contour_target_ff[cc]['Time']
#     #     time_start.append(time_contour_interp[0])
#     #     time_stop.append(time_contour_interp[-1])
#     # df_contour = pd.DataFrame(list(zip(time_start, time_stop)), columns=['start_time', 'end_time'])
#     timestamp = []
#     for cc in contour_ind_list:
#         freq_contour_interp = contour_target_ff[cc]['Freq']
#         time_contour_interp = contour_target_ff[cc]['Time']
#         time_start = time_contour_interp[0]
#         time_stop = time_contour_interp[-1]
#         duration = time_stop - time_start
#         time_center = (time_stop + time_start)*.5
#
#         if duration >= duration_thre:
#             # print('cc: '+str(cc))
#             timestamp.append(time_center)
#             count_long += 1
#
#             # feature "rocca":
#             if fea == 'rocca' or fea == 'rocca_cep':
#                 fea_dict = rocca(freq_contour_interp, time_contour_interp)
#                 fea_rocca = np.array(
#                     [fea_dict[kk] for kk in sorted(fea_dict.keys())])
#
#                 t1 = (np.isinf(fea_rocca)).sum()
#                 if t1 > 0:
#                     print('Number of infinity is' + str(t1))
#                     print()
#             else:
#                 fea_rocca = None
#
#             # feature cep
#             if fea == 'cep' or fea == 'rocca_cep':
#                 samples, samplerate = librosa.load(sound_file,
#                                                    offset=time_contour_interp[0],
#                                                    duration=duration,
#                                                    sr=conf['sample_rate'],
#                                                    mono=True)
#             else:
#                 # fea_model_cep_flatten = []
#                 # fea_cep_0 = []
#                 samples = None
#
#             # Add the newly calculated features into a list
#             fea_rocca_list.append(fea_rocca)
#             # fea_cep_list.append(fea_model_cep_flatten)
#             sample_list.append(samples)
#             # fea_cep_list.append(fea_cep_0)
#             label_list.append(species_label)
#     print('Remaining contours after duration check: : ' +
#           str(count_long) + ' / ' + str(len(contour_ind_list)))
#
#     if fea == 'rocca' or fea == 'rocca_cep':
#         if len(fea_rocca_list) != 0:
#             fea_rocca = np.stack(fea_rocca_list)
#         else:
#             fea_rocca = None
#     else:
#         fea_rocca = None
#     if fea == 'cep' or fea == 'rocca_cep':
#         # if len(fea_cep_list) == not 0:
#         if len(sample_list) != 0:
#             # calculate cepstral coeff in parallel
#             pool_cep = mp.Pool(processes=4)
#             fea_cep_list  = pool_cep.starmap(calc_cepstrum_parallel,
#                                        zip(sample_list, repeat(conf)))
#             pool_cep.close()
#             pool_cep.join()
#
#             # convert cepstral coeff into feature vectors through pre-trained LSTM model
#             fea_cep = np.stack(fea_cep_list)
#             fea_cep = np.expand_dims(fea_cep, axis=3)
#             fea_cep = np.expand_dims(fea_cep, axis=4)
#             fea_model_cep = model_fea.predict(fea_cep)
#             # fea_model_cep_flatten = fea_model_cep1.flatten()
#         else:
#             fea_model_cep = None
#     else:
#         fea_model_cep = None
#     fea_name = sorted(fea_dict.keys())
#
#     return fea_rocca, fea_model_cep, label_list, count_long, fea_name, timestamp
#
#
# def fea_context_ae_extract(contour_target_ff, contour_ind, ff,
#                            species_name_this, species_label, bin_wav_pair,
#                            bin_dir_target, duration_thre, conf, encoder, min_fea,
#                            max_fea, dur_max_ind, fea):
#     '''
#     extract context features based on autoencoder
#     :param contour_target_ff:
#     :param contour_ind:
#     :param ff:
#     :param species_name_this:
#     :param species_label:
#     :param bin_wav_pair:
#     :param bin_dir_target:
#     :param duration_thre:
#     :param conf:
#     :param fea:
#     :return:
#     '''
#     contour_ind_list = np.where(contour_ind >= 1.0)[0].tolist()
#     fea_dict = dict()
#     # fea_rocca_list = []
#     # fea_cep_list = []
#     label_list = []
#     count_long = 0
#     sample_list = []
#     sound_file = bin_wav_pair[os.path.join(bin_dir_target, species_name_this,
#                                            ff + '.bin')]
#     # # extract time of contours
#     timestamp = []
#     contour_list = []
#     for cc in contour_ind_list:
#         freq_contour_interp = contour_target_ff[cc]['Freq']
#         time_contour_interp = contour_target_ff[cc]['Time']
#         time_start = time_contour_interp[0]
#         time_stop = time_contour_interp[-1]
#         duration = time_stop - time_start
#         time_center = (time_stop + time_start)*.5
#
#         if duration >= duration_thre:
#             # print('cc: '+str(cc))
#             timestamp.append(time_center)
#             contour_list.append(freq_contour_interp)
#             count_long += 1
#     print('Remaining contours after duration check: : ' +
#           str(count_long) + ' / ' + str(len(contour_ind_list)))
#
#     label_list = []
#     fea_name = []
#     if count_long >= 1:
#         max_minus_min_fea = max_fea - min_fea
#         if (fea == 'ae') or (fea == 'vae'):
#             fea_list_train_norm = []
#             for cc in range(len(contour_list)):
#                 fea_list_train_norm.append((contour_list[cc] - min_fea) / max_minus_min_fea)
#             fea_arr_train = pad_sequences(fea_list_train_norm,
#                                           maxlen=dur_max_ind,
#                                           dtype='float')
#             fea_arr_train_3d = np.expand_dims(fea_arr_train, axis=2)
#             if fea == 'vae':
#                 encoded_fea, _, _ = encoder.predict(fea_arr_train_3d)
#             else:
#                 encoded_fea = encoder.predict(fea_arr_train_3d)
#
#         for ll in range(encoded_fea.shape[0]):
#             label_list.append(species_label)
#         for ee in range(encoded_fea.shape[1]):
#             fea_name.append('ae'+str(ee+1))
#     else:
#         encoded_fea = None
#
#     return encoded_fea, label_list, count_long, fea_name, timestamp
#
#
# def fea_context(df_ff, df_ff_fea, timestamp, conf, fea_name):
#     context_win = conf['context_win']
#     df_ff['Time'] = np.array(timestamp)  # add center time into meta info dataframe
#     # select contours within context windows
#     df_ff_tot = pd.concat([df_ff, df_ff_fea], axis=1)
#
#     fea_context_list = []
#     fea_rocca_mean = [ff+'_mean' for ff in fea_name[0:]]
#     fea_rocca_std = [ff + '_std' for ff in fea_name[0:]]
#     fea_name_context = ['num_contour', 'num_contour_half', 'num_contour_quarter',
#                         'dur_mean', 'dur_std', 'dur_5',
#                         'dur_25', 'dur_50', 'dur_75', 'dur_95',
#                         'time_diff_mean', 'time_diff_std', 'time_diff_5',
#                         'time_diff_25', 'time_diff_50', 'time_diff_75',
#                         'time_diff_95',
#                         'dur_wei_mean', 'dur_wei_std', 'dur_wei_5',
#                         'dur_wei_25', 'dur_wei_50', 'dur_wei_75', 'dur_wei_95']\
#                        + fea_rocca_mean + fea_rocca_std
#     for _, row in df_ff_tot.iterrows():
#         fea_context = []
#         df_context = df_ff_tot.loc[np.abs(df_ff_tot['Time']-row['Time'])<context_win]
#
#         # number of contours within context window
#         fea_context.append(df_context.shape[0])
#         # number of contours within a half context window
#         fea_context.append((df_ff_tot.loc[np.abs(df_ff_tot['Time']-row['Time'])<context_win*.5]).shape[0] / df_context.shape[0])
#         # number of contours within a quarter context window
#         fea_context.append((df_ff_tot.loc[np.abs(df_ff_tot['Time']-row['Time'])<context_win*.25]).shape[0] / df_context.shape[0])
#
#         # duration
#         fea_dur = np.array(df_context['dur'])
#         fea_dur_percent = np.percentile(fea_dur, [5, 25, 50, 75, 95])
#         fea_context.append(fea_dur.mean())
#         fea_context.append(fea_dur.std())
#         for ff in fea_dur_percent:
#             fea_context.append(ff)
#
#         # time diff
#         fea_time_diff = np.array((df_context['Time']-row['Time']).abs())
#         fea_time_diff_percent = np.percentile(fea_time_diff, [5, 25, 50, 75, 95])
#         fea_context.append(fea_time_diff.mean())
#         fea_context.append(fea_time_diff.std())
#         for ff in fea_time_diff_percent:
#             fea_context.append(ff)
#
#         # weighted duration w.r.t. time diff
#         fea_dur_weighted = np.array(df_context['dur'])*(1. - (np.array((df_context['Time']-row['Time']).abs())/conf['context_win']))
#         fea_dur_weighted_percent = np.percentile(fea_dur_weighted, [5, 25, 50, 75, 95])
#         fea_context.append(fea_dur_weighted.mean())
#         fea_context.append(fea_dur_weighted.std())
#         for ff in fea_dur_weighted_percent:
#             fea_context.append(ff)
#
#         # means of individual rocca features
#         # original mean
#         fea_rocca_mean = (np.array(df_context.iloc[:,4:])).mean(axis=0)
#         for ff in fea_rocca_mean:
#             fea_context.append(ff)
#         # weighted mean
#         # fea_rocca_weight = np.array(df_context.iloc[:, 4:]).T*(1. - (np.array((df_context['Time']-row['Time']).abs())/conf['context_win']))
#         # fea_rocca_mean_weight = fea_rocca_weight.mean(axis=1)
#         # for ff in fea_rocca_mean_weight:
#         #     fea_context.append(ff)
#
#         # std of individual rocca features
#         fea_rocca_std = (np.array(df_context.iloc[:,4:])).mean(axis=0)
#         for ff in fea_rocca_std:
#             fea_context.append(ff)
#         # fea_rocca_std_weight = fea_rocca_weight.std(axis=1)
#         # for ff in fea_rocca_std_weight:
#         #     fea_context.append(ff)
#
#         fea_context_list.append(fea_context)
#     df_context_fea = pd.DataFrame(fea_context_list, columns=fea_name_context)
#
#     return df_context_fea
#
#
# def fea_context_ae(df_ff, df_ff_fea, timestamp, conf, fea_name):
#     context_win = conf['context_win']
#     df_ff['Time'] = np.array(timestamp)  # add center time into meta info dataframe
#     # select contours within context windows
#     df_ff_tot = pd.concat([df_ff, df_ff_fea], axis=1)
#
#     fea_name_context = []
#     for aa in range(conf['ae_latent_dim']):
#         fea_name_context.append('ae_mean'+str(aa+1))
#     for aa in range(conf['ae_latent_dim']):
#         fea_name_context.append('ae_std'+str(aa+1))
#
#     fea_context_list = []
#     for _, row in df_ff_tot.iterrows():
#         fea_context = []
#         df_context = df_ff_tot.loc[np.abs(df_ff_tot['Time']-row['Time'])<context_win]
#
#         # ae
#         fea_ae = np.array((df_context.iloc[:, 4:]))
#         fea_ae_mean = fea_ae.mean(axis=0)
#         fea_ae_std = fea_ae.std(axis=0)
#         for ff in fea_ae_mean:
#             fea_context.append(ff)
#         for ff in fea_ae_std:
#             fea_context.append(ff)
#
#         fea_context_list.append(fea_context)
#     df_context_fea = pd.DataFrame(fea_context_list, columns=fea_name_context)
#
#     return df_context_fea
#
#
# def fea_freq_cep_label_extract(contour_target_ff, contour_ind, ff,
#                                species_name_this, species_label, bin_wav_pair,
#                                bin_dir_target, duration_thre,
#                                model_dir, conf, fea='rocca_cep'):
#     '''
#     :param contour_target_ff:
#     :param contour_ind:
#     :param ff:
#     :param species_name_this:
#     :param species_label:
#     :param bin_wav_pair:
#     :param bin_dir_target:
#     :param count_long:
#     :param duration_thre:
#     :param fea: 'rocca_cep', 'rocca' or 'cep'
#     :return:
#     '''
#     contour_ind_list = np.where(contour_ind >= 1.0)[0].tolist()
#     fea_dict = dict()
#     fea_rocca_list = []
#     fea_cep_list = []
#     label_list = []
#     count_long = 0
#     if fea != 'rocca':
#         model_fea = load_fea_model(model_dir)
#     sample_list = []
#     sound_file = bin_wav_pair[os.path.join(bin_dir_target,
#                                            species_name_this, ff + '.bin')]
#
#     for cc in contour_ind_list:
#         freq_contour_interp = contour_target_ff[cc]['Freq']
#         time_contour_interp = contour_target_ff[cc]['Time']
#         duration = time_contour_interp[-1] - time_contour_interp[0]
#
#         if duration >= duration_thre:
#             print('cc: '+str(cc))
#             count_long += 1
#
#             # feature "rocca":
#             if fea == 'rocca' or fea == 'rocca_cep':
#                 fea_dict = rocca(freq_contour_interp, time_contour_interp)
#                 fea_rocca = np.array(
#                     [fea_dict[kk] for kk in sorted(fea_dict.keys())])
#
#                 t1 = (np.isinf(fea_rocca)).sum()
#                 if t1 > 0:
#                     print('Number of infinity is' + str(t1))
#                     print()
#
#             else:
#                 fea_rocca = []
#
#             # feature cep
#             if fea == 'cep' or fea == 'rocca_cep':
#                 samples, samplerate = librosa.load(sound_file,
#                                                    offset=time_contour_interp[0],
#                                                    duration=duration,
#                                                    sr=conf['sample_rate'],
#                                                    mono=True)
#
#                 # conf['input_size'] = input_size
#                 # conf['sample_rate'] = samplerate
#                 # pool_cep = mp.Pool(processes=4)
#                 # results = pool_cep.starmap(fft_sample_to_spectro,
#                 #                            zip(SampList, repeat(config)))
#                 # # fea_cep_raw = calc_cepstrum(samples, samplerate, win_size=input_size, step_size=int(input_size/2), fft_size=input_size, filt_num=conf['filt_num'],
#                 # #                             low_freq=conf['low_freq'], high_freq=conf['high_freq'])
#                 #
#                 # # fea_cep_1 = fea_cep_raw[:10]  # beginning of the contour
#                 # mid_point = int(fea_cep_raw.shape[0]/2)
#                 # fea_cep_2 = fea_cep_raw[mid_point-5:mid_point+5]  # center of the contour
#                 # # fea_cep_3 = fea_cep_raw[-10:]  # end of the contour
#                 #
#                 # # fea_cep = np.stack([fea_cep_1, fea_cep_2, fea_cep_3], axis=0)
#                 # fea_cep_0 = fea_cep_2
#                 # fea_cep = np.expand_dims(fea_cep, axis=0)
#                 # fea_cep = np.expand_dims(fea_cep, axis=3)
#                 # fea_cep = np.expand_dims(fea_cep, axis=4)
#
#                 # fea_model_cep1 = model_fea.predict(fea_cep)
#                 # fea_model_cep_flatten = fea_model_cep1.flatten()
#                 # fea_model_cep_flatten = fea_model_cep1
#             else:
#                 # fea_model_cep_flatten = []
#                 # fea_cep_0 = []
#                 samples = []
#
#             # Add the newly calculated features into a list
#             fea_rocca_list.append(fea_rocca)
#             # fea_cep_list.append(fea_model_cep_flatten)
#             sample_list.append(samples)
#             # fea_cep_list.append(fea_cep_0)
#             label_list.append(species_label)
#     print('Remaining contours after duration check: : ' +
#           str(count_long) + ' / ' + str(len(contour_ind_list)))
#
#     if fea == 'rocca' or fea == 'rocca_cep':
#         if len(fea_rocca_list) != 0:
#             fea_rocca = np.stack(fea_rocca_list)
#         else:
#             fea_rocca = []
#     else:
#         fea_rocca =  []
#     if fea == 'cep' or fea == 'rocca_cep':
#         # if len(fea_cep_list) == not 0:
#         if len(sample_list) != 0:
#             # calculate cepstral coeff in parallel
#             pool_cep = mp.Pool(processes=4)
#             fea_cep_list  = pool_cep.starmap(calc_cepstrum_parallel,
#                                        zip(sample_list, repeat(conf)))
#             pool_cep.close()
#             pool_cep.join()
#
#             # convert cepstral coeff into feature vectors through pre-trained LSTM model
#             fea_cep = np.stack(fea_cep_list)
#             fea_cep = np.expand_dims(fea_cep, axis=3)
#             fea_cep = np.expand_dims(fea_cep, axis=4)
#             fea_model_cep = model_fea.predict(fea_cep)
#             # fea_model_cep_flatten = fea_model_cep1.flatten()
#         else:
#             fea_model_cep = []
#     else:
#         fea_model_cep_flatten = []
#     fea_name = sorted(fea_dict.keys())
#
#     return fea_rocca, fea_model_cep, label_list, count_long, fea_name
#
#
# def fea_cep_label_extract(contour_target_ff, contour_ind, ff,
#                                species_name_this, species_label, bin_wav_pair,
#                                bin_dir_target, count_long, duration_thre):
#     contour_ind_list = np.where(contour_ind >= 1.0)[0].tolist()
#     fea_list = []
#     label_list = []
#     for cc in contour_ind_list:
#     # for cc in contour_ind_list[:10]:
#         freq_contour_interp = contour_target_ff[cc]['Freq']
#         time_contour_interp = contour_target_ff[cc]['Time']
#         duration = time_contour_interp[-1] - time_contour_interp[0]
#
#         if duration >= duration_thre:
#             print('Extracting features of '+str(cc)+':')
#             count_long += 1
#             sound_file = bin_wav_pair[
#                 os.path.join(bin_dir_target, species_name_this, ff + '.bin')]
#             samples, samplerate = librosa.load(sound_file,
#                                                offset=time_contour_interp[0],
#                                                duration=duration,
#                                                sr=192000, mono=True)
#             # cepstrum calculation
#             input_size = 4096  # 21.33 msec given sampling rate 192,000 Hz
#             fea_contour = calc_cepstrum(samples, samplerate, win_size=input_size, step_size=int(input_size/2), fft_size=input_size, filt_num=64, low_freq=5000., high_freq=23500)
#             # calc_cepstrum(samples, sample_rate, win_size, step_size, fft_size,
#             #              filt_num=64, low_freq=5000., high_freq=23500):
#             fea_list.append(fea_contour)
#             label_list.append(species_label)
#
#     return fea_list, label_list, count_long
#
#
# def fea_energy_label_extract(contour_target_ff, contour_ind, ff,
#                                species_name_this, species_label, bin_wav_pair,
#                                bin_dir_target, count_long, duration_thre):
#     contour_ind_list = np.where(contour_ind >= 1.0)[0].tolist()
#     fea_list = []
#     label_list = []
#     for cc in contour_ind_list:
#     # for cc in contour_ind_list[:10]:
#         freq_contour_interp = contour_target_ff[cc]['Freq']
#         time_contour_interp = contour_target_ff[cc]['Time']
#         duration = time_contour_interp[-1] - time_contour_interp[0]
#
#         if duration >= duration_thre:
#             print('Extracting features of '+str(cc)+':')
#             count_long += 1
#             sound_file = bin_wav_pair[
#                 os.path.join(bin_dir_target, species_name_this, ff + '.bin')]
#             samples, samplerate = librosa.load(sound_file,
#                                                offset=time_contour_interp[0],
#                                                duration=duration,
#                                                sr=192000, mono=True)
#             # cepstrum calculation
#             input_size = 4096  # 21.33 msec given sampling rate 192,000 Hz
#             fea_contour = calc_energy(samples, samplerate, win_size=input_size, step_size=int(input_size/2), fft_size=input_size, filt_num=64, low_freq=5000., high_freq=23500)
#             # calc_cepstrum(samples, sample_rate, win_size, step_size, fft_size,
#             #              filt_num=64, low_freq=5000., high_freq=23500):
#             fea_list.append(fea_contour)
#             label_list.append(species_label)
#
#     return fea_list, label_list, count_long
#
#
# def fea_label_generate(contour_target, bin_wav_pair, duration_thre,
#                        percentile_list, bin_dir_target, species_id, gap=0.0):
#     count_long = 0
#     count_all = 0
#     duration_list = []
#     fea_list = []
#     label_list = []
#     duration_max = 0
#     for ff in contour_target.keys():
#         print(ff)
#         file_contour = contour_target[ff][1]
#         # cc_lim = np.min([3, len(file_contour)]) # DEBUG
#         # for cc in range(cc_lim):
#         for cc in range(len(file_contour)):
#             time_contour = file_contour[cc]['Time']
#             timestep_prctile = np.percentile(np.diff(time_contour),
#                                              percentile_list)
#             duration = time_contour[-1] - time_contour[0]
#             duration_list.append(duration)
#             duration_max = max([duration_max, duration])
#             count_all += 1
#             if duration >= duration_thre:
#                 print("Contour: " + str(cc))
#                 count_long += 1
#                 sound_file = bin_wav_pair[
#                     os.path.join(bin_dir_target, contour_target[ff][0],
#                                  ff + '.bin')]
#                 samples, samplerate = librosa.load(sound_file,
#                                                    offset=time_contour[0]-gap,
#                                                    duration=duration+2.*gap,
#                                                    sr=96000,
#                                                    mono=True)
#                 fea_list.append(fea_cqt_powerlaw(samples, samplerate))
#                 label_list.append(species_id[contour_target[ff][0]])
#
#     return fea_list, label_list, duration_max, count_long, count_all
#
#
# def fea_label_generate_no_harmonics(contour_target, bin_wav_pair, bin_dir_target,
#                                     species_id, conf):
#     '''
#     fea_cepstrum_label_extract
#     :param contour_target:
#     :param bin_wav_pair:
#     :param bin_dir_target:
#     :param species_id:
#     :param conf:
#     :return:
#     '''
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     data_list = []
#     df_ff_fea_list = []
#     df_ff_list = []
#     # iterate over files
#     # for ff in contour_target.keys():
#     # for ff in [next(iter(contour_target))]:  # retrieve only the first key
#     # for ff in sorted(list(contour_target.keys()))[2:4]:
#     for ff in sorted(list(contour_target.keys())):
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         if len_contour >= 1:
#             # Retrieve contours
#             contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, conf['time_reso'])
#
#             # Exploratory analysis
#             # pairwise comparison: check time overlap & harmonic relationship
#             print('Remove contours from one harmonic structure...')
#             contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, conf['time_reso'])
#
#             # extracty features on the selected contours with longer duration than the threshold
#             print('Extract features on the selected contours...')
#             species_name_this = contour_target[ff][0]
#             species_label = species_id[contour_target[ff][0]]
#             fea_list_ff, label_list_ff, count_long = fea_cepstrum_label_extract\
#                 (contour_target_ff, contour_ind, ff, species_name_this,
#                  species_label, bin_wav_pair, bin_dir_target, count_long,
#                  conf['duration_thre'])
#             # make a dataframe to hold ff, cc, fea (n x 64), label
#             col_fea = []
#             col_ff = []
#             col_cc = []
#             col_label = []
#             for cc in range(len(fea_list_ff)):
#                 col_fea.append(fea_list_ff[cc])
#                 for tt in range(fea_list_ff[cc].shape[0]):
#                     col_ff.append(ff)
#                     col_cc.append(cc)
#                     col_label.append(label_list_ff[cc])
#                 # for tt in range(len(fea_list_ff[cc])):
#         else:
#             print('File is empty and has no contours.')
#
#         if len(col_fea) >= 1:
#             col_fea_arr = np.vstack(col_fea)
#         else:
#             col_fea_arr = col_fea
#         df_ff_fea = pd.DataFrame(col_fea_arr)
#         df_ff_fea_list.append(df_ff_fea)
#
#         df_ff = pd.DataFrame(list(zip(col_ff, col_cc, col_label)), columns=['file', 'contour_id', 'label'])
#         df_ff_list.append(df_ff)
#
#     df_tot_fea = pd.concat(df_ff_fea_list, axis=0)
#     df_tot_fea = df_tot_fea.reset_index(drop=True)
#     df_tot_meta = pd.concat(df_ff_list, axis=0)
#     df_tot_meta = df_tot_meta.reset_index(drop=True)
#     df_tot = pd.concat([df_tot_meta, df_tot_fea], axis=1)
#
#     return df_tot, duration_max, count_long, count_all
#
#
# def fea_label_generate_no_harmonics_v2(contour_target, bin_wav_pair, bin_dir_target,
#                                     species_id, conf):
#     '''
#     frame-based features: cepstrum for audio of a fixed length
#     cepstrum: fea_cepstrum_label_extract
#     :param contour_target:
#     :param bin_wav_pair:
#     :param bin_dir_target:
#     :param species_id:
#     :param conf:
#     :return:
#     '''
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     data_list = []
#     df_ff_fea_list = []
#     df_ff_list = []
#     # iterate over files
#     # for ff in contour_target.keys():
#     # for ff in [next(iter(contour_target))]:  # retrieve only the first key
#     # for ff in sorted(list(contour_target.keys()))[2:4]:
#     for ff in sorted(list(contour_target.keys())):
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         if len_contour >= 1:
#             # Retrieve contours
#             contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, conf['time_reso'])
#
#             # Exploratory analysis
#             # pairwise comparison: check time overlap & harmonic relationship
#             print('Remove contours from one harmonic structure...')
#             contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, conf['time_reso'])
#
#             # extracty features on the selected contours with longer duration than the threshold
#             print('Extract features on the selected contours...')
#             species_name_this = contour_target[ff][0]
#             species_label = species_id[contour_target[ff][0]]
#             fea_list_ff, label_list_ff, count_long = fea_cep_label_extract\
#                 (contour_target_ff, contour_ind, ff, species_name_this,
#                  species_label, bin_wav_pair, bin_dir_target, count_long,
#                  conf['duration_thre'])
#             # make a dataframe to hold ff, cc, fea (n x 64), label
#             col_fea = []
#             col_ff = []
#             col_cc = []
#             col_tt = []
#             col_label = []
#             for cc in range(len(fea_list_ff)):
#                 col_fea.append(fea_list_ff[cc])
#                 for tt in range(fea_list_ff[cc].shape[0]):
#                     col_ff.append(ff)
#                     col_cc.append(cc)
#                     col_tt.append(tt)
#                     col_label.append(label_list_ff[cc])
#                 # for tt in range(len(fea_list_ff[cc])):
#         else:
#             print('File is empty and has no contours.')
#
#         if len(col_fea) >= 1:
#             col_fea_arr = np.vstack(col_fea)
#         else:
#             col_fea_arr = col_fea
#         df_ff_fea = pd.DataFrame(col_fea_arr)
#         df_ff_fea_list.append(df_ff_fea)
#
#         df_ff = pd.DataFrame(list(zip(col_ff, col_cc, col_tt, col_label)),
#                              columns=['file', 'contour_id', 'time_id', 'label'])
#         df_ff_list.append(df_ff)
#
#     df_tot_fea = pd.concat(df_ff_fea_list, axis=0)
#     df_tot_fea = df_tot_fea.reset_index(drop=True)
#     df_tot_meta = pd.concat(df_ff_list, axis=0)
#     df_tot_meta = df_tot_meta.reset_index(drop=True)
#     df_tot = pd.concat([df_tot_meta, df_tot_fea], axis=1)
#
#     return df_tot, duration_max, count_long, count_all
#
#
# def fea_label_generate_no_harmonics_v3(contour_target, bin_wav_pair, bin_dir_target,
#                                     species_id, conf):
#     '''
#     fraame-based features: filter-bank energy for audio of a fixed length
#     filter-bank energy
#     :param contour_target:
#     :param bin_wav_pair:
#     :param bin_dir_target:
#     :param species_id:
#     :param conf:
#     :return:
#     '''
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     data_list = []
#     df_ff_fea_list = []
#     df_ff_list = []
#     # iterate over files
#     # for ff in contour_target.keys():
#     # for ff in [next(iter(contour_target))]:  # retrieve only the first key
#     # for ff in sorted(list(contour_target.keys()))[2:4]:
#     for ff in sorted(list(contour_target.keys())):
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         if len_contour >= 1:
#             # Retrieve contours
#             contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, conf['time_reso'])
#
#             # Exploratory analysis
#             # pairwise comparison: check time overlap & harmonic relationship
#             print('Remove contours from one harmonic structure...')
#             contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, conf['time_reso'])
#
#             # extracty features on the selected contours with longer duration than the threshold
#             print('Extract features on the selected contours...')
#             species_name_this = contour_target[ff][0]
#             species_label = species_id[contour_target[ff][0]]
#             fea_list_ff, label_list_ff, count_long = fea_energy_label_extract\
#                 (contour_target_ff, contour_ind, ff, species_name_this,
#                  species_label, bin_wav_pair, bin_dir_target, count_long,
#                  conf['duration_thre'])
#             # make a dataframe to hold ff, cc, fea (n x 64), label
#             col_fea = []
#             col_ff = []
#             col_cc = []
#             col_tt = []
#             col_label = []
#             for cc in range(len(fea_list_ff)):
#                 col_fea.append(fea_list_ff[cc])
#                 for tt in range(fea_list_ff[cc].shape[0]):
#                     col_ff.append(ff)
#                     col_cc.append(cc)
#                     col_tt.append(tt)
#                     col_label.append(label_list_ff[cc])
#                 # for tt in range(len(fea_list_ff[cc])):
#         else:
#             print('File is empty and has no contours.')
#
#         if len(col_fea) >= 1:
#             col_fea_arr = np.vstack(col_fea)
#         else:
#             col_fea_arr = col_fea
#         df_ff_fea = pd.DataFrame(col_fea_arr)
#         df_ff_fea_list.append(df_ff_fea)
#
#         df_ff = pd.DataFrame(list(zip(col_ff, col_cc, col_tt, col_label)),
#                              columns=['file', 'contour_id', 'time_id', 'label'])
#         df_ff_list.append(df_ff)
#
#     df_tot_fea = pd.concat(df_ff_fea_list, axis=0)
#     df_tot_fea = df_tot_fea.reset_index(drop=True)
#     df_tot_meta = pd.concat(df_ff_list, axis=0)
#     df_tot_meta = df_tot_meta.reset_index(drop=True)
#     df_tot = pd.concat([df_tot_meta, df_tot_fea], axis=1)
#
#     return df_tot, duration_max, count_long, count_all
#
#
# # classificatin using only frequency sequences
# def freq_seq_label_generate(contour_target, conf, species_id):
#     """
#     retrieve the whistle contour, the frequency sequences
#     feature extraction, such as z-transform can be applied
#
#     :param contour_target:
#     :param conf:
#     :param species_id:
#     :return:
#     """
#     time_reso = conf['time_reso']
#     duration_thre = conf["duration_thre"]
#     transform = conf['transform']
#
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     fea_list = []
#     label_list = []
#     for ff in contour_target.keys():
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         # Retrieve contours
#         contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, time_reso)
#
#         # Exploratory analysis
#         # pairwise comparison: check time overlap & harmonic relationship
#         print('Remove contours from one harmonic structure...')
#         contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, time_reso)
#
#         # extracty features on the selected contours with longer duration than the threshold
#         print('Extract features on the selected contours...')
#         species_label = species_id[contour_target[ff][0]]
#         fea_list, label_list, count_long, fea_name = fea_freq_label_extract(contour_target_ff, contour_ind, fea_list,
#                                label_list, species_label,
#                                count_long, duration_thre, transform)
#
#
#         # for cc in range(len(file_contour)):
#         #     time_contour = file_contour[cc]['Time']
#         #     freq_contour = file_contour[cc]['Freq']
#         #     print("Average time step is: "+str((time_contour[-1]-time_contour[0])/len(time_contour)))
#         #     if len(file_contour[cc]['Time']) != len(file_contour[cc]['Freq']):
#         #         print("Time and freq sequences are of unequal lengths.")
#         #         print("Time length: "+str(len(file_contour[cc]['Time'])))
#         #         print("Freq length: " + str(len(file_contour[cc]['Freq'])))
#         #         print("")
#         #
#         #     duration = time_contour[-1] - time_contour[0]
#         #     duration_max = max([duration_max, duration])
#         #
#         #     count_all += 1
#         #     if duration >= duration_thre:
#         #         print("Contour: " + str(cc))
#         #         count_long += 1
#         #
#         #         time_contour_2 = np.arange(time_contour[0], time_contour[-1], time_reso)
#         #         freq_contour_interp = np.interp(time_contour_2, time_contour, freq_contour)
#         #
#         #         if transform is "zscorederiv":
#         #             fea_contour = ztransform(freq_contour_interp)
#         #         else:
#         #             fea_contour = freq_contour_interp
#         #
#         #         fea_list.append(fea_contour)
#         #         label_list.append(species_id[contour_target[ff][0]])
#
#     return fea_list, label_list, duration_max, count_long, count_all
#
#
# def freq_seq_label_generate_no_harmonics(contour_target, conf, species_id):
#     """
#     retrieve the whistle contour, the frequency sequences
#     feature extraction, such as z-transform can be applied
#
#     :param contour_target:
#     :param conf:
#     :param species_id:
#     :param selected_id: an array consisting of contour ids
#     :return:
#     """
#     time_reso = conf['time_reso']
#     duration_thre = conf["duration_thre"]
#     transform = conf['transform']
#
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     fea_list = []
#     label_list = []
#     # iterate over files
#     for ff in contour_target.keys():
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         # Retrieve contours
#         contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, time_reso)
#
#         # Exploratory analysis
#         # pairwise comparison: check time overlap & harmonic relationship
#         print('Remove contours from one harmonic structure...')
#         contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, time_reso, overlap_time_thre=10.0)
#
#         # extracty features on the selected contours with longer duration than the threshold
#         print('Extract features on the selected contours...')
#         # print(contour_target[ff])
#         species_label = species_id[contour_target[ff][0]]
#         fea_list, label_list, count_long, fea_name = fea_freq_label_extract(contour_target_ff, contour_ind, fea_list,
#                                label_list, species_label,
#                                count_long, duration_thre, transform)
#     return fea_list, label_list, duration_max, count_long, count_all, fea_name
#
#
# def fea_freq_cep_label_generate_no_harmonics(contour_target, bin_wav_pair,
#                                              bin_dir_target, model_dir, conf,
#                                              species_id):
#     '''
#     feature: rocca + cepstrum
#     for each contour, both frame-based features, such as cepstrum or energy,
#     and contour-based rocca features are extracted.
#     Frame-based features can be either (i) applied directly or (ii) extracted
#     by a pre-trained LSTM model
#     :return:
#     '''
#     time_reso = conf['time_reso']
#     duration_thre = conf["duration_thre"]
#     transform = conf['transform']
#
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     fea_rocca_list = []
#     fea_cep_list = []
#     label_list = []
#     # f = open(os.path.join(bin_dir_target,'contour_count.txt'), 'w+')
#     # iterate over files
#     # for ff in contour_target.keys():
#     for ff in sorted(contour_target.keys()):
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         # Retrieve contours
#         contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, time_reso)
#
#         # Exploratory analysis
#         # pairwise comparison: check time overlap & harmonic relationship
#         print('Remove contours from one harmonic structure...')
#         contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, time_reso)
#
#         # extracty features on the selected contours with longer duration than the threshold
#         print('Extract features on the selected contours...')
#         # print(contour_target[ff])
#         species_name_this = contour_target[ff][0]
#         species_label = species_id[contour_target[ff][0]]
#         fea_rocca_file, fea_cep_file, label_file, count_long_file, fea_name = \
#             fea_freq_cep_label_extract(contour_target_ff, contour_ind,
#                                ff, species_name_this, species_label,
#                                        bin_wav_pair, bin_dir_target,
#                                        duration_thre, model_dir,
#                                        conf, fea=transform )
#         count_long  += count_long_file
#         # with redirect_stdout(f):
#         #     print(ff+': '+contour_target[ff][0]+' '+str(count_long_file))
#
#         if len(label_file) != 0:
#             fea_rocca_list.append(fea_rocca_file)
#             fea_cep_list.append(fea_cep_file)
#             label_list = label_list + label_file
#     # f.close()
#
#     if transform == 'rocca' or transform == 'rocca_cep':
#         if len(fea_rocca_list) == 0:
#             fea_rocca = []
#         else:
#             fea_rocca = np.vstack(fea_rocca_list)
#     else:
#         fea_rocca = []
#     if transform == 'cep' or transform == 'rocca_cep':
#         if len(fea_cep_list) == 0:
#             fea_cep = 0
#         else:
#             fea_cep = np.vstack(fea_cep_list)
#     else:
#         fea_cep = []
#
#     return fea_rocca, fea_cep, label_list, duration_max, count_long, count_all, fea_name
#
#
# def fea_context_base_generate(contour_target, bin_wav_pair, bin_dir_target, conf):
#     time_reso = conf['time_reso']
#     duration_thre = conf["duration_thre"]
#     transform = conf['transform']
#     species_id = conf['species_id']
#
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     data_list = []
#     df_ff_fea_list = []
#     df_ff_list = []
#     # iterate over files
#     for ff in sorted(list(contour_target.keys())):
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         if len_contour >= 1:
#             # Retrieve contours
#             contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, conf['time_reso'])
#
#             # Exploratory analysis
#             # pairwise comparison: check time overlap & harmonic relationship
#             print('Remove contours from one harmonic structure...')
#             # contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, conf['time_reso'])
#             # disble it by using a large overlap_time_thre
#             contour_ind, contour_count = contour_repeat_score(
#                 contour_target_ff, len_contour, conf['time_reso'], overlap_time_thre=10.0)
#
#             # extracty features on the selected contours with longer duration than the threshold
#             print('Extract features on the selected contours...')
#             species_name_this = contour_target[ff][0]
#             species_label = species_id[contour_target[ff][0]]
#             # fea_list_ff, label_list_ff, count_long = fea_context_extract\
#             #     (contour_target_ff, contour_ind, ff, species_name_this,
#             #      species_label, bin_wav_pair, bin_dir_target, count_long,
#             #      conf['duration_thre'])
#             fea_rocca_file, fea_cep_file, label_file, count_long_file, \
#             fea_name, timestamp = \
#                 fea_context_base_extract(contour_target_ff, contour_ind, ff,
#                                          species_name_this, species_label,
#                                          bin_wav_pair, bin_dir_target,
#                                          duration_thre, conf, fea=transform)
#             count_long += count_long_file
#             if fea_rocca_file is not None:
#                 # make a dataframe to hold ff, cc, fea (n x 64), label
#                 col_fea = []
#                 col_label = []
#                 col_ff = []
#                 col_cc = []
#                 for cc in range(fea_rocca_file.shape[0]):
#                     col_fea.append(fea_rocca_file[cc, :])
#                     col_ff.append(ff)
#                     col_cc.append(cc)
#                     col_label.append(label_file[cc])
#                 # meta information
#                 df_ff = pd.DataFrame(list(zip(col_ff, col_cc, col_label)),
#                                      columns=['file', 'contour_id', 'label'])
#                 df_ff_list.append(df_ff)
#
#                 # features
#                 # col_fea_tot = np.hstack((col_fea_context, col_fea))
#                 col_fea_arr = np.vstack(col_fea)
#                 df_ff_fea = pd.DataFrame(col_fea_arr, columns=fea_name)
#                 #
#
#                 # context features
#                 df_ff_fea_context = fea_context(df_ff, df_ff_fea,
#                                                        timestamp, conf,
#                                                         fea_name)
#                 # col_fea_context = []
#                 # for cc in range(fea_context_file.shape[0]):
#                 #     col_fea_context.append(fea_context_file[cc, :])
#                 df_ff_fea_list.append(pd.concat([df_ff_fea, df_ff_fea_context], axis=1))
#
#         else:
#             print('File is empty and has no contours.')
#
#     df_tot_fea = pd.concat(df_ff_fea_list, axis=0)
#     df_tot_fea = df_tot_fea.reset_index(drop=True)
#     df_tot_meta = pd.concat(df_ff_list, axis=0)
#     df_tot_meta = df_tot_meta.reset_index(drop=True)
#     df_tot = pd.concat([df_tot_meta, df_tot_fea], axis=1)
#
#     return df_tot, duration_max, count_long, count_all
#
#
# def fea_context_ae_generate(contour_target, bin_wav_pair, bin_dir_target,
#                             conf, encoder, min_fea, max_fea, dur_max_ind,
#                             fea):
#     time_reso = conf['time_reso']
#     duration_thre = conf["duration_thre"]
#     transform = conf['transform']
#     species_id = conf['species_id']
#
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     data_list = []
#     df_ff_fea_list = []
#     df_ff_list = []
#     # iterate over files
#     for ff in sorted(list(contour_target.keys())):
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         if len_contour >= 1:
#             # Retrieve contours
#             contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, conf['time_reso'])
#
#             # Exploratory analysis
#             # pairwise comparison: check time overlap & harmonic relationship
#             print('Remove contours from one harmonic structure...')
#             # contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, conf['time_reso'])
#             # disble it by using a large overlap_time_thre
#             contour_ind, contour_count = contour_repeat_score(
#                 contour_target_ff, len_contour, conf['time_reso'], overlap_time_thre=10.0)
#
#             # extracty features on the selected contours with longer duration than the threshold
#             print('Extract features on the selected contours...')
#             species_name_this = contour_target[ff][0]
#             species_label = species_id[contour_target[ff][0]]
#             fea_ae_file, label_file, count_long_file, fea_name, timestamp = \
#                 fea_context_ae_extract(contour_target_ff, contour_ind, ff,
#                                          species_name_this, species_label,
#                                          bin_wav_pair, bin_dir_target,
#                                          duration_thre, conf, encoder,
#                                        min_fea, max_fea, dur_max_ind, fea)
#
#             count_long += count_long_file
#             if fea_ae_file is not None:
#                 # make a dataframe to hold ff, cc, fea (n x 64), label
#                 col_fea = []
#                 col_label = []
#                 col_ff = []
#                 col_cc = []
#                 for cc in range(fea_ae_file.shape[0]):
#                     col_fea.append(fea_ae_file[cc, :])
#                     col_ff.append(ff)
#                     col_cc.append(cc)
#                     col_label.append(label_file[cc])
#                 # meta information
#                 df_ff = pd.DataFrame(list(zip(col_ff, col_cc, col_label)),
#                                      columns=['file', 'contour_id', 'label'])
#                 df_ff_list.append(df_ff)
#
#                 # individual features
#                 col_fea_arr = np.vstack(col_fea)
#                 df_ff_fea = pd.DataFrame(col_fea_arr, columns=fea_name)
#
#                 # context features
#                 df_ff_fea_context = fea_context_ae(df_ff, df_ff_fea,
#                                                        timestamp, conf,
#                                                         fea_name)
#                 df_ff_fea_list.append(pd.concat([df_ff_fea, df_ff_fea_context], axis=1))
#
#         else:
#             print('File is empty and has no contours.')
#
#     df_tot_fea = pd.concat(df_ff_fea_list, axis=0)
#     df_tot_fea = df_tot_fea.reset_index(drop=True)
#     df_tot_meta = pd.concat(df_ff_list, axis=0)
#     df_tot_meta = df_tot_meta.reset_index(drop=True)
#     df_tot = pd.concat([df_tot_meta, df_tot_fea], axis=1)
#
#     return df_tot, duration_max, count_long, count_all
#
#
# def freq_autoencoder_generate(contour_target, bin_wav_pair, bin_dir_target, conf):
#     time_reso = conf['time_reso']
#     duration_thre = conf["duration_thre"]
#     transform = conf['transform']
#     species_id = conf['species_id']
#
#     duration_max = 0
#     count_all = 0
#     count_long = 0
#     data_list = []
#     df_ff_fea_list = []
#     df_ff_list = []
#     # iterate over files
#     for ff in sorted(list(contour_target.keys())):
#         print(ff)
#         file_contour = contour_target[ff][1]
#         len_contour = len(file_contour)
#
#         if len_contour >= 1:
#             # Retrieve contours
#             contour_target_ff, count_all, duration_max = contour_retrieve(file_contour, count_all, duration_max, conf['time_reso'])
#
#             # Exploratory analysis
#             # pairwise comparison: check time overlap & harmonic relationship
#             print('Remove contours from one harmonic structure...')
#             # contour_ind, contour_count = contour_repeat_score(contour_target_ff, len_contour, conf['time_reso'])
#             # disble it by using a large overlap_time_thre
#             contour_ind, contour_count = contour_repeat_score(
#                 contour_target_ff, len_contour, conf['time_reso'], overlap_time_thre=10.0)
#
#             # extracty features on the selected contours with longer duration than the threshold
#             print('Extract features on the selected contours...')
#             species_name_this = contour_target[ff][0]
#             species_label = species_id[contour_target[ff][0]]
#             fea_rocca_file, fea_cep_file, label_file, count_long_file, \
#             fea_name, timestamp = \
#                 fea_context_base_extract(contour_target_ff, contour_ind, ff,
#                                          species_name_this, species_label,
#                                          bin_wav_pair, bin_dir_target,
#                                          duration_thre, conf, fea=transform)
#             count_long += count_long_file
#
#
#     return df_tot, duration_max, count_long, count_all
#
#
# def fea_label_shuffle(fea_list, label_list):
#     idx = [ii for ii in range(len(label_list))]
#     random.shuffle(idx)
#     # feature = [fea_list[tt] for tt in idx]
#     feature = [fea_list[tt] for tt in idx]
#     feature = np.array(feature)
#     # target = to_categorical(label_list)[idx, :]
#     # target = label_list[idx, :]
#     target = [label_list[tt] for tt in idx]
#     target = np.array(target)
#     return feature, target
#
#
# def freq_label_shuffle(fea_list, label_list):
#     idx = [ii for ii in range(len(label_list))]
#     random.shuffle(idx)
#     # feature = [fea_list[tt] for tt in idx]
#     feature = [fea_list[tt] for tt in idx]
#     target = to_categorical(label_list)[idx, :]
#
#     return feature, target
#
#
# def save_fea(pkl_path, fea_list, label_list, dur_max, count_long, count_all):
#     # pkl_path = os.path.join(feature_dir, 'train.pkl')
#     # fea_list = [fea_list_train, label_list_train, dur_train_max,
#     #                      count_long_train, count_all_train]
#     with open(pkl_path, 'wb') as f:
#         pickle.dump([fea_list, label_list, dur_max, count_long, count_all], f)
#
#
# def load_fea(pkl_path):
#     # pkl_path = os.path.join(feature_dir, 'train.pkl')
#     with open(pkl_path, 'rb') as f:
#         fea_list, label_list, dur_max, count_long, count_all = pickle.load(f)
#
#         return fea_list, label_list, dur_max, count_long, count_all
#
# # def save_fea(pkl_path, fea_list):
# #     # pkl = os.path.join(feature_dir, 'train.pkl')
# #     # fea_list = [fea_list_train, label_list_train, dur_train_max,
# #     #                      count_long_train, count_all_train]
# #     with open(pkl_path, 'wb') as f:
# #         pickle.dump(fea_list, f) # weird. Will fea_list here refer to the
# #         # values or names of variables?
#
#
# if __name__ == "__main__":
#     # test for extracting info from .bin files
#     species_name = ['bottlenose', 'common', 'spinner', 'melon-headed']
#     bin_dir_all = '/home/ys587/__Data/__whistle/tonals_20190210/label_bin_files/__all'
#     sound_dir = '/home/ys587/__Data/__whistle/__sound_species/'
#     contour_all, bin_wav_pair = bin_extract(bin_dir_all, sound_dir,
#                                             species_name)
#
#     # convert contour_train into selection tables
#     from cape_cod_whale.seltab import bin_to_seltab
#     seltab_out_path = '/home/ys587/__Data/__whistle/__seltab'
#     bin_to_seltab(contour_all, seltab_out_path)
#
#     # record the information of time step. Do they have a single time step or
#     # multiples?
#     timestep_file = "/home/ys587/__Data/__whistle/timestep_info.txt"
#     percentile_list = [0, 25, 50, 75, 100]
#     timestep_info(contour_all, timestep_file, percentile_list)
#
#
# def data_generator(whistle_image_target_4d, label_target_cat, batch_size, network_type='cnn'):
#     # (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     # y_train = np_utils.to_categorical(y_train,10)
#     # X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     # X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     # X_train = X_train.astype('float32')
#     # X_test = X_test.astype('float32')
#     # X_train /= 255
#     # X_test /= 255
#     num_samples = label_target_cat.shape[0]
#     num_batch = int(floor(num_samples/batch_size))
#     if network_type == 'cnn':  # cnn
#         whistle_image_target_4d =  whistle_image_target_4d[:num_batch*batch_size,:,:,:]
#         while 1:
#             for i in range(num_batch):
#                 # 1875 * 32 = 60000 -> # of training samples
#                 yield whistle_image_target_4d[
#                       i * batch_size:(i + 1) * batch_size, :, :, :], \
#                       label_target_cat[i * batch_size:(i + 1) * batch_size,:]
#     elif network_type == 'rnn':
#         whistle_image_target_4d = whistle_image_target_4d[:num_batch * batch_size, :, :]
#         while 1:
#             for i in range(num_batch):
#                 # print(i)
#                 yield whistle_image_target_4d[
#                       i * batch_size:(i + 1) * batch_size, :, :], \
#                       label_target_cat[i * batch_size:(i + 1) * batch_size,:]
#     else:  # conv2d_lstm
#         whistle_image_target_4d = whistle_image_target_4d[:num_batch * batch_size, :, :, :, :]
#         while 1:
#             for i in range(num_batch):
#                 # print(i)
#                 yield whistle_image_target_4d[
#                       i * batch_size:(i + 1) * batch_size, :, :], \
#                       label_target_cat[i * batch_size:(i + 1) * batch_size,:]





