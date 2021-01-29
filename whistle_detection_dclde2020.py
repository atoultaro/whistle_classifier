#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect & classify whale species based on whistle vocalization

Slow. Finished 600+ in 10 hours but I have over 10,000+
Need to save all 18 class scores

Created on 5/19/20
@author: atoultaro
"""
import os
import glob
from math import floor, ceil
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import librosa
# from keras.models import load_model
from tensorflow.keras.models import load_model

# from species_classifier.all_whistle_training import make_sound_sel_table
# from species_classifier.species_lib import fea_pcem_nopulse_from_mel
from whistle_classifier.lib_feature import feature_whistleness, make_sound_sel_table_dclde2020, make_sound_sel_table_empty_dclde2020

# detection on multi-channel sound
# sound_path = '/mnt/DCLDE/noaa-pifsc-bioacoustic'
sound_path = '/mnt/DCLDE/noaa-pifsc-bioacoustic-48k'
deployment = ['1705', '1706']

species_list = ['NO', 'BD', 'MH', 'CD', 'STR', 'SPT', 'SPIN', 'PLT', 'RD', 'RT',
                'WSD', 'FKW', 'BEL', 'KW', 'WBD', 'DUSK', 'FRA', 'PKW', 'LPLT',
                'NAR', 'CLY', 'SPE', 'ASP']
num_species = 10  # noise not included
# # species_to_id = {'BD': 0, 'LCD': 1, 'SCD': 2, 'STR': 3, 'SPT': 4,
# #                  'SPIN': 5, 'PLT': 6, 'RT': 7, 'FKW': 8, 'NO': 9}
# species_to_id = {'BD', 'MH', 'STR', 'SPT', 'PLT', 'RD', 'RT', 'FKW', 'FRA', 'PKW'}

species_to_id = {'NO': 0, 'BD': 1, 'MH': 2, 'CD': 3, 'STR': 4, 'SPT': 5, 'SPIN': 6, 'PLT': 7, 'RD': 8, 'RT': 9,
                'WSD': 10, 'FKW': 11, 'BEL': 12, 'KW': 13, 'WBD': 14, 'DUSK': 15, 'FRA': 16, 'PKW': 17, 'LPLT': 18,
                'NAR': 19, 'CLY': 20, 'SPE': 21, 'ASP': 22}

dclde2020_outpath = '/home/ys587/__Data/__whistle/__whistle_dclde2020'

# trained model
# model_detection_path = os.path.join(dclde2020_outpath, '__trained_model/detection/2020-11-21_172710_resnet34_expt_alldata_run1_f1_lr_0.01/epoch_21_valloss_0.4099_valacc_0.9200.hdf5')  # 64, 50, 1
# model_detection_path = os.path.join(dclde2020_outpath, '')
model_detection_path = '/home/ys587/__Data/__whistle/__whislte_30_species/__fit_result_whistleness/__fea_mel_pcen_contour_no_pulses/2021-01-24_160849_resnet18_expt_alldata_run0_f1_lr_0.001/epoch_121_valloss_0.3218_valacc_0.9160.hdf5'  # use_pcen=True, remove_pulse=True

# model_classification_path = os.path.join(dclde2020_outpath, '__trained_model/classification/20201028_130857_file_mel_cnn14_attention_filter16_f1_p80/23-0.1873.hdf5')  # (151, 128, 1)
# model_classification_path = os.path.join(dclde2020_outpath, '/home/ys587/__Data/__whistle/__domain_adaptation/__fit_result_species/20201103_213954_domain_source_f80/89-0.0974.hdf5')
# model_classification_path = os.path.join(dclde2020_outpath, '/home/ys587/__Data/__whistle/__domain_adaptation/__fit_result_species/20201028_234434_file_cleaned_mel_cnn14_attention_filter64_f1_p79_lr_schedule/55-0.1155.hdf5')
# Attention over time on last two groups of conv layers
# model_classification_path = 'os.path.join(dclde2020_outpath, '/home/ys587/__Data/__whistle/__domain_adaptation/__fit_result_species/20201207_203345_file/19-0.2294.hdf5')'
model_classification_path = '/home/ys587/__Data/__whistle/__whislte_30_species/__fit_result_species/__good/20210128_111347/epoch_116_valloss_0.4936_valacc_0.8401.hdf5'
model_detector = load_model(model_detection_path)
model_classifier = load_model(model_classification_path)

# seltab output
seltab_out_path = os.path.join(dclde2020_outpath, '__seltab_out')

conf = dict()
conf['sample_rate'] = 48000
conf['time_reso'] = 0.02
conf['hop_length'] = int(conf['time_reso']*conf['sample_rate'])
conf['win_size'] = 1.  # 1-7erws window for whistleness
conf['hop_size'] = 0.25  # 0.5 sec stepsize
conf['time_indices_win'] = floor(conf['win_size'] / conf['time_reso'])  # 50
conf['time_indices_hop'] = floor(conf['hop_size'] / conf['time_reso'])  # 25
conf['freq_ind_low'] = 64
conf['secshift'] = 0.0  # 0.524
conf['alpha'] = 1.0-np.exp(np.log(0.15)*.02/10.)

# conf['whistle_thre_min'] = 0.01  # for two-class classifier
conf['whistle_thre_pos'] = 0.8
conf['contour_timethre'] = 20  # 0.4 s for dclde 2011
conf['trained_class_num'] = 'two'

for dd in deployment:
    sound_target = os.path.join(sound_path, dd)
    wav_files = glob.glob(sound_target+'/*.wav')
    wav_files.sort()

    for ww0 in range(len(wav_files)):
        ww = wav_files[ww0]
        ww_basename = os.path.basename(ww)
        print(ww_basename)

        samples, sr = librosa.load(ww, sr=conf['sample_rate'], mono=False)
        if samples.shape[0] == 0 or samples.shape[1] <= 0.5*sr:
            print('No samples or very small number of samples.')
            continue
        whistle_time_start = []
        whistle_time_end = []
        whistle_score = []
        begin_path = []
        file_offset = []
        chan_id = []
        species_id = []
        score_max = []

        score_no = []
        score_bd = []
        score_mh = []
        score_cd = []
        score_str = []
        score_spt = []
        score_spin = []
        score_plt = []
        score_rd = []
        score_rt = []
        score_wsd = []
        score_fkw = []
        score_bel = []
        score_kw = []
        score_wbd = []
        score_dusk = []
        score_fra = []
        score_pkw = []
        score_lplt = []
        score_nar = []
        score_cly = []
        score_spe = []
        score_asp = []

        #for cc in range(samples.shape[0]):
        for cc in [2]:
            print('channel: '+str(cc))
            samples_chan = np.asfortranarray(samples[cc, :])
            samples_chan = samples_chan - samples_chan.mean()
            # samples_chan = samples_chan / samples_chan.std()

            # whistleness feature
            whistle_freq = librosa.feature.melspectrogram(samples_chan,
                                                          sr=conf['sample_rate'],
                                                          hop_length=conf['hop_length'],
                                                          power=1)

            whistle_freq_list = []
            species_freq_list = []
            win_num = floor((whistle_freq.shape[1] - conf['time_indices_win']) / conf['time_indices_hop'])  # 0.5s hop

            # for nn in range(2, win_num-2):
            for nn in range(win_num):
                # whistleness features
                whistle_freq_curr = \
                    whistle_freq[:, nn*conf['time_indices_hop']:
                                    nn*conf['time_indices_hop']+conf['time_indices_win']]
                whistle_freq_curr = feature_whistleness(whistle_freq_curr, use_pcen=True, remove_pulse=True)
                # whistle_freq_curr = lib_feature.fea_pcen_nopulse_from_mel(whistle_freq_curr, freq_min=0)
                whistle_freq_list.append(whistle_freq_curr)

                # # species feature
                # samples_chan_curr = samples_chan[int((nn-2)*conf['hop_size']*conf['sample_rate']):int(((nn+2)*conf['hop_size']+conf['win_size'])*conf['sample_rate'])]
                # species_freq_curr = librosa.feature.melspectrogram(samples_chan_curr,
                #     sr=conf['sample_rate'], n_fft=1024, win_length=960,
                #     hop_length=960, power=1)
                # # species_freq_curr = \
                # #     species_freq[:, (nn-2)*conf['time_indices_hop']:
                # #                     (nn+2)*conf['time_indices_hop']
                # #                     +conf['time_indices_win']]
                # species_freq_curr = librosa.pcen(species_freq_curr * (2 ** 31))
                # species_freq_curr = lib_feature.nopulse_median(species_freq_curr)
                # species_freq_curr = lib_feature.avg_sub(species_freq_curr, conf['alpha'])
                #
                # # species_freq_curr = np.append(species_freq_curr, species_freq_curr.mean(axis=1).reshape(-1, 1), axis=1)
                # species_freq_list.append(species_freq_curr)

            # whistleness features' dimension changes
            if len(whistle_freq_list) >= 2:
                whistle_image = np.stack(whistle_freq_list)
            elif len(whistle_freq_list) == 1:
                whistle_image = np.expand_dims(whistle_freq_list[0], axis=0)
            else:  # len == 0
                continue
            whistle_image_4d = np.expand_dims(whistle_image, axis=3)

            # # species features' dimension changes
            # if len(species_freq_list) >= 2:
            #     whistle_image_species = np.stack(species_freq_list)
            #     whistle_image_species = np.transpose(whistle_image_species, (0, 2, 1))
            # else:
            #     species_freq_list[0] = np.transpose(species_freq_list[0], (0, 2, 1))
            #     whistle_image_species = np.expand_dims(species_freq_list[0], axis=0)
            # whistle_image_species_4d = np.expand_dims(whistle_image_species, axis=3)

            # make predictions
            predictions_detection = model_detector.predict(whistle_image_4d)
            predictions_score = model_classifier.predict(whistle_image_4d)  # <<==
            pred_class = np.argmax(predictions_score, axis=1)
            pred_score_max = np.max(predictions_score, axis=1)

            whistle_win_ind = np.where(predictions_detection[:, 1] > conf['whistle_thre_pos'])[0]
            if whistle_win_ind.shape[0] >= 1:
                # detected whistle start & end time
                whistle_time_start_curr = (whistle_win_ind+2) * conf['hop_size'] + conf['secshift']
                whistle_time_start.append(whistle_time_start_curr)
                whistle_time_end.append(whistle_time_start_curr + conf['win_size'])
                # detected whistle score
                whistle_score.append(predictions_detection[:, 1][whistle_win_ind])
                begin_path.append([ww] * whistle_win_ind.shape[0])
                # file_offset_curr = (whistle_win_ind+2) * conf['hop_size'] + conf['secshift']
                file_offset_curr = (whistle_win_ind + 2) * conf['hop_size']
                file_offset.append(file_offset_curr)
                chan_id.append([cc+1] * whistle_win_ind.shape[0])
                species_id.append(pred_class[whistle_win_ind])
                score_max.append(pred_score_max[whistle_win_ind])

                score_no.append(predictions_score[whistle_win_ind, 0])
                score_bd.append(predictions_score[whistle_win_ind, 1])
                score_mh.append(predictions_score[whistle_win_ind, 2])
                score_cd.append(predictions_score[whistle_win_ind, 3])
                score_str.append(predictions_score[whistle_win_ind, 4])
                score_spt.append(predictions_score[whistle_win_ind, 5])
                score_spin.append(predictions_score[whistle_win_ind, 6])
                score_plt.append(predictions_score[whistle_win_ind, 7])
                score_rd.append(predictions_score[whistle_win_ind, 8])
                score_rt.append(predictions_score[whistle_win_ind, 9])
                score_wsd.append(predictions_score[whistle_win_ind, 10])
                score_fkw.append(predictions_score[whistle_win_ind, 11])
                score_bel.append(predictions_score[whistle_win_ind, 12])
                score_kw.append(predictions_score[whistle_win_ind, 13])
                score_wbd.append(predictions_score[whistle_win_ind, 14])
                score_dusk.append(predictions_score[whistle_win_ind, 15])
                score_fra.append(predictions_score[whistle_win_ind, 16])
                score_pkw.append(predictions_score[whistle_win_ind, 17])
                score_lplt.append(predictions_score[whistle_win_ind, 18])
                score_nar.append(predictions_score[whistle_win_ind, 19])
                score_cly.append(predictions_score[whistle_win_ind, 20])
                score_spe.append(predictions_score[whistle_win_ind, 21])
                score_asp.append(predictions_score[whistle_win_ind, 22])

                # consider put scores from multi-species later
        # make sound selection tables
        if len(whistle_time_start) != 0:
            whistle_time_start = np.concatenate(whistle_time_start)
            whistle_time_end = np.concatenate(whistle_time_end)
            whistle_score = np.concatenate(whistle_score)
            begin_path = np.concatenate(begin_path)
            file_offset = np.concatenate(file_offset)
            chan_id = np.concatenate(chan_id)
            species_id = np.concatenate(species_id)
            seltab_out_file = os.path.splitext(ww_basename)[0]+'.txt'
            score_max = np.concatenate(score_max)

            score_no = np.concatenate(score_no)
            score_bd = np.concatenate(score_bd)
            score_mh = np.concatenate(score_mh)
            score_cd = np.concatenate(score_cd)
            score_str = np.concatenate(score_str)
            score_spt = np.concatenate(score_spt)
            score_spin = np.concatenate(score_spin)
            score_plt = np.concatenate(score_plt)
            score_rd = np.concatenate(score_rd)
            score_rt = np.concatenate(score_rt)
            score_wsd = np.concatenate(score_wsd)
            score_fkw = np.concatenate(score_fkw)
            score_bel = np.concatenate(score_bel)
            score_kw = np.concatenate(score_kw)
            score_wbd = np.concatenate(score_wbd)
            score_dusk = np.concatenate(score_dusk)
            score_fra = np.concatenate(score_fra)
            score_pkw = np.concatenate(score_pkw)
            score_lplt = np.concatenate(score_lplt)
            score_nar = np.concatenate(score_nar)
            score_cly = np.concatenate(score_cly)
            score_spe = np.concatenate(score_spe)
            score_asp = np.concatenate(score_asp)

            make_sound_sel_table_dclde2020(
                os.path.join(seltab_out_path, seltab_out_file),
                whistle_time_start,  whistle_time_end, begin_path, file_offset,
                whistle_score, conf['whistle_thre_pos'], chan=chan_id,
                class_id=species_id, score_max=score_max, score_no=score_no,
                score_bd=score_bd, score_mh=score_mh,
                score_cd=score_cd, score_str=score_str, score_spt=score_spt,
                score_spin=score_spin, score_plt=score_plt, score_rd=score_rd,
                score_rt=score_rt, score_wsd=score_wsd, score_fkw=score_fkw, score_bel=score_bel,
                score_kw=score_kw, score_wbd=score_wbd, score_dusk=score_dusk, score_fra=score_fra,
                score_pkw=score_pkw, score_lplt=score_lplt, score_nar=score_nar, score_cly=score_cly,
                score_spe=score_spe, score_asp=score_asp,
            )
        else:
            # output a black seltab.
            seltab_out_file = os.path.splitext(ww_basename)[0] + '.txt'
            make_sound_sel_table_empty_dclde2020(
                os.path.join(seltab_out_path, seltab_out_file)
            )
            # print('No selection table generated...')



# classification based on a trained classifierns