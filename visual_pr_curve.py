#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 7/7/21
@author: atoultaro
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import confusion_matrix, classification_report

result_path  = '/home/ys587/__Data/__whistle/__whistle_30_species/__fit_result_species/__final'

species_dict = {'BD': 0, 'CD': 1, 'STR': 2, 'SPT': 3, 'SPIN': 4, 'PLT': 5, 'RT': 6,  'FKW': 7}
num_species = len(species_dict)
species_list = list(species_dict.keys())
species_id = list(species_dict.values())

# split_type = 'deployment'
# split_type = 'encounter'
# split_type = 'clip'

# run 0
split_type = 'deployment'
# target_split_folder = '20210614_202736_deployment_run0_mixup_spp'
# target_split_folder = '20210629_134916_deployment_run0_attention_lr_3p33e-5'
target_split_folder = '20210701_010815_deployment_run0_B3_lr_1e-3'
# target_split_folder = '/home/ys587/__Data/__whistle/__whistle_30_species/__fit_result_species/__final/20210629_134916_deployment_run0_attention_lr_3p33e-5'

# split_type = 'encounter'
# target_split_folder = '20210623_162002_encounter_run0_mixup_spp'
# target_split_folder = '20210702_165627_encounter_run0_attention'

# split_type = 'clip'
# target_split_folder = '20210623_162213_clip_run0_mixup_spp'
# target_split_folder = '20210702_205644_clip_run0_attention'
# target_split_folder = '/home/ys587/__Data/__whistle/__whistle_30_species/__fit_result_species/__final/20210702_205644_clip_run0_attention'

if split_type == 'deployment':
    # deployment: e.g. STAR2006_test_results.npz
    fold_list = ['STAR2000', 'STAR2003', 'STAR2006', 'HICEAS2002', 'PICEAS2005']

    # for ii in range(len(fold_list)):
    label_test_all = []
    label_pred_all = []
    for ee in fold_list:
        label_temp = np.load(os.path.join(result_path, target_split_folder, ee + '_test_results.npz'))
        label_test_all.append(label_temp['label_test'])
        label_pred_all.append(label_temp['label_pred'])
    label_pred_all = np.concatenate(label_pred_all)
    label_test_all = np.concatenate(label_test_all)

    precision_list = []
    recall_list = []
    threshold_list = []
    # for cc in range(num_species):
    for cc in range(1):
        label_pred_cc = label_pred_all[:, cc]

        label_test_curr = label_test_all
        label_test_curr[np.where(label_test_curr != cc)[0]] = 0
        precision_curr, recall_curr, threshold_curr = precision_recall_curve(label_test_curr, label_pred_cc)
        precision_list.append(precision_curr)
        recall_list.append(recall_curr)
        threshold_list.append(threshold_curr)


    # print(classification_report(label_test_all, np.argmax(label_pred_all, axis=1), target_names=species_list,
    #                             digits=3))  # )  #, target_names=None, sample_weight=None2, output_dict=False, zero_division='warn')
    # np.set_printoptions(linewidth=200, precision=3, suppress=True)
    #
    # print("Confusion matrix:")
    # # cm = confusion_matrix(label_train[:label_train_pred.shape[0]], np.argmax(label_train_pred, axis=1), labels=species_id)
    # cm = confusion_matrix(label_test_all, np.argmax(label_pred_all, axis=1), labels=species_id)
    #
    # cm2 = cm * 1.0
    # for ii in range(cm.shape[0]):
    #     cm_row = cm[ii, :] * 1.0
    #
    #     cm_row_sum = cm_row.sum()
    #     if cm_row_sum != 0:
    #         cm2[ii, :] = cm_row / cm_row_sum
    #     else:
    #         cm2[ii, :] = np.zeros(cm.shape[1])
    #
    # print(species_list)
    # print('')
    # print(cm)
    # print('')
    # print(cm2)

    # from sklearn.metrics import plot_confusion_matrix
    # from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
    #
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_list)
    # disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=species_list)
    #
    # fig, ax = plt.subplots(figsize=[15, 15])
    # disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal', values_format=None,
    #           colorbar=True)
    # # plt.show()
    #
    # fig2, ax = plt.subplots(figsize=[15, 15])
    # disp2.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal', values_format='.3f',
    #            colorbar=True)
    # plt.show()
