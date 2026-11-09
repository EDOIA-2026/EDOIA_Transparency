#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 01:02:55 2026

@author: blind for peer review


"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from caa_significance_5factor_080226 import configure_sig_test, sig_test

############################################################################# hyperparameters
freqnames = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
electrodenames = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

nrep, nelec, nfreq = 6, len(electrodenames), len(freqnames)
npics = 31 # with one extra

lowpass = 49; localbaseline = True; rerefer = 'average';
tmin = 0; tmax = 4; baseline = (-0.5, -0.1);

# default values
band_ranges = [np.arange(1, 4) - 1, np.arange(4, 8) - 1, np.arange(8, 13) - 1, np.arange(13, 31) - 1, np.arange(31, 46) - 1]
freqs = np.arange(1, lowpass + 1)

nsub = 82
subids_all = ['A1', 'A2']
subids_all.extend([str(x) for x in range(1, nsub + 1)])
nsub += 2
subids_all84 = subids_all.copy()

prefs_84 = pd.read_csv('EEG+Chinese+Adaptive_May+4,+2022_10.29.csv')
prefs_84 = prefs_84.loc[prefs_84.ResponseId.isin(subids_all)]
prefs_84 = prefs_84.filter(regex = 'Q4_')
prefs_84.columns = np.arange(1, 31)
prefs_84 = prefs_84.to_numpy().astype(float)

# remove bad subjects and subjects treid only few dishes
# only treid <= 2
subids_all.remove('31'); subids_all.remove('37'); subids_all.remove('65'); subids_all.remove('68')
# those subjects who went through a headset disconnection
subids_all.remove('77'); subids_all.remove('50')
# those subjects eeg way off reasonable range
subids_all.remove('13'); subids_all.remove('25'); subids_all.remove('59'); subids_all.remove('74')

subids = subids_all.copy()

nsub = len(subids)

imid = list(range(1, 31)) # 1-30

# read preference
prefs_all = pd.read_csv('EEG+Chinese+Adaptive_May+4,+2022_10.29.csv')
prefs_all = prefs_all.loc[prefs_all.ResponseId.isin(subids_all)]
prefs_all = prefs_all.filter(regex = 'Q4_')
prefs_all.columns = np.arange(1, 31)
prefs_all = prefs_all.to_numpy().astype(float)
prefs_all_not_nan = ~np.isnan(prefs_all)
all_not_nan = prefs_all_not_nan
prefs = prefs_all.copy()

### attribute preference
prefs_flavor = pd.read_csv('EEG+Chinese+Adaptive_May+4,+2022_10.29.csv')
prefs_flavor = prefs_flavor.loc[prefs_flavor.ResponseId.isin(subids_all)]
prefs_flavor = prefs_flavor.filter(regex = 'Q5_')
prefs_flavor.columns = np.arange(1, 31)
prefs_flavor = prefs_flavor.to_numpy().astype(float)

prefs_ingre = pd.read_csv('EEG+Chinese+Adaptive_May+4,+2022_10.29.csv')
prefs_ingre = prefs_ingre.loc[prefs_ingre.ResponseId.isin(subids_all)]
prefs_ingre = prefs_ingre.filter(regex = 'Q6_')
prefs_ingre.columns = np.arange(1, 31)
prefs_ingre = prefs_ingre.to_numpy().astype(float)

prefs_cook = pd.read_csv('EEG+Chinese+Adaptive_May+4,+2022_10.29.csv')
prefs_cook = prefs_cook.loc[prefs_cook.ResponseId.isin(subids_all)]
prefs_cook = prefs_cook.filter(regex = 'Q7_')
prefs_cook.columns = np.arange(1, 31)
prefs_cook = prefs_cook.to_numpy().astype(float)

# sig_quality
sig_quality = np.load('sigqual_74subs_8frontalelecs.npy')

# ## plot prefs distribution
pref_means = np.nanmean(prefs, axis = 0); pref_means = pref_means[~np.isnan(pref_means)]
pref_stds = np.nanstd(prefs, axis = 0); pref_stds = pref_stds[~np.isnan(pref_stds)]


### read eeg data
### read eeg data 
### read eeg data 
eeg_file = open('eeglabdata_autorej_eyebase_tfrs_unnormalized_0316_withposineganos.pkl', 'rb')

temp = pickle.load(eeg_file)
eeg_file.close()

alltfrs = temp.get('alltfrs')
alltried = temp.get('alltried'); posinegas = temp.get('posinegas'); posinegas_ids = temp.get('posinegas_ids'); 
all_epoch_ids = temp.get('all_epoch_ids')
bad_elecs = temp.get('bad_elecs'); remain_icas = temp.get('remain_icas'); remain_epochs = temp.get('remain_epochs'); basetfrs = temp.get('basetfrs')
del(temp)

### stand all data
for index, sub in enumerate(subids):
    print(index)
    
    # 5 bands data
    # band average for base
    sub_base = np.asarray([basetfrs[index][:, band].mean(1) for band in band_ranges]).transpose()
    sub_base = np.repeat(sub_base[:, :, np.newaxis], 513, axis = 2)
    alltfrs[index] = np.asarray([[10 * np.log10(alltfrs[index][d, exp] / sub_base) for exp in range(6)] for d in range(alltried[index].shape[1])])

##### run event study for each out-of-sample subject on each dish at each time point for each featur

# data not yet base-normalized and standardized, can first stand all alltfrs then do the following
def event_study_withsig(sub_index, focalelecs, window_start, window_stop):
    tried = alltried[sub_index][0].astype(int)
    
    # average over elecs
    elecs_id = [electrodenames.index(e) for e in focalelecs if e not in bad_elecs[sub_index]]
    sub_tfrs = np.nanmean(alltfrs[sub_index][:, :, elecs_id, :], 2)
    allelec_ids = [[electrodenames.index(e) for e in focalelecs if e not in bad_elecs[sub]] for sub in np.arange(nsub)]
    
    # storage
    ab_out = np.ones([len(tried), 6, 5, window_stop - window_start]) * np.nan
    ab_in = np.ones([len(tried), 6, 5, 192]) * np.nan
    
    rmse_out = np.ones([len(tried), 6, 5]) * np.nan
    
    for d in range(len(tried)):
        # print(d)
        
        # the average does not include the focal subject
        othersub_tfrs = np.asarray([np.nanmean(alltfrs[other][np.where(alltried[other][0] == tried[d])[0], :, allelec_ids[other], :], (0, 1)) for other in np.setdiff1d(np.arange(nsub), sub_index) if tried[d] in alltried[other][0].astype(int)])
        othersub_tfrs = np.mean(othersub_tfrs, 0)
    
        # enumerate 6 exposures
        for exp in range(6):
            if exp in (all_epoch_ids[sub_index][d] - 1):
                ys = sub_tfrs[d, exp, :, 0:192].transpose()
                
                # in-sample calibration
                model = LinearRegression().fit(othersub_tfrs[:, 0:192].transpose(), ys)
                ys_hat = model.predict(othersub_tfrs[:, 0:192].transpose())

                # calculate abnormal activities
                ys_out = sub_tfrs[d, exp, :, (257 + window_start):(257 + window_stop)].transpose()
                ys_out_hat = model.predict(othersub_tfrs[:, (257 + window_start):(257 + window_stop)].transpose())

                ab_out[d, exp] = (ys_out - ys_out_hat).transpose()
                ab_in[d, exp] = (ys - ys_hat).transpose()
                rmse_out[d, exp] = np.asarray([np.sqrt(np.mean((ys_out[:64, b] - ys_out_hat[:64, b]) ** 2)) for b in range(5)])
            else:
                pass
    return(ab_in, ab_out, rmse_out)


focalelecs = ['AF3', 'F3', 'F7', 'AF4', 'F4', 'F8']
ab_acts_notinclude = [event_study_withsig(sub, focalelecs, -64, 192) for sub in range(nsub)]
ab_acts = ab_acts_notinclude.copy()

# show the overall lines
congrus = []; incongrus = []
congru_subs = []; incongru_subs = []
congru_dishs = []; incongru_dishs = []
n_con = 0; n_incon = 0
higher_threshold = 7; lower_threshold = 6

con_prefs = []; incon_prefs = []
for sub in range(nsub):
    print(sub)
    elecs_id = [electrodenames.index(e) for e in focalelecs if e not in bad_elecs[sub]]

    tried = np.intersect1d(np.where(all_not_nan[sub])[0], alltried[sub][0]).astype(int) # 0 - 29
    pref = prefs[sub]

    for d_index, d in enumerate(tried):
        pref_d = pref[d]; pref_cook_d = prefs_cook[sub, d]; pref_ingre_d = prefs_ingre[sub, d]; pref_flavor_d = prefs_flavor[sub, d]
        which_d = list(alltried[sub][0]).index(d)
        
        posi_inds = np.where(posinegas[sub][which_d])[0]; nega_inds = np.where(1 - np.asarray(posinegas[sub][which_d]))[0]
        posi_inds = all_epoch_ids[sub][which_d][posi_inds] - 1
        nega_inds = all_epoch_ids[sub][which_d][nega_inds] - 1
        
        if len(posi_inds) > 0:
            if pref_d >= higher_threshold:
                congrus.append(np.concatenate((np.mean(ab_acts[sub][0][which_d, posi_inds, 2:4], (0, 1)), np.mean(ab_acts[sub][1][which_d, posi_inds, 2:4], (0, 1)))))
                congru_subs.append(sub); congru_dishs.append(d)
                n_con += 1
                con_prefs.append(np.array([pref_d, pref_cook_d, pref_ingre_d, pref_flavor_d]))
            if pref_d <= lower_threshold:
                incongrus.append(np.concatenate((np.mean(ab_acts[sub][0][which_d, posi_inds, 2:4], (0, 1)), np.mean(ab_acts[sub][1][which_d, posi_inds, 2:4], (0, 1)))))
                incongru_subs.append(sub); incongru_dishs.append(d)
                n_incon += 1
                incon_prefs.append(np.array([pref_d, pref_cook_d, pref_ingre_d, pref_flavor_d]))
                
        if len(nega_inds) > 0:
            if pref_d >= higher_threshold:
                incongrus.append(np.concatenate((np.mean(ab_acts[sub][0][which_d, nega_inds, 2:4], (0, 1)), np.mean(ab_acts[sub][1][which_d, nega_inds, 2:4], (0, 1)))))
                incongru_subs.append(sub); incongru_dishs.append(d)
                n_incon += 1
                incon_prefs.append(np.array([pref_d, pref_cook_d, pref_ingre_d, pref_flavor_d]))
            if pref_d <= lower_threshold:
                congrus.append(np.concatenate((np.mean(ab_acts[sub][0][which_d, nega_inds, 2:4], (0, 1)), np.mean(ab_acts[sub][1][which_d, nega_inds, 2:4], (0, 1)))))
                congru_subs.append(sub); congru_dishs.append(d)
                n_con += 1
                con_prefs.append(np.array([pref_d, pref_cook_d, pref_ingre_d, pref_flavor_d]))

cri = 1.96
congrus = np.asarray(congrus); incongrus = np.asarray(incongrus)
congrus = congrus[:, 192:384]; incongrus = incongrus[:, 192:384]
congrus_CIs = congrus.mean(0) + cri * np.std(congrus, 0) / np.sqrt(congrus.shape[0])
congrus_CIs = np.vstack((congrus_CIs, congrus.mean(0) - cri * np.std(congrus, 0) / np.sqrt(congrus.shape[0])))
incongrus_CIs = incongrus.mean(0) + cri * np.std(incongrus, 0) / np.sqrt(incongrus.shape[0])
incongrus_CIs = np.vstack((incongrus_CIs, incongrus.mean(0) - cri * np.std(incongrus, 0) / np.sqrt(incongrus.shape[0])))

fig, ax = plt.subplots(figsize = (8, 5.5))
ax.plot(np.arange(-500, 1000, 1500 / len(congrus.mean(0))), congrus.mean(0), label = 'Congruent')
ax.fill_between(np.arange(-500, 1000, 1500 / len(congrus.mean(0))), congrus_CIs[1, :], congrus_CIs[0, :], color = 'b', alpha = .07)
ax.plot(np.arange(-500, 1000, 1500 / len(incongrus.mean(0))), incongrus.mean(0), label = 'Incongruent', color = 'r')
ax.fill_between(np.arange(-500, 1000, 1500 / len(incongrus.mean(0))), incongrus_CIs[1, :], incongrus_CIs[0, :], color = 'r', alpha = .07)
ax.axhline(0, color = 'grey', ls = '--', lw = 1)
ax.axvline(0, color = 'grey', ls = '--', lw = 1)
ax.legend(loc = 'upper left')
ax.annotate('Statement Onset', (-195, -0.48))
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Abnormal Activity (dB)')
ax.set_xlim((-500, 1000))
fig.show()


######## CAA hypotheses testing
######## CAA hypotheses testing
######## CAA hypotheses testing
window_start = int(00/1000 * 128) # ms
window_stop = int(1000/1000 * 128) #
L2 = window_stop - window_start

# find bad dishes with structural change pre and post phrases
bad_dish = np.zeros(30)
allelec_ids = [[electrodenames.index(e) for e in focalelecs if e not in bad_elecs[sub]] for sub in np.arange(nsub)]

window_start += 64
window_stop += 64
# prepare data
ab_test_features = []
for sub in range(nsub):
    print(sub)
    tried = np.intersect1d(np.where(all_not_nan[sub])[0], alltried[sub][0]).astype(int) # 0 - 29
    pref = prefs[sub]; pref_flavor = prefs_flavor[sub]; pref_ingre = prefs_ingre[sub]; pref_cook = prefs_cook[sub]
    median_pref = np.nanmedian(pref)
        
    for d_index, d in enumerate(tried):
        pref_d = pref[d]; pref_flavor_d = pref_flavor[d]; pref_ingre_d = pref_ingre[d]; pref_cook_d = pref_cook[d]
        which_d = list(alltried[sub][0]).index(d)
        
        posi_inds = np.where(posinegas[sub][which_d])[0]; nega_inds = np.where(1 - np.asarray(posinegas[sub][which_d]))[0]
        
        posi_inds = all_epoch_ids[sub][which_d][posi_inds] - 1
        nega_inds = all_epoch_ids[sub][which_d][nega_inds] - 1
        
        # calcaulate R_m,t and R^bar_m
        othersub_tfrs = np.asarray([np.nanmean(alltfrs[other][np.where(alltried[other][0] == d)[0], :, allelec_ids[other], 2:4], (0, 1, 2)) for other in np.setdiff1d(np.arange(nsub), sub) if d in alltried[other][0].astype(int)])
        R_mt = np.mean(othersub_tfrs, 0)
        Rbar_m = np.mean(R_mt[:192])
        quant = np.sum((R_mt[(192 + window_start):(192 + window_stop)] - Rbar_m) ** 2) / np.sum((R_mt[:192] - Rbar_m) ** 2)
        
        if len(posi_inds) > 0:            
            S2 = np.sum((np.nanmean(ab_acts[sub][0][which_d, posi_inds, 2:4], (0, 1))) ** 2) / 186 # 192 - 6 parameters for the 5 band predictors
            CAR = np.nansum(np.nanmean(ab_acts[sub][1][which_d, posi_inds, 2:4, window_start:window_stop], (0, 1)))
            temp = np.hstack((CAR, np.mean(np.nanmean(ab_acts[sub][0][which_d, posi_inds, 2:4, :192], (0, 1)) > 0), CAR / np.sqrt(L2 * S2)))
            temp1 = np.array([sub, d, pref_d, pref_flavor_d, pref_ingre_d, pref_cook_d, 1 * (pref_d >= median_pref), 1])
            ab_test_features.append(np.hstack((temp1, temp)).reshape((1, -1)))
        
        if len(nega_inds) > 0:
            S2 = np.sum((np.nanmean(ab_acts[sub][0][which_d, nega_inds, 2:4], (0, 1))) ** 2) / 186
            CAR = np.nansum(np.nanmean(ab_acts[sub][1][which_d, nega_inds, 2:4, window_start:window_stop], (0, 1)))
            temp = np.hstack((CAR, np.mean(np.nanmean(ab_acts[sub][0][which_d, nega_inds, 2:4, :192], (0, 1)) > 0), CAR / np.sqrt(L2 * S2)))
            temp1 = np.array([sub, d, pref_d, pref_flavor_d, pref_ingre_d, pref_cook_d, 1 * (pref_d >= median_pref), 0])
            ab_test_features.append(np.hstack((temp1, temp)).reshape((1, -1)))

ab_test_features = np.concatenate(ab_test_features)
ab_test_features = pd.DataFrame(ab_test_features)
ab_test_features.columns = ['Sub', 'Dish', 'Pref', 'Pref_flavor', 'Pref_ingre', 'Pref_cook', 'Like_median', 'Phrase', 'AlphaBeta', 'AlphaBeta_ARpos', 't_CARi']

temp = ab_test_features[(ab_test_features.Pref >= higher_threshold) | (ab_test_features.Pref <= lower_threshold)]
temp['Like_01'] = (temp['Pref'] >= higher_threshold) * 1
print(np.mean(temp['Like_01']), temp.shape)

Y = 'Like_01'
temp['Congru'] = 1 * (temp[Y] == temp.Phrase)

# Build and cache the corrected five-predictor quantities once. Thereafter,
# sig_test(which) accepts any subgroup selected from the legacy temp table.
caa_5factor_data = configure_sig_test(globals())

#  test
b = 'AlphaBeta'

# this dish level liking percentages will be calculated later in this file
dish_like_perc = final_preds[['Dish', 'dish_like_perc']].drop_duplicates(keep = 'first').sort_values(by = 'Dish')
dish_like_perc = dish_like_perc.dish_like_perc.values
dish_like_perc = np.insert(dish_like_perc, 26, np.nan)

work = np.where((dish_like_perc <= 0.7) & (dish_like_perc > 0.3))[0]
work = np.where((dish_like_perc <= 0.65) & (dish_like_perc > 0.35))[0]
work = np.where((dish_like_perc <= 0.6) & (dish_like_perc > 0.4))[0]
work = np.where((dish_like_perc <= 0.55) & (dish_like_perc > 0.45))[0]
print(len(work))

# Four significance tests for the currently selected subgroup.
which = temp.loc[temp.Congru == 1]
which = temp.loc[(temp.Congru == 1) & (~temp.Dish.isin(work))]
which = temp.loc[(temp.Congru == 1) & (temp.Dish.isin(work))]
congruent_sig_results = sig_test(which)
print(congruent_sig_results.to_string(index=False))

which = temp.loc[temp.Congru == 0]
which = temp.loc[(temp.Congru == 0) & (temp.Dish.isin(work))]
which = temp.loc[(temp.Congru == 0) & (~temp.Dish.isin(work))]
incongruent_sig_results = sig_test(which)
print(incongruent_sig_results.to_string(index=False))


### prediction using overall car

final_preds = temp[['Sub', 'Dish', 'Pref', 'Like_median', 'Like_01']].drop_duplicates()
final_preds.reset_index(inplace = True)
final_preds['Pred'] = np.ones(final_preds.shape[0]) * np.nan
final_preds['Pred_CARi'] = np.ones(final_preds.shape[0]) * np.nan

for i in range(final_preds.shape[0]):
    each = temp.loc[(temp.Sub == final_preds.iloc[i]['Sub']) & (temp.Dish == final_preds.iloc[i]['Dish']) & (temp.Like_median == final_preds.iloc[i]['Like_median'])]

    thres = -0
    
    each = np.asarray(each[['Phrase', 'AlphaBeta']])
    num_nega = np.sum(each[:, 1] < thres)
    
    if num_nega == 0:
        final_preds['Pred'][i] = np.nan
    elif num_nega == 1:
        final_preds['Pred'][i] = 1 - each[:, 0][np.where(each[:, 1] < thres)[0]]
        final_preds['Pred_CARi'][i] = each[:, 1][np.where(each[:, 1] < thres)[0]]
    elif num_nega == 2:
        where = np.where(np.abs(each[:, 1]) == np.nanmax(np.abs(each[:, 1])))
        final_preds['Pred'][i] = 1 - each[where[0], 0]
        final_preds['Pred_CARi'][i] = each[where[0], 1]

np.sum(np.isnan(final_preds.Pred))


# add dish level majority
# final_preds['dish_major_local'] = np.nan
final_preds['dish_major'] = np.nan
final_preds['dish_like_perc'] = np.nan
final_preds['dish_like_perc_loo'] = np.nan
for i in range(final_preds.shape[0]):
    sub = final_preds['Sub'].values[i]; dish = final_preds['Dish'].values[i]

    test = np.delete(prefs_84, subids_all84.index(subids[int(sub)]), 0)[:, int(dish)]
    final_preds['dish_major'][i] = 1 * (np.mean(test[~np.isnan(test)] >= higher_threshold) >= 0.5)
    final_preds['dish_like_perc'][i] = np.mean(prefs_84[:, int(dish)][~np.isnan(prefs_84[:, int(dish)])] >= higher_threshold)
    final_preds['dish_like_perc_loo'][i] = np.mean(test[~np.isnan(test)] >= higher_threshold)
        

np.sum(np.isnan(final_preds.dish_major))
np.sum(np.isnan(final_preds.Pred))

final_preds = final_preds[final_preds['Pred'].notna()]
final_preds = final_preds.reset_index()
final_preds = final_preds.drop(columns = ['index', 'level_0'])

temp_pred = final_preds.loc[(final_preds.dish_like_perc >= 0) & (final_preds.dish_like_perc <= 1)]
temp_pred = final_preds.loc[(final_preds.dish_like_perc > 0.3) & (final_preds.dish_like_perc <= 0.7)]
temp_pred = final_preds.loc[(final_preds.dish_like_perc > 0.35) & (final_preds.dish_like_perc <= 0.65)]
temp_pred = final_preds.loc[(final_preds.dish_like_perc > 0.40) & (final_preds.dish_like_perc <= 0.60)]
temp_pred = final_preds.loc[(final_preds.dish_like_perc > 0.45) & (final_preds.dish_like_perc <= 0.55)]
temp_pred = final_preds.loc[(final_preds.dish_like_perc > 0.47) & (final_preds.dish_like_perc <= 0.53)]

print(temp_pred.shape)
print(len(np.unique(temp_pred.Dish)))

print(np.round(np.mean(temp_pred[Y] == temp_pred['Pred']), 3))
print(np.round(np.mean(temp_pred[Y] == temp_pred['dish_major']), 3))

print(stats.ttest_1samp(1 * (temp_pred[Y] == temp_pred['Pred']), 0.5))
print(stats.ttest_rel(1 * (temp_pred[Y] == temp_pred['Pred']), 1 * (temp_pred[Y] == temp_pred['dish_major'])))
print('F-scores: ')
print(f1_score(temp_pred[Y], temp_pred['Pred']))
print(f1_score(temp_pred[Y], temp_pred['dish_major']))
