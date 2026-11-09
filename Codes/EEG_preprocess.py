# EDOIA EEG preparation source

# ============================================================================
# EEG preparation, epoch repair, ICA, time-frequency extraction
# ============================================================================
import os
import numpy as np
import pickle
import pandas as pd
import glob
import mne
import mne_icalabel
from mne.time_frequency import tfr_morlet
from autoreject import AutoReject

############################################################################# hyperparameters
freqnames = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
electrodenames = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

nrep, nelec, nfreq = 6, len(electrodenames), len(freqnames)
npics = 31 # with one extra

nsub = 82
subids_all = ['A1', 'A2']
subids_all.extend([str(x) for x in range(1, nsub + 1)])
nsub += 2
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

# bread block sequences
blockseqs = blockseqs.astype(int)

## create annotation
### function to specify image info in events table
def append_events_onlyimages(events, blockseqs, nrep, aunames):
    nau = len(aunames)
    
    eventtable = events.copy()
    # edit au
    for i in range(nau):
        eventtable['type'].loc[eventtable.type == aunames[i]] = ['FA/{}/R{}'.format(aunames[i], r) for r in range(1, 7)]
    
    # expend image part
    for i in range(nrep):
        temp = eventtable.loc[eventtable.type == 'B{}_Video'.format(i + 1)]
        rowid = np.where(eventtable.type == 'B{}_Video'.format(i + 1))[0][0]
        above = eventtable.iloc[:(rowid + 1), :]
        below = eventtable.iloc[(rowid + 1):, :]
        
        for j in range(31):
            # add only image                                                                B   I   P   PN 
            temp = pd.concat([temp, pd.DataFrame([[temp['latency'].tail(1).item() + 2, 4, 'Image/B{}/I{}/P{}/PN{}'.format(i+1, blockseqs[i][j][0], 1, blockseqs[i][j][1]),
                                                   temp['timestamp'].tail(1).item() + 2]], columns = temp.columns)])
            # add gap_element
            temp = pd.concat([temp, pd.DataFrame([[temp['latency'].tail(1).item() + 4, 2, 'gap_element',
                                                   temp['timestamp'].tail(1).item() + 4]], columns = temp.columns)])
            
        temp.latency = temp.latency - 2
        temp.timestamp = temp.timestamp - 2
        temp = temp.iloc[1:, :]

        eventtable = pd.concat([above, temp])
        eventtable = pd.concat([eventtable, below])
    return(eventtable)


### parameters
lowpass = 49; localbaseline = True; rerefer = 'average';
tmin = 0; tmax = 4; baseline = (-0.5, -0.1);

# default values
electrodenames = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
nrep = 6
aunames = ['AU1', 'AU4', 'AU5', 'AU6', 'AU44', 'AU45']
band_ranges = [np.arange(1, 4) - 1, np.arange(4, 8) - 1, np.arange(8, 13) - 1, np.arange(13, 31) - 1, np.arange(31, 46) - 1]
freqs = np.arange(1, lowpass + 1)

##### storages
##### storages
##### storages    
alltfrs = [[] for s in subids]
alltried = []
posinegas = []
all_epoch_ids = []
bad_elecs = []
remain_icas = np.zeros((nsub, 2)) # 0 for remained 1 for total
remain_epochs = np.zeros(nsub)
basetfrs = np.ones((nsub, len(electrodenames), lowpass)) * np.nan
posinegas_ids = []

for index, sub in enumerate(subids):
    
    print('Process subject: ', sub)
    # try:
    tried = np.arange(30)  # 0-29
    
    ###### events 
    markerfilenames = glob.glob('{} *Marker.csv'.format(sub))
    events = pd.read_csv(markerfilenames[0])
    events = events.drop(columns = ['key', 'marker_value', 'marker_id'])
    
    ####### read in eeg data file and find bad electrodes
    eegfilenames = glob.glob('{} *00.csv'.format(sub))
    eegraw = pd.read_csv(eegfilenames[0], header = 1)
    
    # see if origin time is in data
    if 'OriginalTimestamp' in eegraw.columns:
        temp = list(range(2, 56))
    else:
        temp = list(range(1, 55))
    temp.insert(0, 0)
    eegraw = eegraw.iloc[:, temp]
    eegraw = eegraw.to_numpy().transpose()
    # change uV to V
    eegraw[3:17, :] = eegraw[3:17, :] / 1e6
    
    # find bad channels using individual electrode EEG quality data
    temp = eegraw[41:]
    temp = temp[:, ~np.isnan(temp).any(axis=0)]
    temp = temp <= 2 # if all eq <= 2 then bad channel
    temp = np.where(temp.all(axis = 1))[0]
    bads = [electrodenames[i] for i in temp]
    bad_elecs.append(bads)
    
    # drop unnecessary columns from eegraw
    eegtimes = eegraw[0]
    eegraw = eegraw[3:17]
    coltypes = ['eeg'] * 14
    colnames = electrodenames
    
    ############## create raw object
    # note the sfreq
    if sub in ['A1', 'A2', '1', '2', '3', '4', '5', '6', '7', '8']:
        inforaw = mne.create_info(colnames, ch_types = coltypes, sfreq = 128)
    else:
        inforaw = mne.create_info(colnames, ch_types = coltypes, sfreq = 256)
        
    inforaw.set_montage('standard_1020')
    inforaw['bads'] = bads
    
    # create raw object
    eeg = mne.io.RawArray(eegraw, inforaw)
    
    if sub in ['A1', 'A2', '1', '2', '3', '4', '5', '6', '7', '8']:
        eeg = eeg.filter(l_freq = 1, h_freq = lowpass)  # higher low pass for ICA according to MNE
    else:
        eeg = eeg.resample(sfreq = 128).filter(l_freq = 1, h_freq = lowpass)
    
    # re-reference
    if rerefer == 'average':
        eeg.set_eeg_reference('average', projection = False)
    else:
        eeg.set_eeg_reference([rerefer])
    
    ############################### create epochs
    eventsapp = append_events_onlyimages(events, blockseqs, nrep, aunames)
    eventsclean = eventsapp.iloc[[any(substring in x for substring in ['Image']) for x in eventsapp.type], :].reset_index(drop = True)
    if sub == '30':
        temp = np.where([any(substring in x for substring in ['phase', 'Instruction', 'FA/AU4/R3', 'FA/AU4/R4', 'FA/AU4/R5', 'FA/AU4/R6']) for x in eventsclean.type])[0]
    else:
        temp = np.where([any(substring in x for substring in ['phase', 'Instruction']) for x in eventsclean.type])[0]
    eventsclean = eventsclean.drop(temp, axis = 0).reset_index(drop = True)
    
    annot = mne.Annotations(onset = eventsclean.latency.tolist(),
                            duration = eventsclean.duration.tolist(),
                            description = eventsclean.type.tolist(),
                            orig_time = events.iloc[0]['timestamp'] - events.iloc[0]['latency'])
    eeg.set_meas_date(annot.orig_time)
    eeg.set_annotations(annot)
    
    events_from_annot, event_dict = mne.events_from_annotations(eeg)
    
    ## epoching
    eegepochs = mne.Epochs(eeg, events_from_annot, tmin = -0.5, tmax = 4, preload = True,
                            event_id = event_dict, baseline = baseline)
    
    #### reject bad epochs by algorithm
    ar = AutoReject(n_interpolate = [1, 2, 3], random_state = 31123)
    eegepochs_clean = ar.fit_transform(eegepochs)

    #### ICA before linear baseline correction
    ica = mne.preprocessing.ICA(method = 'infomax', fit_params = dict(extended=True), max_iter = "auto", random_state = 31123)
    ica.fit(eegepochs_clean)
    labels = mne_icalabel.label_components(eegepochs_clean, ica, method = 'iclabel')

    exclude = [c for c in range(ica.n_components_) if labels['labels'][c] != 'brain' or labels['y_pred_proba'][c] < 0.6]
    
    remain_icas[index, 0] = ica.n_components_ - len(exclude)
    remain_icas[index, 1] = ica.n_components_
    
    ica.exclude = exclude
    ica.apply(eegepochs)
    
    # baseline correction after ICA
    if localbaseline == True:
        eegepochs.apply_baseline(baseline)
    
    # second autoreject after ICA adjustment
    ar = AutoReject(n_interpolate = [1, 2, 3], random_state = 31123)
    eegepochs_clean_again = ar.fit_transform(eegepochs)
    remain_epochs[index] = len(eegepochs_clean_again)
    
    #### baseline of closed eye and open eye
    baseevent = eventsapp.iloc[[any(substring in x for substring in ['eye']) for x in eventsapp.type], :].reset_index(drop = True)
    baseannot = mne.Annotations(onset = baseevent.latency.tolist(),
                                duration = baseevent.duration.tolist(),
                                description = baseevent.type.tolist(),
                                orig_time = events.iloc[0]['timestamp'] - events.iloc[0]['latency'])
    eeg.set_annotations(baseannot)
    
    events_from_annot, event_dict = mne.events_from_annotations(eeg)
    
    baseeegepochs = mne.Epochs(eeg, events_from_annot, tmin = -0.5, tmax = 12, preload = True,
                               event_id = event_dict, baseline = baseline)
    ica.apply(baseeegepochs)
    
    # baseline correction after ICA
    if localbaseline == True:
        baseeegepochs.apply_baseline(baseline)

    n_cycles = freqs / 2
    
    basetfr = tfr_morlet(baseeegepochs.average(), freqs = freqs, n_cycles = n_cycles,
                         return_itc = False, average = True, decim = 1, picks = electrodenames)
    
    basetfr = basetfr.data.mean(2)
    basetfrs[index] = basetfr

    ##### store data
    num_epochs = np.asarray([eegepochs_clean_again['I{}'.format(im + 1)].get_data().shape[0] for im in tried])
    tried = np.delete(tried, np.where(num_epochs == 0)[0])
    
    triedepochs = np.zeros([2, len(tried)])
    triedepochs[0] = tried
    triedepochs[1] = np.asarray([eegepochs_clean_again['I{}'.format(im + 1)].get_data().shape[0] for im in tried])
    posinega = [[] for im in tried]
    posinegas_id = [[] for im in tried]
    currenttfrs = np.ones([len(tried), 6, 14, len(freqs), 513]) * np.nan
    epoch_id = []
    
    
    for imindex, im in enumerate(tried):
        this_im_ids = eegepochs['I{}'.format(im + 1)].event_id.keys()
        posi_ids = np.where([int(key[len(key) - 1]) for key in this_im_ids])[0] + 1
        nega_ids = np.where([1 - int(key[len(key) - 1]) for key in this_im_ids])[0] + 1
        
        currentsub = eegepochs_clean_again['I{}'.format(im + 1)]
        temp = np.asarray([int(event[7]) for event in currentsub.event_id.keys()])  # 1 - 6
        epoch_id.append(temp)
        
        tfr = tfr_morlet(currentsub, freqs = freqs, n_cycles = n_cycles,
                         return_itc = False, average = False, decim = 1, picks = electrodenames)
        
        tfrdata = tfr.crop(tmin, tmax).data
        currenttfrs[imindex, (epoch_id[imindex] - 1)] = tfrdata
        
        posinega[imindex] = [int(key[len(key) - 1]) for key in currentsub.event_id.keys()]
        
        posi_id = np.asarray([np.where(posi_ids == temp[no])[0] + 1 for no, phrase in enumerate(posinega[imindex]) if phrase == 1]).flatten()
        nega_id = np.asarray([np.where(nega_ids == temp[no])[0] + 1 for no, phrase in enumerate(posinega[imindex]) if phrase == 0]).flatten()
        temp = np.asarray(posinega[imindex])
        temp[np.where(temp == 1)] = posi_id
        temp[np.where(temp == 0)] = nega_id
        posinegas_id[imindex] = temp

    # all 5 bands
    alltfrs[index] = np.stack([currenttfrs[:, :, :, band, :].mean(3) for band in band_ranges], axis = 3)
    
    alltried.append(triedepochs)
    posinegas.append(posinega)
    all_epoch_ids.append(epoch_id)
    posinegas_ids.append(posinegas_id)

# save out data
f = open('eeglabdata_autorej_eyebase_tfrs_unnormalized_0316_withposineganos.pkl', 'wb')

pickle.dump({'alltfrs': alltfrs, 'alltried': alltried, 'posinegas': posinegas, 'posinegas_ids': posinegas_ids,
            'all_epoch_ids': all_epoch_ids, 'bad_elecs': bad_elecs,
            'remain_icas': remain_icas, 'remain_epochs': remain_epochs,
            'basetfrs': basetfrs}, f)
