import os
import time
import numpy as np
import pandas as pd

from pyeeg_toolbox.dsp.cwt import dcmwt

def detect_emi_presence(mtg_eeg_data: dict):

    fs = mtg_eeg_data['fs']
    mtg_labels = mtg_eeg_data['mtg_labels']
    
    emi_in_ch = np.zeros((len(mtg_labels), 1), dtype=bool)
    for ch_data_idx, mtg in enumerate(mtg_labels):
        
        # Detect Power Line Noise
        mtg_signal = mtg_eeg_data['data'][ch_data_idx]
        cmwt_freqs_emi, dcwt_emi = dcmwt(mtg_signal, fs, list(range(30, 120, 10)), nr_cycles=6)
        emi_present =  cmwt_freqs_emi[np.argmax(np.mean(dcwt_emi, axis=1))] == 60
        emi_in_ch[ch_data_idx] = emi_present

    return emi_in_ch

if __name__ == '__main__':
    fs = 1024
    mtg_labels = ['ch1','ch2','ch3','ch4','ch5']
    data = np.random.rand(len(mtg_labels), fs*3600)
    mtg_eeg_data = {'fs':1024, 'mtg_labels':mtg_labels, 'data':data}

    detect_emi_presence(mtg_eeg_data)