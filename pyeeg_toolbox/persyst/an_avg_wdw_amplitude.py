import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict
from pyeeg_toolbox.persyst.spike_cumulator import SpikeCumulator
from pyeeg_toolbox.eeg_io.eeg_io import EEG_IO
from scipy.signal import find_peaks, peak_prominences
from studies_info import fr_four_patients
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer

class AverageWdwAnalyzer(SpikeAmplitudeAnalyzer):
    """
    A class that analyzes for each channel, the averaged time windows that coincide with a spike that was detected on any channel.
    """

    def __init__(self, 
                 pat_id:str=None,
                 ieeg_data_path:str=None, 
                 sleep_data_path:str=None, 
                 ispikes_data_path:str=None, 
                 sleep_stages_map:Dict[int, str]={0: "Unknown", 1: "N3", 2: "N2", 3:"N1", 4:"REM", 5:"Wake"},
                 )->None:
        """
        Initialize the SpikeAmplitudeAnalyzer class..

        Args:
            pat_id (str): The ID of the patient.
            ieeg_data_path (str): The path to the iEEG data files.
            sleep_data_path (str): The path to the sleep stage data files.
            ispikes_data_path (str): The path to the iSpikes data files.
            sleep_stages_map (Dict[int, str]): A dictionary mapping sleep stage codes to their names.

        Returns:
            None
        """
        self.pat_id = pat_id
        self.ieeg_data_path = ieeg_data_path
        self.sleep_data_path = sleep_data_path
        self.ispikes_data_path = ispikes_data_path
        self.sleep_stages_map = sleep_stages_map

        if self.sleep_data_path is None:
            self.sleep_data_path = self.ieeg_data_path
        if self.ispikes_data_path is None:
            self.ispikes_data_path = self.ieeg_data_path

        self.eeg_file_extension = ".lay"
        self.pat_files_ls = None
        self.spike_cumulator = None

        super().__init__(
            pat_id=self.pat_id,
            ieeg_data_path=self.ieeg_data_path, 
            sleep_data_path=self.sleep_data_path, 
            ispikes_data_path=self.ispikes_data_path, 
            sleep_stages_map=self.sleep_stages_map,
        )

    def run(self, file_extension:str='.lay', mtg_t:str='ir', plot_ok:bool=False)->None:
        """
        This function orchestrates the entire wdw analysis process.

        It performs the following steps:
        1. Retrieves all files with the specified extension (default is '.lay'), from the specified directory.
        2. Calculates the total duration of each sleep stage for the patient.
        3. Cumulates for each slep stage, the signals from the windows that temporally coincide with spike events from any channel
        using a matrix where each row corresponds to an EEG channel.

        Parameters:
        file_extension (str): The file extension to filter for. Default is '.lay'.
        mtg_t (str): The montage type to use for EEG data reading. Default is 'ir'. Options:
            'sr' = Scalp Referential
            'sb' = Scalp Bipolar
            'ir' = Intracranial Referential
            'ib' = Intracranial Bipolar
        plot_ok (bool): A flag indicating whether to plot the EEG segments containing spikes. Default is False.

        Returns:
        None
        """
        self.get_files_in_folder(file_extension)
        self.get_sleep_stages_duration_sec()
        self.get_channel_avg_wdw(mtg_t, plot_ok) # 'sr', 'sb', 'ir', 'ib'


    def get_channel_avg_wdw(self, mtg_t:str='ir', plot_ok:bool=False)->SpikeCumulator:
        """
        This function cumulates for each channel and sleep stage, the windows that temporally coincide with a spike event coming from any channel.

        Parameters:
        mtg_t (str): The montage type to use for EEG data reading. Default is intracranial referential='ir'.
        plot_ok (bool): A flag indicating whether to plot the EEG segments containing spikes. Default is False.

        Returns:
        None
        """
        assert len(self.pat_files_ls) > 0, f"No files found in folder {self.ieeg_data_path}"

        # Initialize structure to count number of sample windows and accumulate these sample windows
        self.initialize_spike_cumulator(mtg_t=mtg_t)

        this_pat_eeg_file_path = ""           
        for record_idx in np.arange(start=0, stop=len(self.pat_files_ls)):
            this_pat_eeg_file_path = self.pat_files_ls[record_idx]
            eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_file_path, mtg_t=mtg_t)
            #fs = eeg_reader.fs

            # Read sleep data and spike detections
            sleep_data_df = self.read_sleep_stages_data(this_pat_eeg_file_path)
            spike_data_df = self.read_spike_data(this_pat_eeg_file_path)
            spike_data_df = spike_data_df.sort_values(by=['Time'], ascending=True)
            # Clean spike channel using same method used for eeg channel cleaning
            spike_data_df.Channel = eeg_reader.clean_channel_labels(spike_data_df.Channel.values.tolist())

            for spike_idx in range(len(spike_data_df)):

                spike_center_sec = spike_data_df.at[spike_idx,'Time']
                spike_wdw_start_sec = (spike_center_sec-0.5)
                spike_wdw_start_sample = int((spike_center_sec-0.5)*eeg_reader.fs)
                spike_wdw_end_sample = int((spike_center_sec+0.5)*eeg_reader.fs)
                if spike_wdw_start_sample < 0 or spike_wdw_end_sample > eeg_reader.n_samples-1:
                    continue

                spike_sleep_stage_code = sleep_data_df.I1_1[sleep_data_df.Time==int(spike_center_sec)].to_numpy().flatten()
                assert len(spike_sleep_stage_code)==1, "Could not assign a sleep stage to spike"
                if np.isnan(spike_sleep_stage_code):
                    continue
                spike_sleep_stage_name = self.sleep_stages_map[int(spike_sleep_stage_code[0])]
                spike_polarity = spike_data_df.at[spike_idx,'Sign']>0

                # spike_det_ch_name = spike_data_df.at[spike_idx,'Channel']
                # spike_eeg_ch_name = [ch for ch in eeg_reader.ch_names if spike_det_ch_name.lower()==ch.lower()]
                # assert len(spike_eeg_ch_name)==1, "Could not assign a channel to spike"
                # spike_eeg_ch_name = spike_eeg_ch_name[0]
                # spike_eeg_ch_idx = eeg_reader.ch_names.index(spike_eeg_ch_name)

                for eeg_chi, eeg_chname in enumerate(eeg_reader.ch_names):

                    # Get the EEG segment containing the spike
                    spike_wdw = eeg_reader.get_data(picks=eeg_chi, start=spike_wdw_start_sample, stop=spike_wdw_end_sample).flatten()
                    if not spike_polarity:
                        spike_wdw = spike_wdw*-1

                    # Undersample the EEG segment containing the spike
                    fs_us = self.spike_cumulator.get_undersampling_frequency()
                    spike_wdw_us = self.undersample_signal(spike_wdw, fs_us)

                    spike_feats=(0,0)#self.get_spike_features(spike_wdw_us, fs_us, True)
                    spike_ampl = spike_feats[0]
                    spike_freq= spike_feats[1]
                    if np.isnan(spike_feats[0]):
                        continue

                    self.spike_cumulator.add_spike(spike_sleep_stage_name, eeg_chname, spike_wdw_us, spike_ampl, spike_freq)

                    if plot_ok: # or np.isnan(spike_feats[0]):
                        plt.figure(figsize=(10,6))
                        plt.subplot(1, 2, 1)
                        time_vec = spike_wdw_start_sec+np.arange(len(spike_wdw))/eeg_reader.fs
                        plt.plot(time_vec, spike_wdw, '-k', linewidth=1)
                        plt.plot([np.mean(time_vec)]*2, [np.min(spike_wdw), np.max(spike_wdw)], '--r', linewidth=1)
                        plt.xlim(np.min(time_vec), np.max(time_vec))

                        plt.subplot(1, 2, 2)
                        time_vec = spike_wdw_start_sec+np.arange(len(spike_wdw_us))/fs_us
                        plt.plot(time_vec, spike_wdw_us, '-k', linewidth=1)
                        plt.plot([np.mean(time_vec)]*2, [np.min(spike_wdw_us), np.max(spike_wdw_us)], '--r', linewidth=1)
                        plt.xlim(np.min(time_vec), np.max(time_vec))    

                        plt.suptitle(f"PatientID {this_pat_eeg_file_path.name}\n SpikeNr:{spike_idx+1}/{len(spike_data_df)}\nSpikeCh:{spike_eeg_ch_name}, SleepStage:{spike_sleep_stage_name}, Polarity: {spike_polarity}")

                        # Display plot and wait for user input.
                        #plot_ok = not plt.waitforbuttonpress()
                        plt.waitforbuttonpress()
                        plt.close()
                        #if not plot_ok:
                        #    return None

            print(f"{eeg_reader.filename}: {(record_idx+1)/len(self.pat_files_ls)*100}%")

        return self.spike_cumulator
    
    def save_wdw_cumulator(self, filepath:str=None):
        """
        Save the SpikeCumulator object to a file using pickle serialization.

        Parameters:
        filepath (str): The path to the file where the SpikeCumulator object will be saved.

        Returns:
        None
        """
        if self.spike_cumulator is not None:
            with open(filepath, 'wb') as handle:
                pickle.dump(self.spike_cumulator, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_wdw_cumulator(self, filepath:str=None) -> None:
        """
        Load a SpikeCumulator object from a file using pickle deserialization.

        Parameters:
        filepath (str): The path to the file where the SpikeCumulator object is saved.

        Returns:
        None
        """
        with open(filepath, 'rb') as handle:
            self.spike_cumulator = pickle.load(handle)
        return self.spike_cumulator