import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Dict
from pyeeg_toolbox.persyst.avg_wdw_cumulator import AvgWdwCumulator
from pyeeg_toolbox.eeg_io.eeg_io import EEG_IO
from scipy.signal import find_peaks, peak_prominences
from studies_info import fr_four_patients
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from pyeeg_toolbox.utils.convert_mapped_channels import correct_1096_chnames

class DailySpikeWdwAnalyzer(SpikeAmplitudeAnalyzer):
    """
    A class that analyzes for each channel, the averaged time windows that coincide with a spike that was detected on any channel.
    """

    def __init__(self, 
                 pat_id:str=None,
                 ieeg_data_path:str=None, 
                 sleep_data_path:str=None, 
                 ispikes_data_path:str=None,
                 ch_coordinates_data_path:str=None,
                 szr_info_data_path:str=None,
                 sleep_stages_map:Dict[int, str]={0: "Unknown", 1: "N3", 2: "N2", 3:"N1", 4:"REM", 5:"Wake"},
                 output_path:Path=None,
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
        self.ch_coordinates_data_path = ch_coordinates_data_path
        self.szr_info_data_path = szr_info_data_path
        self.sleep_stages_map = sleep_stages_map
        self.output_path = output_path

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

    def MinMaxScaler(self, data):
        return (data - np.min(data))/(np.max(data)-np.min(data))
    
    def run(self, file_extension:str='.lay', mtg_t:str='ir', force_recalc:bool=False)->None:
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
        force_recalc (bool): A flag indicating whether force_recalc.

        Returns:
        None
        """
        self.get_files_in_folder(file_extension)
        eeg_filepaths_byday_df = self.get_eeg_files_by_day(file_extension, mtg_t)
        stages_duration_byday = self.get_sleep_stages_duration_sec_byday(eeg_filepaths_byday_df)

        eeg_filepaths_ls = eeg_filepaths_byday_df.EEG_Filepath.values.tolist()
        assert len(eeg_filepaths_ls) > 0, f"No files found in folder {self.ieeg_data_path}"
        Parallel(n_jobs=1)(delayed(self.get_channel_avg_wdw_vectorized)(this_eeg_fpath, mtg_t, force_recalc) for this_eeg_fpath in eeg_filepaths_ls)

        avg_spike_by_day_stage_ch_df = self.get_avg_wdw_by_day_by_ch(eeg_filepaths_byday_df, mtg_t, force_recalc)
        avg_spike_by_day_stage_ch_df = self.get_channel_coordinates(avg_spike_by_day_stage_ch_df)
        avg_spike_by_day_stage_ch_df = self.get_soz_info(avg_spike_by_day_stage_ch_df)

        # Add patient ID to the dataframe
        avg_spike_by_day_stage_ch_df['PatID'] = self.pat_id
        cols = ['PatID'] + [col for col in avg_spike_by_day_stage_ch_df.columns if col != 'PatID']
        avg_spike_by_day_stage_ch_df = avg_spike_by_day_stage_ch_df[cols]

        avg_spike_by_day_stage_ch_df.to_csv(self.output_path / f"{self.pat_id}_AvgSpikeWdwByDay.csv", index=False)

        pass
        
    def get_eeg_files_by_day(self, file_extension, mtg_t:str='ir')->None:
        """
        This function retrieves all EEG files in the specified directory and groups them by day.

        Parameters:
        None

        Returns:
        None
        """
        # Get the eeg measurement date of all files
        eeg_meas_date_ls = []
        for eeg_fpath in self.pat_files_ls:
            eeg_meas_date_ls.append(EEG_IO(eeg_filepath=eeg_fpath, mtg_t=mtg_t).meas_date)
            pass
        
        # Find the file with the earliest measurement date and use it as reference date
        min_file_date = np.min(eeg_meas_date_ls)
        ref_date = min_file_date.replace(hour=0, minute=0, second=0, microsecond=0)

        eeg_days_dict = {'EEG_Filepath':self.pat_files_ls, 'EEG_MeasDate':eeg_meas_date_ls, 'EEG_Day':[(fdate-ref_date).days for fdate in eeg_meas_date_ls]}
        eeg_filepaths_byday_df = pd.DataFrame(eeg_days_dict)

        # merge days with less than 18 files with the next day (if the under-represented day is the first day),
        # or with the previous (if the under-represented day is the last day)
        days_ls = eeg_filepaths_byday_df.EEG_Day.unique()
        for day in days_ls:
            nr_files_in_day = len(eeg_filepaths_byday_df.EEG_Filepath[eeg_filepaths_byday_df.EEG_Day==day])
            if nr_files_in_day < 18 and day != days_ls[-1]:
                eeg_filepaths_byday_df.loc[eeg_filepaths_byday_df.EEG_Day==day, 'EEG_Day'] = day+1
            if nr_files_in_day < 18 and day == days_ls[-1]:
                eeg_filepaths_byday_df.loc[eeg_filepaths_byday_df.EEG_Day==day, 'EEG_Day'] = day-1
            pass
        pass
        
        return eeg_filepaths_byday_df

    def get_sleep_stages_duration_sec_byday(self, eeg_filepaths_byday_df:pd.DataFrame=None)->dict:
        """
        Calculate and print the total duration of each sleep stage for the patient.

        Parameters:
        None

        Returns:
        dict: A dictionary where keys are sleep stage names and values are the total duration of each sleep stage in seconds.
        """

        eeg_files_ls = eeg_filepaths_byday_df.EEG_Filepath.values.tolist()
        eeg_days_ls = eeg_filepaths_byday_df.EEG_Day.values.tolist()

        # Count the seconds of each sleep stage for each day
        new_dict = lambda : {v:0 for v in self.sleep_stages_map.values()}
        sleep_stage_scounter_byday_dict = {i:new_dict() for i in range(np.min(eeg_days_ls), np.max(eeg_days_ls)+1)}
        for i,_ in enumerate(eeg_files_ls):
            this_pat_eeg_fpath = eeg_files_ls[i]
            eeg_day = eeg_days_ls[i]
            pat_sleep_data_path = self.sleep_data_path / this_pat_eeg_fpath.name.replace(self.eeg_file_extension, '_ScalpSleepStages.csv')
            sleep_data_df = pd.read_csv(pat_sleep_data_path, skiprows=7)
            for sleep_key in self.sleep_stages_map.keys():
                sleep_stage_scounter_byday_dict[eeg_day][self.sleep_stages_map[sleep_key]] += np.sum(sleep_data_df.I1_1==sleep_key)
                pass
                
        # create a dictionary with the total duration of each sleep stage for each day
        print(f"Days analyzed = {np.min(eeg_days_ls)} to {np.max(eeg_days_ls)}")
        stages_duration_byday = {'DayNr':[], **{v:[] for v in self.sleep_stages_map.values()}}
        for k,v in sleep_stage_scounter_byday_dict.items():
            print(f"\nDay {k}, Sleep Stages Duration (s)")
            stages_duration_byday['DayNr'].append(k)
            for k2,v2 in v.items():
                print(f"{k2}={v2} seconds")
                stages_duration_byday[k2].append(v2)
                pass
            pass

        return stages_duration_byday
 

    def get_detailes_spike_events(self, eeg_fpath, eeg_reader):
        # Read sleep data and spike detections
        sleep_data_df = self.read_sleep_stages_data(eeg_fpath)
        spike_data_df = self.read_spike_data(eeg_fpath).sort_values(by=['Time'], ascending=True)

        if len(spike_data_df)==0:
            return None, None

        #  Get the indices of the spike windows
        spk_wdw_dur_s = 1
        spikes_polarity_vec = spike_data_df['Sign'].to_numpy()
        spikes_center_sec_vec = spike_data_df['Time'].to_numpy()
        spikes_center_samples = (spikes_center_sec_vec*eeg_reader.fs).astype(int)
        spikes_start_samples = ((spikes_center_sec_vec-(spk_wdw_dur_s/2))*eeg_reader.fs).astype(int)
        spikes_end_samples = ((spikes_center_sec_vec+(spk_wdw_dur_s/2))*eeg_reader.fs).astype(int)
        
        # Delete invalid spike indices
        spikes_to_keep = np.logical_not(np.logical_or(spikes_start_samples<0, spikes_end_samples>=eeg_reader.n_samples))
        spikes_polarity_vec = spikes_polarity_vec[spikes_to_keep]
        spikes_center_sec_vec = spikes_center_sec_vec[spikes_to_keep]
        spikes_center_samples = spikes_center_samples[spikes_to_keep]
        spikes_start_samples = spikes_start_samples[spikes_to_keep]
        spikes_end_samples = spikes_end_samples[spikes_to_keep]

        # Get the sleep stage of each spike
        spike_sleep_stage_code = np.array([sleep_data_df.I1_1[sleep_data_df.Time==int(np.round(sc_sec))].to_numpy()[0] for sc_sec in spikes_center_sec_vec])
        spikes_to_keep = np.logical_not(np.isnan(spike_sleep_stage_code))
        spike_sleep_stage_code = spike_sleep_stage_code[spikes_to_keep]
        spikes_polarity_vec = spikes_polarity_vec[spikes_to_keep]
        spikes_center_sec_vec = spikes_center_sec_vec[spikes_to_keep]
        spikes_center_samples = spikes_center_samples[spikes_to_keep]
        spikes_start_samples = spikes_start_samples[spikes_to_keep]
        spikes_end_samples = spikes_end_samples[spikes_to_keep]
        spike_sleep_stage_name = np.array([self.sleep_stages_map[int(ss_code)] for ss_code in spike_sleep_stage_code])

        # Create list of ranges
        ranges = [(start, end) for start, end in zip(spikes_start_samples, spikes_end_samples)]
        # Create list of indices
        spike_wdw_indices = np.r_[tuple(slice(start, end) for start, end in ranges)]

        spikes_info_df = pd.DataFrame({
                            'stage_code':spike_sleep_stage_code,
                            'stage_name':spike_sleep_stage_name, 
                            'polarity':spikes_polarity_vec, 
                            'center_sec':spikes_center_sec_vec, 
                            'center_sample':spikes_center_samples, 
                            'start_sample':spikes_start_samples, 
                            'end_sample':spikes_end_samples}
                            )

        return spike_wdw_indices, spikes_info_df


    def get_channel_avg_wdw_vectorized(self, this_pat_eeg_fpath, mtg_t:str='ir', force_recalc:bool=False)->AvgWdwCumulator:
        """
        This function cumulates for each channel and sleep stage, the windows that temporally coincide with a spike event coming from any channel.

        Parameters:
        mtg_t (str): The montage type to use for EEG data reading. Default is intracranial referential='ir'.
        plot_ok (bool): A flag indicating whether to plot the EEG segments containing spikes. Default is False.

        Returns:
        None
        """

        eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_fpath, mtg_t=mtg_t)
        spike_cumulator_fn = self.output_path / f"CumulatedSpikes/{eeg_reader.filename.replace(".dat", '_AvgWdwCumulator.pickle')}"
        if not os.path.isfile(spike_cumulator_fn) or force_recalc:
            print(this_pat_eeg_fpath.name)
            sleep_stages_ls = list(self.sleep_stages_map.values())
            self.spike_cumulator = AvgWdwCumulator(eeg_channels_ls=eeg_reader.ch_names, sleep_stage_ls=sleep_stages_ls, sig_wdw_dur_s=1, sig_wdw_fs=64)
            fs_us = self.spike_cumulator.get_undersampling_frequency()

            spike_wdw_indices, spk_df = self.get_detailes_spike_events(this_pat_eeg_fpath, eeg_reader)
            if spike_wdw_indices is None:
                return

            nr_total_spikes = len(spk_df.center_sample)
            all_ch_sigs = eeg_reader.get_data()
            for eeg_chi, ch_name in enumerate(eeg_reader.ch_names):
                ch_spike_wdws = all_ch_sigs[eeg_chi][spike_wdw_indices].reshape(nr_total_spikes, int(eeg_reader.fs))
                
                # Undersample the EEG segment containing the spikes
                undersampled_spike_wdws = np.zeros((ch_spike_wdws.shape[0], fs_us))
                for i, spike_wdw in enumerate(ch_spike_wdws):
                    undersampled_spike_wdws[i] = self.undersample_signal(spike_wdw, fs_us)

                for k, stage_name in self.sleep_stages_map.items():
                    stage_indices = np.where(spk_df.stage_name==stage_name)
                    if len(stage_indices[0])>0:
                        stage_spike_wdws = np.abs(undersampled_spike_wdws[stage_indices])
                        avg_spike = np.mean(stage_spike_wdws, axis=0)
                        nr_spikes_in_avg = stage_spike_wdws.shape[0]
                        self.spike_cumulator.set_nr_spikes_in_avg(sleep_stage=stage_name, nr_spikes_in_avg=nr_spikes_in_avg)
                        self.spike_cumulator.add_spike(stage_name, ch_name, avg_spike)
                        pass

                pass
            
        self.save_spike_cumulator(filepath=spike_cumulator_fn)
        pass

    def get_avg_wdw_by_day_by_ch(self, eeg_filepaths_byday_df, mtg_t, force_recalc):

        eeg_files_ls = eeg_filepaths_byday_df.EEG_Filepath.values.tolist()
        chanels_ls = []
        for eeg_fpath in eeg_files_ls:
            eeg_reader = EEG_IO(eeg_fpath, mtg_t=mtg_t)
            chanels_ls.extend(eeg_reader.ch_names)
            pass
        chanels_ls = list(set(chanels_ls)) 
        
        sleep_stages_ls = list(self.sleep_stages_map.values())

        avg_spike_by_day_stage_ch_df = []
        for sleep_stage in sleep_stages_ls:
            days_ls = eeg_filepaths_byday_df.EEG_Day.unique()
            for day in days_ls:
                spike_cumulator_fpath = self.output_path / f"CumulatedSpikes/{eeg_files_ls[0].name.replace(".lay", '_AvgWdwCumulator.pickle')}"
                fs_us = self.load_spike_cumulator(spike_cumulator_fpath).sig_wdw_fs
                day_stage_cum = {ch_name:np.zeros(fs_us) for ch_name in chanels_ls}
                day_stage_cum = {**{'clip_cnt':[0], 'spike_cnt':[0]}, **day_stage_cum}
                
                day_eeg_files_ls = eeg_filepaths_byday_df.EEG_Filepath[eeg_filepaths_byday_df.EEG_Day==day].values.tolist()
                for eeg_fpath in day_eeg_files_ls:
                    #eeg_reader = EEG_IO(eeg_fpath, mtg_t=mtg_t)
                    spike_cumulator_fpath = self.output_path / f"CumulatedSpikes/{eeg_fpath.name.replace(".lay", '_AvgWdwCumulator.pickle')}"

                    if not os.path.isfile(spike_cumulator_fpath):
                        continue
                    temp_spk_cum = self.load_spike_cumulator(filepath=spike_cumulator_fpath)

                    nr_spikes = temp_spk_cum.spike_counter[sleep_stage][0]
                    day_stage_cum['clip_cnt'][0] += 1
                    day_stage_cum['spike_cnt'][0] += nr_spikes
                    for ch_name in chanels_ls:
                        ch_idx = temp_spk_cum.get_ch_idx(ch_name)
                        day_stage_cum[ch_name] += temp_spk_cum.spike_cum_dict[sleep_stage][ch_idx]*nr_spikes
                        pass
                    pass
            
                nr_chs = len(chanels_ls)
                nr_records = day_stage_cum['clip_cnt'][0]
                nr_total_spikes = day_stage_cum['spike_cnt'][0]
                data_df = {'Stage':[sleep_stage]*nr_chs, 'DayNr':[day]*nr_chs, 'NrHourRecords':[nr_records]*nr_chs, 'NrSpikeWdws':[nr_total_spikes]*nr_chs, 'ChName':chanels_ls, 'AvgSpikeAmplitude':[]}
                plot_ok = False
                for ch_name in chanels_ls:
                    if nr_total_spikes>0:
                        day_stage_cum[ch_name] /= nr_total_spikes
                    data_df['AvgSpikeAmplitude'].append(np.max(day_stage_cum[ch_name])-np.min(day_stage_cum[ch_name]))
                    if plot_ok:
                        time_vec = np.arange(len(day_stage_cum[ch_name]))/fs_us
                        plt.plot(time_vec, day_stage_cum[ch_name], '-k', linewidth=1)
                        plt.plot([np.mean(time_vec)]*2, [np.min(day_stage_cum[ch_name]), np.max(day_stage_cum[ch_name])], '--r', linewidth=1)
                        plt.xlim(np.min(time_vec), np.max(time_vec))
                        plt.waitforbuttonpress()
                        plt.close()
                data_df = pd.DataFrame(data_df)
                if len(avg_spike_by_day_stage_ch_df)==0:
                    avg_spike_by_day_stage_ch_df = data_df.copy()
                else:
                    avg_spike_by_day_stage_ch_df = pd.concat([avg_spike_by_day_stage_ch_df, data_df], ignore_index=True)
                pass
        return avg_spike_by_day_stage_ch_df
                

    def get_channel_coordinates(self, avg_spike_by_day_stage_ch_df):
        spike_dets_chnames = avg_spike_by_day_stage_ch_df.ChName.unique().tolist()
        # Load channel coordinates
        pat_id = self.pat_id
        ch_coords_fn = (''.join([c for c in pat_id if c.isdigit()]))+'_elecInfo.csv'
        coords_fpath = self.ch_coordinates_data_path/ ch_coords_fn
        ch_coords_data = pd.read_csv(coords_fpath)[['name', 'x', 'y', 'z']]

        # Check if all channels used for the spike detection can be given 3D localization coordinates
        assert np.sum([c.lower() in ch_coords_data.name.str.lower().to_list() for c in spike_dets_chnames]) == len(spike_dets_chnames)

        ch_coords_dict = defaultdict(set)#{chname.lower():(0,0,0) for chname in spike_dets_chnames}
        for chname in spike_dets_chnames:
            sel_ch_coords_data = ch_coords_data[ch_coords_data.name.str.fullmatch(chname.lower(), case=False)].reset_index(drop=True)

            # Check that only one set of coordinateds is being selected for the specific channel
            assert len(sel_ch_coords_data)==1

            x_coord=0
            y_coord=0
            z_coord=0
            try:
                x_coord = float(sel_ch_coords_data.x[0])
                y_coord = float(sel_ch_coords_data.y[0])
                z_coord = float(sel_ch_coords_data.z[0])

            except:
                print(f"Invalid coordinates in channel: {chname}")
                continue
            coords_are_nonzero = x_coord>0 and y_coord>0 and z_coord>0
            ch_coords_dict[sel_ch_coords_data.name[0].lower()] = [x_coord, y_coord, z_coord]

        avg_spike_by_day_stage_ch_df['x'] = pd.Series(np.zeros(len(avg_spike_by_day_stage_ch_df))-1, dtype='float')
        avg_spike_by_day_stage_ch_df['y'] = pd.Series(np.zeros(len(avg_spike_by_day_stage_ch_df))-1, dtype='float')
        avg_spike_by_day_stage_ch_df['z'] = pd.Series(np.zeros(len(avg_spike_by_day_stage_ch_df))-1, dtype='float')
        #avg_spike_by_day_stage_ch_df['xyz'] = avg_spike_by_day_stage_ch_df.ChName.str.lower().map(lambda x[0]: ch_coords_dict[x])
        for chname in ch_coords_dict.keys():
            sel_rows = avg_spike_by_day_stage_ch_df.ChName.str.fullmatch(chname, case=False)
            if len(sel_rows)>0:
                avg_spike_by_day_stage_ch_df.loc[sel_rows, 'x'] = ch_coords_dict[chname][0]
                avg_spike_by_day_stage_ch_df.loc[sel_rows, 'y'] = ch_coords_dict[chname][1]
                avg_spike_by_day_stage_ch_df.loc[sel_rows, 'z'] = ch_coords_dict[chname][2]
                pass

        del_rows = avg_spike_by_day_stage_ch_df[avg_spike_by_day_stage_ch_df.x==-1].index
        avg_spike_by_day_stage_ch_df.drop(del_rows, inplace=True)
        #avg_spike_by_day_stage_ch_df = avg_spike_by_day_stage_ch_df[avg_spike_by_day_stage_ch_df.xyz.apply(len)>0]

        return avg_spike_by_day_stage_ch_df
    

    def parse_szr_info_file(self, szr_info_fpath):
        szr_info_df = pd.read_csv(szr_info_fpath)
        szr_info_df = szr_info_df[['vig.', 'origin']].reset_index()
        ch_szr_involvment_dict = {'Origin': [], 'Early': [], 'Late': []}
        for i in np.arange(len(szr_info_df)):
            szr_info_str = szr_info_df.at[i, 'origin']
            szr_info_str = szr_info_str.replace(' ', '')

            origin_idx = szr_info_str.lower().find('origin')
            early_idx = szr_info_str.lower().find('early')
            late_idx = szr_info_str.lower().find('late')
            origin_channs_start = origin_idx+7
            origin_channs_end = early_idx

            early_channs_start = early_idx+6
            early_channs_end = late_idx

            late_channs_start = late_idx+5
            late_channs_end = len(szr_info_str)

            origin_channs = szr_info_str[origin_channs_start:origin_channs_end]
            early_channs = szr_info_str[early_channs_start:early_channs_end]
            late_channs = szr_info_str[late_channs_start:late_channs_end]

            origin_channs_ls = origin_channs.split(",")
            early_channs_ls = early_channs.split(",")
            late_channs_ls = late_channs.split(",")

            if len(origin_channs_ls[0]) > 0:
                ch_szr_involvment_dict['Origin'].extend(origin_channs_ls)
            if len(early_channs_ls[0]) > 0:
                ch_szr_involvment_dict['Early'].extend(early_channs_ls)
            if len(late_channs_ls[0]) > 0:
                ch_szr_involvment_dict['Late'].extend(late_channs_ls)

        pass

        soz_chann_ls = list(set(ch_szr_involvment_dict['Origin']))
        soz_chann_ls = [c.lower() for c in soz_chann_ls]
        assert len(soz_chann_ls)>0, "No SOZ channels found in the seizure info file"

        if '1096' in szr_info_fpath.name:
            soz_chann_ls = correct_1096_chnames(soz_chann_ls)
        
        soz_chann_ls = EEG_IO.clean_channel_labels(None, soz_chann_ls)

        return soz_chann_ls

    def get_soz_info(self, avg_spike_by_day_stage_ch_df):
        # Load channel coordinates
        pat_id = self.pat_id
        szr_info_fn = (''.join([c for c in pat_id if c.isdigit()]))+'_clinicalSzrInfo.csv'
        szr_info_fpath = self.szr_info_data_path/ szr_info_fn
        soz_chann_ls = self.parse_szr_info_file(szr_info_fpath)

        soz_chann_ls = [c.lower() for c in soz_chann_ls]
        avg_spike_by_day_stage_ch_df['SOZ'] = avg_spike_by_day_stage_ch_df.ChName.str.lower().map(lambda x: x in soz_chann_ls)

        assert len(avg_spike_by_day_stage_ch_df['SOZ'].unique())>1, "No channels could be assigned to a SOZ"

        return avg_spike_by_day_stage_ch_df

    def plot_daily_spike_activity(self):
        spk_df = pd.read_csv(self.output_path / f"{self.pat_id}_AvgSpikeWdwByDay.csv")
        stages_names_ls = list(self.sleep_stages_map.values())
        pat_id = self.pat_id
        days_ls = spk_df.DayNr.unique()

        # Create a 3D scatter plot for each Sleep Stage    
        fig = make_subplots(
                    rows=1, cols=6,
                    horizontal_spacing = 0.01,  vertical_spacing  = 0.1,
                    subplot_titles=(stages_names_ls),
                    start_cell="top-left",
                    specs=[
                        [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                        ]
                    )

        for day in days_ls:
            for ss_idx, stage_name in enumerate(stages_names_ls):
                stage_df = spk_df[spk_df.DayNr==day & spk_df.Stage.str.fullmatch(stage_name, case=False)]

                chavg_spike_chname = stage_df.ChName.to_numpy()
                chavg_spike_ampl = self.MinMaxScaler(stage_df.AvgSpikeAmplitude.to_numpy())
                x_coords = stage_df.x.to_numpy()
                y_coords = stage_df.y.to_numpy()
                z_coords = stage_df.z.to_numpy()

                #Count
                fig.add_trace(
                    go.Scatter3d(
                        x = x_coords,
                        y = y_coords,
                        z = z_coords,
                        mode='markers',  # Show markers and labels
                        marker=dict(
                            size=10,
                            color=chavg_spike_ampl,
                            colorscale='viridis',
                            opacity=0.8,
                            showscale=True,
                            cmin=np.min(chavg_spike_ampl),
                            cmax=np.max(chavg_spike_ampl),
                            colorbar=dict(title="Scaled<br>Amplitude", len=0.50, y=0.8),
                        ),
                        #text=stage_df['Channel'],
                        text=[f"{chname}: {prob:.2f}" for chname, amp, prob in zip(chavg_spike_chname, chavg_spike_ampl, chavg_spike_ampl)],  # Custom text for the fourth dimension
                        hoverinfo='text',
                        # hovertemplate="X: %{x}<br>Y: %{y}<br>%{text}<extra></extra>"  # Display fourth dimension on hover
                    ),
                    row=1, col=ss_idx+1
                )

            # Add SOZ channels
            for ss_idx, stage_name in enumerate(stages_names_ls):
                stage_df = spk_df[spk_df.DayNr==day & spk_df.Stage.str.fullmatch(stage_name, case=False)]

                chavg_spike_chname = stage_df.ChName.to_numpy()
                chavg_spike_ampl = self.MinMaxScaler(stage_df.AvgSpikeAmplitude.to_numpy())
                coords_weights = (chavg_spike_ampl/np.max(chavg_spike_ampl))
                #coords_weights = np.pow(coords_weights,2)
                #coords_weights = MinMaxScaler().fit_transform(coords_weights.reshape(-1,1)).flatten()
                
                x_coords = stage_df.x.to_numpy()
                y_coords = stage_df.y.to_numpy()
                z_coords = stage_df.z.to_numpy()

                max_ampl_x = x_coords[np.argmax(chavg_spike_ampl)]
                max_ampl_y = z_coords[np.argmax(chavg_spike_ampl)]
                max_ampl_z = y_coords[np.argmax(chavg_spike_ampl)]

                x_strides = (max_ampl_x-x_coords)*coords_weights
                avg_x_stride = np.mean(x_strides)
                final_x = max_ampl_x-avg_x_stride

                y_strides = (max_ampl_y-y_coords)*coords_weights
                avg_y_stride = np.mean(y_strides)
                final_y = max_ampl_y-avg_y_stride

                z_strides = (max_ampl_z-z_coords)*coords_weights
                avg_z_stride = np.mean(z_strides)
                final_z = max_ampl_z-avg_z_stride

                weighted_x = [int(final_x)]
                weighted_y = [int(final_y)]
                weighted_z = [int(final_z)]

                #Count
                fig.add_trace(
                    go.Scatter3d(
                        x = weighted_x,
                        y = weighted_y,
                        z = weighted_z,
                        mode='markers',  # Show markers and labels
                        marker=dict(
                            symbol="circle-open",
                            size=10,
                            color='Red',
                            colorscale='viridis',
                            opacity=0.8,
                            showscale=True,
                            cmin=np.min(chavg_spike_ampl),
                            cmax=np.max(chavg_spike_ampl),
                            colorbar=dict(title="Scaled<br>Amplitude", len=0.50, y=0.8),
                            line=dict(color='MediumPurple',width=2)
                        ),
                    ),
                    row=1, col=ss_idx+1
                )

            fig.update_layout(title_text=f"{pat_id}<br>(Avg. Window Amplitude)", showlegend=False)

            fig.layout.scene1.camera.eye=dict(x=2.4, y=2.4, z=2.4)
            fig.layout.scene2.camera.eye=dict(x=2.4, y=2.4, z=2.4)
            fig.layout.scene3.camera.eye=dict(x=2.4, y=2.4, z=2.4)
            fig.layout.scene4.camera.eye=dict(x=2.4, y=2.4, z=2.4)
            fig.layout.scene5.camera.eye=dict(x=2.4, y=2.4, z=2.4)

            fig.show()
            # fig_fpath = spike_cumulators_path /"Images" / f"Spike_Wdw_Amplitude_{pat_id}.html"
            # fig.write_html(fig_fpath)
            pass

        return fig