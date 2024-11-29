import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from joblib import Parallel, delayed
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from pyeeg_toolbox.persyst.an_avg_wdw_amplitude import AverageWdwAnalyzer
from pyeeg_toolbox.persyst.an_spike_wdw_avg_by_day import DailySpikeWdwAnalyzer
from collections import defaultdict

from studies_info import fr_four_patients
import pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude as spk_plt
import pyeeg_toolbox.persyst.an_plot_avg_wdw_amplitude as wdw_plt

FORCE_RECALC = False

# Define directory to save the cumulated spike signals
output_path = Path(os.getcwd()) / "Output"
os.makedirs(output_path, exist_ok=True)

study_info = fr_four_patients()

def analyze_patient_spikes(study_info, pat_id):
    print(pat_id)
    pat_data_path = study_info.eeg_data_path / pat_id
    spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
    spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', plot_ok=True)
    spike_cumulator_fn = output_path / f"{pat_id}_SpikeCumulator.pickle"
    spike_amplitude_analyzer.save_spike_cumulator(filepath=spike_cumulator_fn)

def analyze_patient_windows(study_info, pat_id):
    print(pat_id)
    pat_data_path = study_info.eeg_data_path / pat_id
    spike_amplitude_analyzer = AverageWdwAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
    spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', plot_ok=False)
    spike_cumulator_fn = output_path / f"{pat_id}_WdwCumulator.pickle"
    spike_amplitude_analyzer.save_wdw_cumulator(filepath=spike_cumulator_fn)

def analyze_spike_wdws_by_day(study_info, pat_id):
    print(pat_id)
    pat_data_path = study_info.eeg_data_path / pat_id
    pat_coords_path = study_info.channel_coordinates_data_path
    pat_szr_info_path = study_info.seizure_info_data_path
    spike_amplitude_analyzer = DailySpikeWdwAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path, ch_coordinates_data_path=pat_coords_path, szr_info_data_path=pat_szr_info_path, output_path=output_path)
    #spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', force_recalc=FORCE_RECALC)
    #spike_amplitude_analyzer.plot_daily_spike_activity()
    #spike_amplitude_analyzer.get_wavg_coords_by_day()
    tot_spk_centroid_displacement = spike_amplitude_analyzer.get_daily_centroid_shift()
    #spike_amplitude_analyzer.plot_daily_centroid_shift()
    #spike_amplitude_analyzer.plot_daily_centroid_shift_with_SOZ()
    pass

# def plot_spike_avg_data(study_info, output_path):
#     # os.makedirs(f"{output_path}/Images/", exist_ok=True)
#     # spk_plt.plot_sleep_stage_durations(study_info, output_path)
#     # spk_plt.plot_avg_spike_rate_per_stage_per_patient(study_info, output_path)
#     # spk_plt.plot_avg_spike_waveform_per_stage_per_patient(study_info, output_path)
#     # spk_plt.plot_avg_spike_waveform_per_stage_per_patient_single_axis(study_info, output_path)
#     # spk_plt.plot_chscaled_avg_spike_waveform_per_stage_per_patient(study_info, output_path)
#     spk_plt.plot_avg_spike_amplitude_3D(study_info, output_path)

def plot_wdw_avg_data(study_info, output_path):
    os.makedirs(f"{output_path}/Images/", exist_ok=True)
    wdw_plt.plot_avg_wdw_waveform_per_stage_per_patient(study_info, output_path)
    wdw_plt.plot_avg_wdw_waveform_per_stage_per_patient_single_axis(study_info, output_path)
    wdw_plt.plot_avg_wdw_amplitude_3D(study_info, output_path)



# Process Data, get avg spike per channel and spike features (amplitude and frequency)
#results = Parallel(n_jobs=1)(delayed(analyze_patient_spikes)(study_info, pat_id) for pat_id in study_info.patients.keys())
#plot_spike_avg_data(study_info, output_path)


# output_path = Path(os.getcwd()) / "Wdw_Output"
# os.makedirs(output_path, exist_ok=True)
# # Analyze data and plot results
# results = Parallel(n_jobs=1)(delayed(analyze_patient_windows)(study_info, pat_id) for pat_id in study_info.patients.keys())
#plot_wdw_avg_data(study_info, output_path)

output_path = Path(os.getcwd()) / "WdwByDay_Output"
os.makedirs(output_path, exist_ok=True)
# Analyze data and plot results
#results = Parallel(n_jobs=1)(delayed(analyze_spike_wdws_by_day)(study_info, pat_id) for pat_id in study_info.patients.keys())
#plot_wdw_avg_data(study_info, output_path)


# Statistical Analysis of the spike centroid displacement across days
all_pats_dayily_centroid_shifts = defaultdict(list)
for pat_id in study_info.patients.keys():
    print(pat_id)
    pat_data_path = study_info.eeg_data_path / pat_id
    pat_coords_path = study_info.channel_coordinates_data_path
    pat_szr_info_path = study_info.seizure_info_data_path
    spike_amplitude_analyzer = DailySpikeWdwAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path, ch_coordinates_data_path=pat_coords_path, szr_info_data_path=pat_szr_info_path, output_path=output_path)
    #spike_amplitude_analyzer.plot_daily_centroid_shift()
    dayily_centroid_shifts = spike_amplitude_analyzer.get_daily_centroid_shift()
    
    pat_days_max = np.max(list(dayily_centroid_shifts.values()))
    pat_days_min = np.min(list(dayily_centroid_shifts.values()))
    for k in dayily_centroid_shifts.keys():
        for v in dayily_centroid_shifts[k]:
            #v = (v - pat_days_min) / (pat_days_max - pat_days_min)
            all_pats_dayily_centroid_shifts[k].append(v)        
        pass 

all_pats_dayily_centroid_shifts_df = pd.DataFrame(all_pats_dayily_centroid_shifts)
all_pats_dayily_centroid_shifts_df.to_csv(output_path / "All_pats_dayily_centroid_shifts.csv")
#sns.violinplot(data=all_pats_spk_displacement_df[['N3','N2','N1','REM','Wake']])
sns.boxplot(data=all_pats_dayily_centroid_shifts_df[['N3','N2','N1','REM','Wake']])
plt.waitforbuttonpress()
plt.close()
pass