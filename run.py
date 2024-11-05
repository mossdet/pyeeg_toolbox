import os
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from studies_info import fr_four_patients
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import(
    plot_sleep_stage_durations,
    plot_avg_spike_waveform_per_stage_per_patient_single_axis,
    plot_avg_spike_rate_per_stage_per_patient,
    plot_avg_spike_waveform_per_stage_per_patient,
    plot_chscaled_avg_spike_waveform_per_stage_per_patient,
    plot_avg_spike_amplitude_3D,)


# Define directory to save the cumulated spike signals
output_path = Path(os.getcwd()) / "Output"
os.makedirs(output_path, exist_ok=True)

study_info = fr_four_patients()

def analyze_patient(study_info, pat_id):
    print(pat_id)
    pat_data_path = study_info.eeg_data_path / pat_id
    spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
    spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', plot_ok=False)
    spike_cumulator_fn = output_path / f"{pat_id}_SpikeCumulator.pickle"
    spike_amplitude_analyzer.save_spike_cumulator(filepath=spike_cumulator_fn)

def analyze_data(study_info, output_path):
    # os.makedirs(f"{output_path}/Images/", exist_ok=True)
    # plot_sleep_stage_durations(study_info, output_path)
    # plot_avg_spike_rate_per_stage_per_patient(study_info, output_path)
    # plot_avg_spike_waveform_per_stage_per_patient(study_info, output_path)
    # plot_avg_spike_waveform_per_stage_per_patient_single_axis(study_info, output_path)
    #plot_chscaled_avg_spike_waveform_per_stage_per_patient(study_info, output_path)
    plot_avg_spike_amplitude_3D(study_info, output_path)


# Process Data, get avg spike per channel and spike features (amplitude and frequency)
results = Parallel(n_jobs=1)(delayed(analyze_patient)(study_info, pat_id) for pat_id in study_info.patients.keys())

# Analyze data and plot results
analyze_data(study_info, output_path)