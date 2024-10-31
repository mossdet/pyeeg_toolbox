import os
from pathlib import Path

from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from studies_info import fr_four_patients
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_sleep_stage_durations
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_avg_spike_rate_per_stage_per_patient
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_avg_spike_waveform_per_stage_per_patient
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_chscaled_avg_spike_waveform_per_stage_per_patient
from joblib import Parallel, delayed


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

results = Parallel(n_jobs=32)(delayed(analyze_patient)(study_info, pat_id) for pat_id in study_info.patients.keys())

# #Obtain average spike for each sleep stage and each patient
# for pat_id in study_info.patients.keys():
#     print(pat_id)
#     pat_data_path = study_info.eeg_data_path / pat_id
#     spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
#     spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', plot_ok=False)
#     spike_cumulator_fn = output_path / f"{pat_id}_SpikeCumulator.pickle"
#     spike_amplitude_analyzer.save_spike_cumulator(filepath=spike_cumulator_fn)

# # Define directory to save the cumulated spike signals

# plot_sleep_stage_durations(study_info, output_path)
# plot_chscaled_avg_spike_waveform_per_stage_per_patient(study_info, output_path)
# plot_avg_spike_waveform_per_stage_per_patient(study_info, output_path)
# plot_avg_spike_rate_per_stage_per_patient(study_info, output_path)

