import os
from pathlib import Path

from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from pyeeg_toolbox.studies.studies_info import StudiesInfo
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_sleep_stage_durations
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_avg_spike_rate_per_stage_per_patient
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_avg_spike_waveform_per_stage_per_patient
from pyeeg_toolbox.persyst.an_plot_avg_spike_amplitude import plot_chscaled_avg_spike_waveform_per_stage_per_patient


# Define directory to save the cumulated spike signals
output_path = Path(os.getcwd()) / "Output"
os.makedirs(output_path, exist_ok=True)

study = StudiesInfo()
study.fr_four_init()

#Obtain average spike for each sleep stage and each patient
for pat_id in study.study_patients.keys():
    print(pat_id)
    pat_data_path = study.eeg_data_path / pat_id
    spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
    spike_amplitude_analyzer.run(file_extension='.lay', mtg_t='ir', plot_ok=False)
    spike_cumulator_fn = output_path / f"{pat_id}_SpikeCumulator.pickle"
    spike_amplitude_analyzer.save_spike_cumulator(filepath=spike_cumulator_fn)

# Define directory to save the cumulated spike signals

plot_chscaled_avg_spike_waveform_per_stage_per_patient(output_path)
plot_avg_spike_waveform_per_stage_per_patient(output_path)

plot_sleep_stage_durations(output_path)
plot_avg_spike_rate_per_stage_per_patient(output_path)

