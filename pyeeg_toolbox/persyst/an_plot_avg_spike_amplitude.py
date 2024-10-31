import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from pathlib import Path
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from sklearn.preprocessing import MinMaxScaler
from studies_info import fr_four_patients


FIGSIZE = (16, 8)
STAGES_COLORS = {'N1':(250,223,99), 'N2':(41,232,178), 'N3':(76,169,238), 'REM':(47,69,113), 'Wake':(224,115,120), 'Unknown':(128,128,128)}


def plot_sleep_stage_durations(study_info, spike_cumulators_path):
    
    plt.style.use('seaborn-v0_8-darkgrid')
    nr_pats = len(study_info.patients.keys())
    fig, axs = plt.subplots(2, nr_pats, figsize=FIGSIZE)
    plt.get_current_fig_manager().full_screen_toggle()
    plt.suptitle(f"Sleep-Stage Duration by Patient")

    wedgeprops = {"edgecolor" : "black", 'linewidth': 0.5, 'antialiased': True}

    sleep_ref_img_path = os.getcwd()+'/pyeeg_toolbox/persyst/SleepStages_Reference.png'
    sleep_ref_img= Image.open(sleep_ref_img_path)
    rsz_ratio = 4
    img_rsz = (int(sleep_ref_img.size[0]/rsz_ratio), int(sleep_ref_img.size[1]/rsz_ratio))
    sleep_ref_img = sleep_ref_img.resize(img_rsz)

    for pidx, pat_id in enumerate(study_info.patients.keys()):

        print(pat_id)
        pat_data_path = study_info.eeg_data_path / pat_id
        spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
        spike_amplitude_analyzer.get_files_in_folder(file_extension='.lay')

        # Get duration of sleep stages
        sleep_stage_secs_counter_dict = spike_amplitude_analyzer.get_sleep_stages_duration_sec()

        # Plot Sleep Stages only
        to_plot_stage_names = ['N3', 'N2', 'N1', 'REM'] # "Wake", "Unknown"
        to_plot_stages_colors = [np.array(STAGES_COLORS[k])/255 for k in to_plot_stage_names]
        stages_dur_hours = [sleep_stage_secs_counter_dict[k]/3600 for k in to_plot_stage_names]
        stages_dur_perc = (stages_dur_hours/np.sum(stages_dur_hours))
        axs[0,pidx].pie(x=stages_dur_perc, labels=to_plot_stage_names, colors=to_plot_stages_colors, wedgeprops=wedgeprops, autopct='%.0f%%', startangle=180)
        if pidx == 0:
            axs[0,pidx].set_ylabel('Duration (%)')
        axs[0,pidx].set_title(f'{pat_id}\nSleep Stages')
        plt.tight_layout()

        # Plot all detected stages
        to_plot_stage_names = ['N3', 'N2', 'N1', 'REM', 'Wake', 'Unknown']
        to_plot_stages_colors = [np.array(STAGES_COLORS[k])/255 for k in to_plot_stage_names]
        stages_dur_hours = [sleep_stage_secs_counter_dict[k]/3600 for k in to_plot_stage_names]
        stages_dur_perc = (stages_dur_hours/np.sum(stages_dur_hours))
        sns.barplot(x=to_plot_stage_names, y=stages_dur_hours, hue=range(len(stages_dur_hours)), legend=False, palette=to_plot_stages_colors, ax=axs[1,pidx])
        if pidx == 0:
            axs[1,pidx].set_ylabel('Duration (hours)')
        axs[1,pidx].set_title(f'{pat_id}\nAll Stages')
        plt.tight_layout()

        pass


    # Overlay image on plot
    im_width, im_height = sleep_ref_img.size
    bbox = fig.get_window_extent() 
    fig.figimage(sleep_ref_img, xo=int(bbox.x1+im_width), yo=int(bbox.y1+im_height/2), zorder=3, alpha=.7, origin='upper')
    plt.tight_layout()

    plt.waitforbuttonpress()
    plt.close()

def plot_avg_spike_rate_per_stage_per_patient(study_info, spike_cumulators_path):

    to_plot_stage_names = ['N3', 'N2', 'N1', 'REM', 'Wake', 'Unknown']    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(1, len(study_info.patients.keys()), figsize=FIGSIZE)

    for pat_idx, pat_id in enumerate(study_info.patients.keys()):

        print(pat_id)
        pat_data_path = study_info.eeg_data_path / pat_id
        spike_cumulator_fn = spike_cumulators_path / f"{pat_id}_SpikeCumulator.pickle"
        spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
        spike_amplitude_analyzer.get_files_in_folder(file_extension='.lay')

        # Get duration of sleep stages
        sleep_stage_secs_counter_dict = spike_amplitude_analyzer.get_sleep_stages_duration_sec()
        spike_cumulator = spike_amplitude_analyzer.load_spike_cumulator(filepath=spike_cumulator_fn)

        sleep_stages = spike_cumulator.sleep_stage_ls
        spike_rate_per_stage_dict = defaultdict(float)
        for stage_name in sleep_stages:
            stage_dur_m = sleep_stage_secs_counter_dict[stage_name]/60
            stage_spike_cnt = np.sum(spike_cumulator.spike_counter[stage_name])
            stage_spike_rate = stage_spike_cnt/stage_dur_m
            spike_rate_per_stage_dict[stage_name] = stage_spike_rate


        # Plot all detected stages
        to_plot_stages_colors = [np.array(STAGES_COLORS[k])/255 for k in to_plot_stage_names]
        spike_rates = [spike_rate_per_stage_dict[k] for k in to_plot_stage_names]
        sns.barplot(x=to_plot_stage_names, y=spike_rates, hue=range(len(to_plot_stage_names)), legend=False, palette=to_plot_stages_colors, ax=axs[pat_idx])
        axs[pat_idx].set_xlabel('Sleep Stage')
        axs[pat_idx].set_ylabel('Occ.Rate/minute')
        axs[pat_idx].set_title(f'{pat_id}')
        plt.tight_layout()

    plt.get_current_fig_manager().full_screen_toggle()
    plt.suptitle(f"Occ.Rate of Spikes in any channel\nGrouped by Sleep-Stages and Patients")
    plt.waitforbuttonpress()
    plt.close() 

def plot_avg_spike_waveform_per_stage_per_patient(study_info, spike_cumulators_path):

    to_plot_stage_names = ['N3', 'N2', 'N1', 'REM', 'Wake', 'Unknown']
    plt.style.use('seaborn-v0_8-darkgrid')
    sp_nr_rows = len(study_info.patients.keys())
    sp_nr_cols = len(to_plot_stage_names)
    fig, axs = plt.subplots(sp_nr_rows, sp_nr_cols, figsize=FIGSIZE)

    for pat_idx, pat_id in enumerate(study_info.patients.keys()):

        print(pat_id)
        pat_data_path = study_info.eeg_data_path / pat_id
        spike_cumulator_fn = spike_cumulators_path / f"{pat_id}_SpikeCumulator.pickle"
        spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
        spike_amplitude_analyzer.get_files_in_folder(file_extension='.lay')

        # Get duration of sleep stages
        spike_cumulator = SpikeAmplitudeAnalyzer().load_spike_cumulator(filepath=spike_cumulator_fn)

        sleep_stages = spike_cumulator.sleep_stage_ls
        channel_names = spike_cumulator.eeg_channels_ls

        # get avg_spike_by_stage
        avg_spike_by_stage = {s:np.zeros_like(spike_cumulator.spike_cum_dict[sleep_stages[0]][0]) for s in sleep_stages}
        for stidx, stage_name in enumerate(sleep_stages):
            stage_spike_cnt = np.sum(spike_cumulator.spike_counter[stage_name])
            stage_avg_spike = np.sum(spike_cumulator.spike_cum_dict[stage_name], axis=0)/stage_spike_cnt
            stage_avg_spike *= 1000*1000
            avg_spike_by_stage[stage_name] = stage_avg_spike


        volt_min = 1*10**10
        volt_max = -1*10**10

        # Determine which stage has highest amplitude
        line_color_ls = ['k']*len(sleep_stages)
        amp_ptp_ls = []
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            amp_ptp = np.max(stage_avg_spike) - np.min(stage_avg_spike)
            amp_ptp_ls.append(amp_ptp)

        line_color_ls[np.argmax(amp_ptp_ls)] = 'r'
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            stage_spike_cnt = np.sum(spike_cumulator.spike_counter[stage_name])

            amp_ptp = np.max(stage_avg_spike) - np.min(stage_avg_spike)
            time_vec = np.arange(len(stage_avg_spike))/spike_cumulator.sig_wdw_fs

            lc = line_color_ls[stidx]
            axs[pat_idx, stidx].plot(time_vec, stage_avg_spike, color=lc, linewidth=0.5)
            if stidx == 0:
                axs[pat_idx, stidx].set_ylabel(f'{pat_id}\nVoltage(uV)')
            if pat_idx == 0:
                axs[pat_idx, stidx].set_title(f'{stage_name}')
            text_str = f'Spikes:{int(stage_spike_cnt)}\nPtP:{amp_ptp:.1f}uV'
            axs[pat_idx, stidx].text(0.05, 0.95, text_str, transform=axs[pat_idx, stidx].transAxes, fontsize=8, va='top', ha='left')
            plt.tight_layout()
            if np.max(stage_avg_spike)>volt_max:
                volt_max = np.max(stage_avg_spike)
            if np.min(stage_avg_spike)<volt_min:
                volt_min = np.min(stage_avg_spike)

        # Modify y_lim so that it is equal for all stages
        for stidx, stage_name in enumerate(spike_cumulator.sleep_stage_ls):
            #axs[pat_idx, stidx].set_ylim(volt_min-np.abs(volt_min*0.1), volt_max+np.abs(volt_max*0.1))
            axs[pat_idx, stidx].set_ylim(volt_min, volt_max)
            axs[pat_idx, stidx].set_xlim(0, 1)

    plt.get_current_fig_manager().full_screen_toggle()
    plt.suptitle(f"Avg. Spike across Channels\n Grouped by Sleep-Stages and Patients")
    plt.tight_layout()
    plt.waitforbuttonpress()
    plt.close()

    pat_data_path = study_info.eeg_data_path / pat_id

def get_scaled_stage_avg_spike(spike_cumulator):

    fs = spike_cumulator.sig_wdw_fs
    channels_ls = spike_cumulator.eeg_channels_ls
    nr_chs = len(channels_ls)
    spike_wdw_len = int(spike_cumulator.sig_wdw_dur_s*fs)
    sleep_stages = spike_cumulator.sleep_stage_ls
    stages_ch_avg_spike = {stage:np.zeros(shape=(nr_chs, spike_wdw_len)) for stage in sleep_stages}

    # get per ch average spike
    for stage_name in sleep_stages:
        for chidx, ch_name in enumerate(channels_ls):
            ch_avg_spike = spike_cumulator.get_average_spike(stage_name, ch_name)
            stages_ch_avg_spike[stage_name][chidx] = ch_avg_spike
            pass
    
    # Get scaler for each channel avg spike based on the Wake stage
    #std_scalers_dict = [{s:''} for s in sleep_stages]
    std_scalers_ls = [MinMaxScaler() for ch in channels_ls]
    stage_name = 'Wake'
    for chidx, ch_name in enumerate(channels_ls):
        ch_avg_spike = stages_ch_avg_spike[stage_name][chidx]
        std_scalers_ls[chidx].fit(ch_avg_spike.reshape(-1,1))


    # Scale for each stage, each channel's average spike based on the Wake stage
    for stage_name in sleep_stages:
        for chidx, ch_name in enumerate(channels_ls):
            ch_avg_spike = (stages_ch_avg_spike[stage_name][chidx])
            scaled_avg_spike = std_scalers_ls[chidx].transform(ch_avg_spike.reshape(-1,1)).flatten()
            stages_ch_avg_spike[stage_name][chidx] = scaled_avg_spike
            pass

    return stages_ch_avg_spike


def plot_chscaled_avg_spike_waveform_per_stage_per_patient(study_info, spike_cumulators_path):

    to_plot_stage_names = ['N3', 'N2', 'N1', 'REM', 'Wake', 'Unknown']    
    plt.style.use('seaborn-v0_8-darkgrid')
    sp_nr_rows = len(study_info.patients.keys())
    sp_nr_cols = len(to_plot_stage_names)
    fig, axs = plt.subplots(sp_nr_rows, sp_nr_cols, figsize=FIGSIZE)

    for pat_idx, pat_id in enumerate(study_info.patients.keys()):

        print(pat_id)
        pat_data_path = study_info.eeg_data_path / pat_id
        spike_cumulator_fn = spike_cumulators_path / f"{pat_id}_SpikeCumulator.pickle"


        spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
        spike_amplitude_analyzer.get_files_in_folder(file_extension='.lay')

        # Get duration of sleep stages
        spike_cumulator = spike_amplitude_analyzer.load_spike_cumulator(filepath=spike_cumulator_fn)

        scaled_chavg_spike_by_stage = get_scaled_stage_avg_spike(spike_cumulator)

        sleep_stages = spike_cumulator.sleep_stage_ls
        channel_names = spike_cumulator.eeg_channels_ls
        nr_chs = len(channel_names)

        # get avg_spike_by_stage
        avg_spike_by_stage = {s:np.zeros_like(spike_cumulator.spike_cum_dict[sleep_stages[0]][0]) for s in sleep_stages}
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = np.mean(scaled_chavg_spike_by_stage[stage_name], axis=0) /nr_chs
            stage_avg_spike *= 1000
            avg_spike_by_stage[stage_name] = stage_avg_spike


        volt_min = 1*10**10
        volt_max = -1*10**10

        # Determine which stage has highest amplitude
        line_color_ls = ['k']*len(sleep_stages)
        amp_ptp_ls = []
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            amp_ptp = np.max(stage_avg_spike) - np.min(stage_avg_spike)
            amp_ptp_ls.append(amp_ptp)

        line_color_ls[np.argmax(amp_ptp_ls)] = 'r'
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            stage_spike_cnt = np.sum(spike_cumulator.spike_counter[stage_name])

            amp_ptp = np.max(stage_avg_spike) - np.min(stage_avg_spike)
            time_vec = np.arange(len(stage_avg_spike))/spike_cumulator.sig_wdw_fs

            lc = line_color_ls[stidx]
            axs[pat_idx, stidx].plot(time_vec, stage_avg_spike, color=lc, linewidth=0.5)
            if stidx == 0:
                axs[pat_idx, stidx].set_ylabel(f'{pat_id}\nVoltage(uV)')
            if pat_idx == 0:
                axs[pat_idx, stidx].set_title(f'{stage_name}')
            text_str = f'Spikes:{int(stage_spike_cnt)}\nPtP:{amp_ptp:.1f}uV'
            axs[pat_idx, stidx].text(0.05, 0.95, text_str, transform=axs[pat_idx, stidx].transAxes, fontsize=8, va='top', ha='left')
            plt.tight_layout()
            if np.max(stage_avg_spike)>volt_max:
                volt_max = np.max(stage_avg_spike)
            if np.min(stage_avg_spike)<volt_min:
                volt_min = np.min(stage_avg_spike)

        # Modify y_lim so that it is equal for all stages
        for stidx, stage_name in enumerate(spike_cumulator.sleep_stage_ls):
            #axs[pat_idx, stidx].set_ylim(volt_min-np.abs(volt_min*0.1), volt_max+np.abs(volt_max*0.1))
            axs[pat_idx, stidx].set_ylim(volt_min, volt_max)
            axs[pat_idx, stidx].set_xlim(0, 1)

    plt.get_current_fig_manager().full_screen_toggle()
    plt.suptitle(f"Avg. Spike across Channels\n Grouped by Sleep-Stages and Patients")
    plt.tight_layout()
    plt.waitforbuttonpress()
    plt.close()


                

def plot_average_spike_waveform_per_stage(spike_cumulator):
    sleep_stages = spike_cumulator.sleep_stage_ls
    channel_names = spike_cumulator.eeg_channels_ls
    for stage_name in spike_cumulator.sleep_stage_ls:
        stage_avg_spike = np.zeros((spike_cumulator.sig_wdw_dur_s*spike_cumulator.sig_wdw_fs))
        max_spike_rate_across_all_chs = np.max(spike_cumulator.spike_counter[stage_name])
        for chi, chname in enumerate(spike_cumulator.eeg_channels_ls):
            ch_cum_spike = spike_cumulator.spike_cum_dict[stage_name][chi]
            ch_nr_spikes = spike_cumulator.spike_counter[stage_name][chi]
            avg_ch_spike = ch_cum_spike / ch_nr_spikes
            stage_avg_spike += ch_cum_spike 
            pass

if __name__ == '__main__':


    # Define directory to save the cumulated spike signals
    spike_cumulators_path = Path(os.getcwd()) / "Output"
    plot_sleep_stage_durations(spike_cumulators_path)
    plot_avg_spike_rate_per_stage_per_patient(spike_cumulators_path)
    plot_avg_spike_waveform_per_stage_per_patient(spike_cumulators_path)

