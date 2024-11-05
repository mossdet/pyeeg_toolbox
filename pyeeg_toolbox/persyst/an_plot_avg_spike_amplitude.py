import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
from PIL import Image
from pathlib import Path
from plotly.subplots import make_subplots
from pyeeg_toolbox.persyst.an_avg_spike_amplitude import SpikeAmplitudeAnalyzer
from sklearn.preprocessing import MinMaxScaler
from studies_info import fr_four_patients



import matplotlib
matplotlib.use('QtAgg')

FIGSIZE = (16, 8)
STAGES_COLORS = {'N1':(250,223,99), 'N2':(41,232,178), 'N3':(76,169,238), 'REM':(47,69,113), 'Wake':(224,115,120), 'Unknown':(128,128,128)}


def plot_sleep_stage_durations(study_info, spike_cumulators_path):
    
    plt.style.use('seaborn-v0_8-darkgrid')
    nr_pats = len(study_info.patients.keys())
    fig, axs = plt.subplots(2, nr_pats, figsize=FIGSIZE)

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

        # Plot all detected stages
        to_plot_stage_names = ['N3', 'N2', 'N1', 'REM', 'Wake', 'Unknown']
        to_plot_stages_colors = [np.array(STAGES_COLORS[k])/255 for k in to_plot_stage_names]
        stages_dur_hours = [sleep_stage_secs_counter_dict[k]/3600 for k in to_plot_stage_names]
        stages_dur_perc = (stages_dur_hours/np.sum(stages_dur_hours))
        sns.barplot(x=to_plot_stage_names, y=stages_dur_hours, hue=range(len(stages_dur_hours)), legend=False, palette=to_plot_stages_colors, ax=axs[1,pidx])
        if pidx == 0:
            axs[1,pidx].set_ylabel('Duration (hours)')
        axs[1,pidx].set_title(f'{pat_id}\nAll Stages')


    # Overlay image on plot
    im_width, im_height = sleep_ref_img.size
    bbox = fig.get_window_extent() 
    fig.figimage(sleep_ref_img, xo=int(bbox.x1+im_width), yo=int(bbox.y1+im_height/2), zorder=3, alpha=.7, origin='upper')

    plt.get_current_fig_manager().full_screen_toggle()
    plt.suptitle(f"Sleep-Stage Duration by Patient")
    plt.tight_layout()

    fig_fpath = f"{spike_cumulators_path}/Images/SleepStagesDuration_byPatient.png"
    plt.savefig(fig_fpath)

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
        # Load saved spike data
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

    plt.get_current_fig_manager().full_screen_toggle()
    plt.suptitle(f"Occ.Rate of Spikes in any channel\nGrouped by Sleep-Stages and Patients")
    plt.tight_layout()

    fig_fpath = f"{spike_cumulators_path}/Images/SpikeTotalOccRate_byPatient_bySleepStage.png"
    plt.savefig(fig_fpath)

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

        # Load saved spike data
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

        # Calculate Peak-Peak Amplitude for the avg. spike from each sleep stage
        line_color_dict = {stage:'k' for stage in sleep_stages}
        amp_ptp_dict = {stage:0 for stage in sleep_stages}
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            amp_ptp = np.max(stage_avg_spike) - np.min(stage_avg_spike)
            amp_ptp_dict[stage_name]=amp_ptp

        # Determine which stage has highest amplitude
        max_spike_ampl_stage = list(amp_ptp_dict.keys())[np.argmax(list(amp_ptp_dict.values()))]
        line_color_dict[max_spike_ampl_stage] = 'r'
        
        for stidx, stage_name in enumerate(to_plot_stage_names):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            stage_spike_cnt = np.sum(spike_cumulator.spike_counter[stage_name])

            amp_ptp = amp_ptp_dict[stage_name]
            time_vec = np.arange(len(stage_avg_spike))/spike_cumulator.sig_wdw_fs

            lc = line_color_dict[stage_name]
            axs[pat_idx, stidx].plot(time_vec, stage_avg_spike, color=lc, linewidth=0.5)
            if stidx == 0:
                axs[pat_idx, stidx].set_ylabel(f'{pat_id}\nVoltage(uV)')
            if pat_idx == 0:
                axs[pat_idx, stidx].set_title(f'{stage_name}')
            text_str = f'Spikes:{int(stage_spike_cnt)}\nPtP:{amp_ptp:.1f}uV'
            axs[pat_idx, stidx].text(0.05, 0.95, text_str, transform=axs[pat_idx, stidx].transAxes, fontsize=8, va='top', ha='left')
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

    fig_fpath = f"{spike_cumulators_path}/Images/AvgSpike_byPatient_bySleepStage.png"
    plt.savefig(fig_fpath)

    plt.waitforbuttonpress()
    plt.close()

    pat_data_path = study_info.eeg_data_path / pat_id


def plot_avg_spike_waveform_per_stage_per_patient_single_axis(study_info, spike_cumulators_path):

    to_plot_stage_names = ['N3', 'N2', 'N1', 'REM', 'Wake']
    plt.style.use('seaborn-v0_8-darkgrid')
    sp_nr_rows = len(study_info.patients.keys())
    sp_nr_cols = 1
    fig, axs = plt.subplots(sp_nr_rows, 1, figsize=FIGSIZE)

    for pat_idx, pat_id in enumerate(study_info.patients.keys()):

        print(pat_id)
        pat_data_path = study_info.eeg_data_path / pat_id
        spike_cumulator_fn = spike_cumulators_path / f"{pat_id}_SpikeCumulator.pickle"
        spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
        spike_amplitude_analyzer.get_files_in_folder(file_extension='.lay')

        # Load saved spike data
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

        # Calculate Peak-Peak Amplitude for the avg. spike from each sleep stage
        line_color_dict = {stage:'k' for stage in sleep_stages}
        amp_ptp_dict = {stage:0 for stage in sleep_stages}
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            amp_ptp = np.max(stage_avg_spike) - np.min(stage_avg_spike)
            amp_ptp_dict[stage_name]=amp_ptp

        for stidx, stage_name in enumerate(to_plot_stage_names):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            stage_spike_cnt = np.sum(spike_cumulator.spike_counter[stage_name])

            amp_ptp = amp_ptp_dict[stage_name]
            time_vec = np.arange(len(stage_avg_spike))/spike_cumulator.sig_wdw_fs

            lc = np.array(STAGES_COLORS[stage_name])/255 
            axs[pat_idx].plot(time_vec, stage_avg_spike, color=lc, linewidth=3, linestyle='-')
            if stidx == 0:
                axs[pat_idx].set_ylabel(f'{pat_id}\nVoltage(uV)')
            if pat_idx == 0:
                axs[pat_idx].set_title(f'{stage_name}')

            if np.max(stage_avg_spike)>volt_max:
                volt_max = np.max(stage_avg_spike)
            if np.min(stage_avg_spike)<volt_min:
                volt_min = np.min(stage_avg_spike)
        axs[pat_idx].legend(labels=to_plot_stage_names, loc='upper right')

        # Modify y_lim so that it is equal for all stages
        for stidx, stage_name in enumerate(spike_cumulator.sleep_stage_ls):
            #axs[pat_idx].set_ylim(volt_min-np.abs(volt_min*0.1), volt_max+np.abs(volt_max*0.1))
            axs[pat_idx].set_ylim(volt_min, volt_max)
            axs[pat_idx].set_xlim(0, 1)

    plt.get_current_fig_manager().full_screen_toggle()
    plt.suptitle(f"Avg. Spike across Channels\n Grouped by Sleep-Stages and Patients")
    plt.tight_layout()

    fig_fpath = f"{spike_cumulators_path}/Images/AvgSpike_byPatient_bySleepStage.png"
    plt.savefig(fig_fpath)

    plt.waitforbuttonpress()
    plt.close()

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

        # Load saved spike data
        spike_cumulator = SpikeAmplitudeAnalyzer().load_spike_cumulator(filepath=spike_cumulator_fn)

        # scale avg. spike by channel using Wake as reference
        scaled_chavg_spike_by_stage = get_scaled_stage_avg_spike(spike_cumulator)

        sleep_stages = spike_cumulator.sleep_stage_ls
        channel_names = spike_cumulator.eeg_channels_ls
        nr_chs = len(channel_names)

        # get avg_spike_by_stage
        avg_spike_by_stage = {s:np.zeros_like(spike_cumulator.spike_cum_dict[sleep_stages[0]][0]) for s in sleep_stages}
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = np.mean(scaled_chavg_spike_by_stage[stage_name], axis=0) /nr_chs
            stage_avg_spike *= 1000*1000
            avg_spike_by_stage[stage_name] = stage_avg_spike

        volt_min = 1*10**10
        volt_max = -1*10**10

        # Calculate Peak-Peak Amplitude for the avg. spike from each sleep stage
        line_color_dict = {stage:'k' for stage in sleep_stages}
        amp_ptp_dict = {stage:0 for stage in sleep_stages}
        for stidx, stage_name in enumerate(sleep_stages):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            amp_ptp = np.max(stage_avg_spike) - np.min(stage_avg_spike)
            amp_ptp_dict[stage_name]=amp_ptp

        # Determine which stage has highest amplitude
        max_spike_ampl_stage = list(amp_ptp_dict.keys())[np.argmax(list(amp_ptp_dict.values()))]
        line_color_dict[max_spike_ampl_stage] = 'r'

        for stidx, stage_name in enumerate(to_plot_stage_names):
            stage_avg_spike = avg_spike_by_stage[stage_name]
            stage_spike_cnt = np.sum(spike_cumulator.spike_counter[stage_name])

            amp_ptp = amp_ptp_dict[stage_name]
            time_vec = np.arange(len(stage_avg_spike))/spike_cumulator.sig_wdw_fs

            lc = line_color_dict[stage_name]
            axs[pat_idx, stidx].plot(time_vec, stage_avg_spike, color=lc, linewidth=0.5)
            if stidx == 0:
                axs[pat_idx, stidx].set_ylabel(f'{pat_id}\nVoltage(uV)')
            if pat_idx == 0:
                axs[pat_idx, stidx].set_title(f'{stage_name}')
            text_str = f'Spikes:{int(stage_spike_cnt)}\nPtP:{amp_ptp:.1f}uV'
            axs[pat_idx, stidx].text(0.05, 0.95, text_str, transform=axs[pat_idx, stidx].transAxes, fontsize=8, va='top', ha='left')
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
    plt.suptitle(f"Avg. Spike across Channels\n Scaled by Channel\n Grouped by Sleep-Stages and Patients")
    plt.tight_layout()

    fig_fpath = f"{spike_cumulators_path}/Images/AvgSpike_ChannelScaling_byPatient_bySleepStage.png"
    plt.savefig(fig_fpath)

    plt.waitforbuttonpress()
    plt.close()

def get_channel_coordinates(pat_id, study_info, spike_dets_chnames):
    # Load channel coordinates
    ch_coords_fn = (''.join([c for c in pat_id if c.isdigit()]))+'_elecInfo.csv'
    coords_fpath = study_info.channel_coordinates_data_path / ch_coords_fn
    ch_coords_data = pd.read_csv(coords_fpath)[['name', 'x', 'y', 'z']]

    # Check if all channels used for the spike detection can be given 3D localization coordinates
    assert np.sum([c in ch_coords_data.name.str.lower().to_list() for c in spike_dets_chnames]) == len(spike_dets_chnames)

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
        ch_coords_dict[sel_ch_coords_data.name[0].lower()] = (x_coord, y_coord, z_coord)
    return ch_coords_dict

def get_ch_avg_spike_features(pat_id, stage_names, spike_cumulator, ch_coords_dict):
    # get amplitude of ch_avg_spike for each channel and each sleep stage
    stage_specific_ch_avg_spike_data = []
    for stidx, stage_name in enumerate(stage_names):
        for ch_name in ch_coords_dict.keys():
            ch_avg_spike = spike_cumulator.get_average_spike(stage_name, ch_name)
            ch_avg_spike_amplitude = np.max(ch_avg_spike)-np.min(ch_avg_spike)
            entree_data = (pat_id, stage_name, ch_name, ch_avg_spike_amplitude, ch_coords_dict[ch_name])
            stage_specific_ch_avg_spike_data.append(entree_data)
            pass
    df = pd.DataFrame(data=stage_specific_ch_avg_spike_data, columns = ['PatID', 'SleepStage', 'Channel', 'ChAvgSpike_Amplitude', 'ChCoordinates'])
    return df

def plot_avg_spike_amplitude_3D(study_info, spike_cumulators_path):

    to_plot_stage_names = ['N3', 'N2', 'N1', 'REM', 'Wake']
    plt.style.use('seaborn-v0_8-darkgrid')

    for pat_idx, pat_id in enumerate(study_info.patients.keys()):
        sp_nr_rows = 1
        sp_nr_cols = len(to_plot_stage_names)
        fig, axs = plt.subplots(sp_nr_rows, sp_nr_cols, figsize=FIGSIZE)

        print(pat_id)
        pat_data_path = study_info.eeg_data_path / pat_id
        spike_cumulator_fn = spike_cumulators_path / f"{pat_id}_SpikeCumulator.pickle"
        spike_amplitude_analyzer = SpikeAmplitudeAnalyzer(pat_id=pat_id, ieeg_data_path=pat_data_path)
        spike_amplitude_analyzer.get_files_in_folder(file_extension='.lay')

        # Load saved spike data
        spike_cumulator = SpikeAmplitudeAnalyzer().load_spike_cumulator(filepath=spike_cumulator_fn)
        sleep_stages = spike_cumulator.sleep_stage_ls
        channel_names = spike_cumulator.eeg_channels_ls

        ch_coords_dict = get_channel_coordinates(pat_id, study_info, channel_names)
        get_ch_avg_spike_features(pat_id, to_plot_stage_names, spike_cumulator, ch_coords_dict)

        # get amplitude of ch_avg_spike for each channel and each sleep stage
        stage_specific_ch_avg_spike_data = []
        for stidx, stage_name in enumerate(to_plot_stage_names):
            nr_tot_spikes_in_stage = np.sum(spike_cumulator.spike_counter[stage_name])
            for ch_name in ch_coords_dict.keys():
                nr_spikes_in_channel = spike_cumulator.get_nr_cumulated_spikes(stage_name, ch_name)
                ch_spike_probability = (nr_spikes_in_channel/nr_tot_spikes_in_stage)
                ch_avg_spike = spike_cumulator.get_average_spike(stage_name, ch_name)
                ch_weighted_avg_spike_amplitude = (np.max(ch_avg_spike)-np.min(ch_avg_spike))* ch_spike_probability
                entree_data = (pat_id, stage_name, ch_name, ch_spike_probability, ch_weighted_avg_spike_amplitude, ch_coords_dict[ch_name][0], ch_coords_dict[ch_name][1], ch_coords_dict[ch_name][2])
                stage_specific_ch_avg_spike_data.append(entree_data)
                pass
        ch_avg_spike_feats_df = pd.DataFrame(data=stage_specific_ch_avg_spike_data, columns = ['PatID', 'SleepStage', 'Channel', 'ChSpikeProbability', 'ChAvgSpike_Amplitude', 'X', 'Y', 'Z'])
        
        fig = generate_interactive_spike_features_plot(ch_avg_spike_feats_df, to_plot_stage_names)
        fig.show()

        fig_fpath = spike_cumulators_path /"Images" / f"Spike_Prob_Ampl_{pat_id}.html"
        #fig_fpath = spike_cumulators_path / f"Spike_Prob_Ampl_{pat_id}.html"
        fig.write_html(fig_fpath)
        pass
                
def generate_interactive_spike_features_plot(ch_spike_feats_df, stages_names_ls):
    pat_id = ch_spike_feats_df.PatID[0]
    # Create a 3D scatter plot for each Sleep Stage    
    fig = make_subplots(
                rows=2, cols=6,
                horizontal_spacing = 0.01,  vertical_spacing  = 0.1,
                subplot_titles=(stages_names_ls),
                start_cell="top-left",
                specs=[
                    [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                    [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                    ]
                )
    pass
    for ss_idx, stage_name in enumerate(stages_names_ls):
        stage_df = ch_spike_feats_df[ch_spike_feats_df.SleepStage.str.fullmatch(stage_name, case=False)]

        chavg_spike_chname = stage_df.Channel.to_numpy()
        chavg_spike_ampl = MinMaxScaler().fit_transform(stage_df.ChAvgSpike_Amplitude.to_numpy().reshape(-1,1)).flatten()
        ch_spike_prob = MinMaxScaler().fit_transform(stage_df.ChSpikeProbability.to_numpy().reshape(-1,1)).flatten()
        #chavg_spike_ampl = stage_df.ChAvgSpike_Amplitude.to_numpy()
        #ch_spike_prob = stage_df.ChSpikeProbability.to_numpy()

        #Count
        fig.add_trace(
            go.Scatter3d(
                x = stage_df.X.to_numpy(),
                y = stage_df.Y.to_numpy(),
                z = stage_df.Z.to_numpy(),
                mode='markers',  # Show markers and labels
                marker=dict(
                    size=10,
                    color=ch_spike_prob,
                    colorscale='viridis',
                    opacity=0.8,
                    showscale=True,
                    cmin=np.min(ch_spike_prob),
                    cmax=np.max(ch_spike_prob),
                    colorbar=dict(title="Scaled<br>Spike P", len=0.50, y=0.8),
                ),
                #text=stage_df['Channel'],
                text=[f"{chname}: {prob:.2f}" for chname, amp, prob in zip(chavg_spike_chname, chavg_spike_ampl, ch_spike_prob)],  # Custom text for the fourth dimension
                hoverinfo='text',
                # hovertemplate="X: %{x}<br>Y: %{y}<br>%{text}<extra></extra>"  # Display fourth dimension on hover
            ),
            row=1, col=ss_idx+1
        )

        #camera = dict(eye=dict(x=0.1, y=0.1, z=1.5))
        #fig.update_layout(scene_camera=camera, title=stage_name)

        # Amplitude        
        fig.add_trace(
            go.Scatter3d(
                x = stage_df.X.to_numpy(),
                y = stage_df.Y.to_numpy(),
                z = stage_df.Z.to_numpy(),
                mode='markers',  # Show markers and labels
                marker=dict(
                    size=10,
                    color=chavg_spike_ampl,
                    colorscale='viridis',
                    opacity=0.8,
                    showscale=True,
                    cmin=np.min(chavg_spike_ampl),
                    cmax=np.max(chavg_spike_ampl),
                    colorbar=dict(title="Scaled<br>A*P", len=0.50, y=0.25),
                ),
                text=stage_df['Channel'],
                #textposition='top center',  # Position of the text labels
                #textfont=dict(size=5),
                hoverinfo='text',
            ),
            row=2, col=ss_idx+1
        )
        pass


    fig.update_layout(title_text=f"{pat_id}<br>(Avg.Spike Amplitude) * P(Spike<sub>ch</sub>)", showlegend=False)

    # # Update layout to maximize each subplot
    # fig.update_layout(
    # title_text=f"{pat_id}\nP-P Amplitude from Ch.Avg. Spike, Normalized for each Sleep Stage", 
    # showlegend=False,
    # margin=dict(l=0, r=0, t=0, b=0),        # Remove outer margins
    # #height=500,                              # Set a fixed height
    # )

    
    fig.layout.scene1.camera.eye=dict(x=2.4, y=2.4, z=2.4)
    fig.layout.scene2.camera.eye=dict(x=2.4, y=2.4, z=2.4)
    fig.layout.scene3.camera.eye=dict(x=2.4, y=2.4, z=2.4)
    fig.layout.scene4.camera.eye=dict(x=2.4, y=2.4, z=2.4)
    fig.layout.scene5.camera.eye=dict(x=2.4, y=2.4, z=2.4)

    return fig


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

