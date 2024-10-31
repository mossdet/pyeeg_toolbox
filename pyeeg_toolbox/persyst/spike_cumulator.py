import numpy as np

class SpikeCumulator():
    """
    A class to accumulate the spike signals for each channel and having each spike centered on the max amplitude
    """

    def __init__(self, 
                 eeg_channels_ls:list=None,
                 sleep_stage_ls:list=None,
                 sig_wdw_dur_s:float=1.0,
                 sig_wdw_fs:float=64.0,
                 ):
        """
        Initializes the class.

        Args:
            _
            _
        """
        self.eeg_channels_ls = [chname.lower() for chname in eeg_channels_ls]
        self.sleep_stage_ls = sleep_stage_ls
        self.sig_wdw_dur_s = sig_wdw_dur_s
        self.sig_wdw_fs = sig_wdw_fs
        nr_chs = len(self.eeg_channels_ls)
        self.spike_counter = {stage:np.zeros(nr_chs) for stage in self.sleep_stage_ls}
        self.spike_cum_dict = {stage:np.zeros(shape=(nr_chs, int(sig_wdw_dur_s*sig_wdw_fs))) for stage in self.sleep_stage_ls}   
        self.spike_ampl = {stage:[[] for _ in range(nr_chs)] for stage in self.sleep_stage_ls}
        self.spike_freq = {stage:[[] for _ in range(nr_chs)] for stage in self.sleep_stage_ls}

        for k in self.spike_cum_dict.keys():
            assert self.spike_cum_dict[k].shape[0] == nr_chs, f"Spike cumulator has wrong nr. channels for stage {k}"
            assert self.spike_cum_dict[k].shape[1] == int(self.sig_wdw_fs*self.sig_wdw_dur_s), f"Spike cumulator has wrong nr. samples for stage {k}"

        pass

    def get_undersampling_frequency(self):
        return self.sig_wdw_fs

    def get_channels_ls(self):
        return self.eeg_channels_ls
    
    def get_ch_idx(self, ch_name):
        chidx = self.eeg_channels_ls.index(ch_name.lower())
        return chidx
    

    def add_spike(self, sleep_stage, ch_name, spike_signal, amplitude, frequency):
        """
        Adds a spike signal to the spike counter and cumulator.
        """
        try:
            ch_idx = self.get_ch_idx(ch_name)
            self.spike_counter[sleep_stage][ch_idx] += 1
            self.spike_cum_dict[sleep_stage][ch_idx] += spike_signal
            self.spike_ampl[sleep_stage][ch_idx].append(amplitude)
            self.spike_freq[sleep_stage][ch_idx].append(frequency)

        except Exception as e:
            print(e)
            print(f"Trying to add a spike to channel {ch_name}, whcih doesn't exist in the spike cumulator, check channel name being added or initialization of spike cumulator")
        
        pass

    def get_average_spike(self, sleep_stage, ch_name):
        ch_idx = self.get_ch_idx(ch_name)
        avg_ch_spike = self.spike_cum_dict[sleep_stage][ch_idx]
        if self.spike_counter[sleep_stage][ch_idx]>0:
            avg_ch_spike /= self.spike_counter[sleep_stage][ch_idx]

        return avg_ch_spike
    
    def get_nr_cumulated_spikes(self, sleep_stage, ch_name):
        ch_idx = self.get_ch_idx(ch_name)
        return self.spike_counter[sleep_stage][ch_idx]