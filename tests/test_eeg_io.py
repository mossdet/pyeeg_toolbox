import numpy as np
from pyeeg_toolbox.eeg_io.eeg_io import EEG_IO


def test_eeg_montage_generation(self)->None:
        this_pat_eeg_file_path = ""
        assert len(self.pat_files_ls) > 0, f"No files found in folder {self.ieeg_data_path}"
        for record_idx in np.arange(start=0, stop=len(self.pat_files_ls)):
            this_pat_eeg_file_path = self.pat_files_ls[record_idx]

            # Read Referential Scalp EEG
            eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_file_path, mtg_t='sr')
            scalp_channels = eeg_reader.get_ch_names()
            scalp_eeg_data = eeg_reader.get_data(start=int(10*eeg_reader.fs), stop=int(20*eeg_reader.fs), plot_ok=True)

            # Read Bipolar Scalp EEG
            eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_file_path, mtg_t='sb')
            scalp_bip_channels = eeg_reader.get_ch_names()
            scalp_bip_eeg_data = eeg_reader.get_data(picks=0, start=int(10*eeg_reader.fs), stop=int(20*eeg_reader.fs), plot_ok=True)

            # Read Referential Intracranial EEG
            eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_file_path, mtg_t='ir')
            ieeg_channels = eeg_reader.get_ch_names()
            ieeg_data = eeg_reader.get_data(picks=0, start=int(10*eeg_reader.fs), stop=int(20*eeg_reader.fs), plot_ok=True)

            # Read Bipolar Intracranial EEG
            eeg_reader = EEG_IO(eeg_filepath=this_pat_eeg_file_path, mtg_t='ib')
            ieeg_bip_channels = eeg_reader.get_ch_names()
            ieeg_bip_data = eeg_reader.get_data(picks=0, start=int(10*eeg_reader.fs), stop=int(20*eeg_reader.fs), plot_ok=True)
            pass