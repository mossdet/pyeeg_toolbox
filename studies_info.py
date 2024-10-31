import socket
from pathlib import Path

class StudiesInfo:
    def __init__(self) -> None:
        self.eeg_data_path = None
        self.sleep_data_path = None
        self.ispikes_data_path = None
        self.study_patients = None
        self.dataset_name = None
    
    def fr_four_init(self):
        self.dataset_name = "Freiburg_Four"
        # Define directories containing the EEG data
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            self.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
            self.sleep_data_path = self.eeg_data_path
            self.ispikes_data_path = self.eeg_data_path
        elif socket.gethostname() == "DLP":
            self.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
            self.sleep_data_path = self.eeg_data_path
            self.ispikes_data_path = self.eeg_data_path
            

        # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
        self.study_patients = {
            'pat_FR_253':['HRC5', 'HP1', 'HP2', 'HP3'], 
            'pat_FR_970':['GC1'], 
            'pat_FR_1084':['M1', 'M2'], 
            'pat_FR_1096':['LDH1'],
            }
    