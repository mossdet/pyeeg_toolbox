import socket
import platform
from pathlib import Path

class EEG_Study_Info:
    def __init__(self) -> None:
        self.eeg_data_path = None
        self.sleep_data_path = None
        self.ispikes_data_path = None
        self.patients = None
        self.dataset_name = None

def get_system_info():
    sys_info={}
    sys_info['hostname']=socket.gethostname()
    sys_info['machine']=platform.machine()
    sys_info['system']=platform.system()
    return sys_info
        
def fr_four_patients():
    study_info = EEG_Study_Info()
    study_info.dataset_name = "Freiburg_Four"
    sys_info = get_system_info()
    # Define directories containing the EEG data
    if sys_info['hostname']=="LAPTOP-TFQFNF6U" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
        study_info.sleep_data_path = study_info.eeg_data_path
        study_info.ispikes_data_path = study_info.eeg_data_path
    elif sys_info['hostname']=="DLP" and sys_info['machine']=="AMD64" and sys_info['system']=="Windows": 
        study_info.eeg_data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")
        study_info.sleep_data_path = study_info.eeg_data_path
        study_info.ispikes_data_path = study_info.eeg_data_path
    elif sys_info['hostname']=="dlp" and sys_info['machine']=="x86_64" and sys_info['system']=="Linux":
        study_info.eeg_data_path = Path("/media/dlp/Extreme Pro/FREIBURG_Simultaneous_OneHrFiles/")
        study_info.sleep_data_path = study_info.eeg_data_path
        study_info.ispikes_data_path = study_info.eeg_data_path
    
    # Define the names of the folders in the data_path directory that contain the files from each patient. Define also the list of bad channels  
    study_info.patients = {
        'pat_FR_253':['HRC5', 'HP1', 'HP2', 'HP3'], 
        'pat_FR_970':['GC1'], 
        'pat_FR_1084':['M1', 'M2'], 
        'pat_FR_1096':['LDH1'],
        }
    
    return study_info
    