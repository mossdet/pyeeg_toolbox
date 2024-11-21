import pandas as pd


def correct_1096_chnames(chann_ls):
    ch_map_fpath = "F:/FREIBURG_Simultaneous_OneHrFiles/iEEG_Electrode_Coordinates/FR1096_Chann_Map.csv"
    ch_map_df = pd.read_csv(ch_map_fpath)
    new_chann_ls = []
    for ch in chann_ls:
        mapped_chname = ch_map_df.NewChanName[ch_map_df.OrigChanName.str.fullmatch(ch, case=False)].values
        assert len(mapped_chname) == 1, f"Channel {ch} with several mappings or not found in the mapping file"
        new_chann_ls.append(mapped_chname[0])
    return new_chann_ls