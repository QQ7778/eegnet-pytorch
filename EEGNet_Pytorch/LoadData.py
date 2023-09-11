import mne
import os
import glob
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, TensorDataset


class LoadData:
    def __init__(self,eeg_file_path: str):
        self.eeg_file_path = eeg_file_path

    def load_raw_data_gdf(self,file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        import scipy.io as sio
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self,file_path_extension: str =''):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)

class LoadBCIC(LoadData):
    '''Subclass of LoadData for loading BCI Competition IV Dataset 2a'''
    def __init__(self, file_to_load, *args):
        self.stimcodes=('769','770','771','772')
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC,self).__init__(*args)

    def get_epochs(self, tmin=-0., tmax=2, baseline=None, downsampled=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data[:,:,:-1],
                  'y_labels':self.y_labels,
                  'fs':self.fs}
        return eeg_data


def cross_validate_sequential_split(x_data, y_labels, kfold):
    train_indices = {}
    eval_indices = {}
    skf = StratifiedKFold(n_splits=kfold, shuffle=False)
    i = 0
    for train_idx, eval_idx in skf.split(x_data, y_labels):
        train_indices.update({i: train_idx})
        eval_indices.update({i: eval_idx})
        i += 1
    return train_indices, eval_indices

def split_xdata(eeg_data, train_idx, eval_idx):
    x_train=np.copy(eeg_data[train_idx,:,:])
    x_eval=np.copy(eeg_data[eval_idx,:,:])
    x_train = torch.from_numpy(x_train).to(torch.float32)
    x_eval = torch.from_numpy(x_eval).to(torch.float32)
    return x_train, x_eval

def split_ydata(y_true, train_idx, eval_idx):
    y_train = np.copy(y_true[train_idx])
    y_eval = np.copy(y_true[eval_idx])
    y_train = torch.from_numpy(y_train)
    y_eval = torch.from_numpy(y_eval)
    return y_train, y_eval


def BCICDataLoader(x_train, y_train, batch_size=64, num_workers=2, shuffle=True):
    
    data = TensorDataset(x_train, y_train)

    train_data = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_data