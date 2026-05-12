import librosa
import numpy as np
import soundfile as sf

import torch
import json
import os

class DataReader(object):
    def __init__(self,
                 mix_json,
                 mix_dir,
                 mix_fs=16000):

        with open(mix_json,'r') as f:
            self.mix_json =  json.load(f) 
        self.utt = list(self.mix_json.keys())
        
        self.mix_dir = mix_dir
        self.mix_fs = mix_fs

    def extract_feature(self, utt):
        mix_path = os.path.join(self.mix_dir,utt) + '.wav'
        text = self.mix_json[utt]
        mix_data = self.get_firstchannel_read(mix_path, self.mix_fs).astype(np.float32)

        mix_input = np.reshape(mix_data, [1, mix_data.shape[0]])

        mix_input = torch.from_numpy(mix_input)

        egs = {
            'utt_id': utt,
            'mix': mix_input,
            'text':text
        }

        return egs

    def __len__(self):
        return len(self.utt)

    def __getitem__(self, index):
        return self.extract_feature(self.utt[index])
    
        

    def get_firstchannel_read(self, path, fs, channel=0):
        wave_data, sr = sf.read(path)
        if sr != fs:
            if len(wave_data.shape) != 1:
                wave_data = wave_data.transpose((1, 0))
            wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
            if len(wave_data.shape) != 1:
                wave_data = wave_data.transpose((1, 0))
        if len(wave_data.shape) > 1:
            wave_data = wave_data[:, channel]
        return wave_data




