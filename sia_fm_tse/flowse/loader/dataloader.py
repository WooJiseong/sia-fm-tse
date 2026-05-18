import numpy as np
import numpy
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import random
import torch.nn.functional as F
import torch
import torch as th
import torchaudio
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset,SequentialSampler,Sampler
import multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import yaml
import os
from pypinyin import pinyin, lazy_pinyin,Style
from tqdm import tqdm
import json
import sys
sys.path.append("../")
from model.modules import MelSpec
EPS = np.finfo(float).eps


def is_clipped(audio, clipping_threshold=0.99):
    return torch.any(torch.abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level using PyTorch"""
    rms = torch.sqrt(torch.mean(audio**2))
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def get_firstchannel_read(path, fs=16000):
    wave_data, sr = torchaudio.load(path)
    if sr != fs:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=fs)
        wave_data = resampler(wave_data)
    wave_data = wave_data[0, :]
    return wave_data



def audioread(path, fs=16000):
    """
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L x C or L
    """
    wave_data, sr = torchaudio.load(path)
    
    if sr != fs:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=fs)
        wave_data = resampler(wave_data)
    
    return wave_data.transpose(0,1)




def add_reverb(cln_wav, rir_wav):
    # cln_wav: L
    # rir_wav: L
    wav_tgt = sps.oaconvolve(cln_wav.numpy(), rir_wav.numpy())
    wav_tgt = wav_tgt[: cln_wav.shape[0]]
    return torch.tensor(wav_tgt)


def db2num(y):
    return torch.pow(10.0, y / 20.0)


def parse_scp(scp, path_list,test=-1, split_token=" "):
    with open(scp) as fid:
        if not test== -1:
            total = 500
            count = 0
        for line in fid:
            if not test==-1:
                if count > total:
                    break
                count += 1
            tmp = line.strip().split(split_token)
            if len(tmp) > 1:
                path_list.append({"inputs": tmp[0], "duration": float(tmp[1])})
            else:
                path_list.append({"inputs": tmp[0]})


def pad(audio, chunk_length, randstate):
    audio_length = audio.shape[0]

    if chunk_length > audio_length:
        st = randstate.randint(chunk_length + 1 - audio_length)
        audio_t = torch.zeros(chunk_length, dtype=audio.dtype)
        audio_t[st : st + audio_length] = audio
        audio = audio_t
    elif chunk_length < audio_length:
        st = randstate.randint(audio_length + 1 - chunk_length)
        audio = audio[st : st + chunk_length]
    return audio


def generate_data_one_noise(
    clean, noise, snr, scale, target_level=-25, clipping_threshold=0.99
):

    # Normalizing to -25 dB FS
    clean = clean / (torch.max(torch.abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = torch.sqrt(torch.mean(clean**2))

    noise = noise / (torch.max(torch.abs(noise)) + EPS)

    noise = normalize(noise, target_level)
    rmsnoise = torch.sqrt(torch.mean(noise**2))

    # Set the infer level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = scale

    rmsnoisy = torch.sqrt(torch.mean(noisyspeech**2))
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel =torch.max(torch.abs(noisyspeech))  / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel

    return noisyspeech, clean


def generate_data_two_noise(
    clean,
    noise1,
    noise2,
    snr_noise1,
    snr_noise2,
    scale,
    target_level=-25,
    clipping_threshold=0.99,
):

    # Normalizing to -25 dB FS
    clean = clean / (torch.max(torch.abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = torch.sqrt(torch.mean(clean**2))

    noise1 = noise1 / (torch.max(torch.abs(noise1)) + EPS)

    noise1 = normalize(noise1, target_level)
    rmsnoise1 = torch.sqrt(torch.mean(noise1**2))

    noise2 = noise2 / (torch.max(torch.abs(noise2)) + EPS)

    noise2 = normalize(noise2, target_level)
    rmsnoise2 = torch.sqrt(torch.mean(noise2**2))

    # Set the infer level for a given SNR
    noise1scalar = rmsclean / (10 ** (snr_noise1 / 20)) / (rmsnoise1 + EPS)
    noise1newlevel = noise1 * noise1scalar
    noise2scalar = rmsclean / (10 ** (snr_noise2 / 20)) / (rmsnoise2 + EPS)
    noise2newlevel = noise2 * noise2scalar

    # Mix noise and clean speech
    noisyspeech = clean + noise1newlevel + noise2newlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = scale

    rmsnoisy = torch.sqrt(torch.mean(noisyspeech**2))
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = torch.max(torch.abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel

    return noisyspeech, clean


def generate_reverdata_one_reverb_noise(clean, noise1, rir, snr, scale):
    clean_rir = rir[:, 0]
    noise_rir = rir[:, 1]
    clean = add_reverb(clean, clean_rir)
    noise1 = add_reverb(noise1, noise_rir)
    noisyspeech, clean = generate_data_one_noise(clean, noise1, snr, scale)
    return noisyspeech, clean


def generate_reverdata_one_noise(clean, noise1, rir, snr, scale):
    clean_rir = rir[:, 0]
    clean = add_reverb(clean, clean_rir)
    noisyspeech, clean = generate_data_one_noise(clean, noise1, snr, scale)
    return noisyspeech, clean


def generate_reverdata_two_reverb_noise(clean, noise1, noise2, rir, snr1, snr2, scale):
    clean_rir = rir[:, 0]
    noise1_rir = rir[:, 1]
    noise2_rir = rir[:, 2]
    clean = add_reverb(clean, clean_rir)
    noise1 = add_reverb(noise1, noise1_rir)
    noise2 = add_reverb(noise2, noise2_rir)
    noisyspeech, clean = generate_data_two_noise(
        clean, noise1, noise2, snr1, snr2, scale
    )
    return noisyspeech, clean


def generate_reverdata_one_reverb_noise_one_noise(
    clean, noise1, noise2, rir, snr1, snr2, scale
):
    clean_rir = rir[:, 0]
    noise1_rir = rir[:, 1]
    clean = add_reverb(clean, clean_rir)
    noise1 = add_reverb(noise1, noise1_rir)
    noisyspeech, clean = generate_data_two_noise(
        clean, noise1, noise2, snr1, snr2, scale
    )
    return noisyspeech, clean



class AutoDataset(Dataset):

    def __init__(
        self,
        clean_scp,
        regular_noise_scp,
        rir_scp,
        text_scp,
        repeat=1,
        num_workers=40,
        sample_rate=16000,
        probability=None,
        snr_ranges=None,
        scale_ranges=None,
    ):
        super(AutoDataset, self).__init__()

        mgr = mp.Manager()
        self.clean_list = mgr.list()
        self.regular_noise_list = mgr.list()
        self.rir_list = mgr.list()
        
        self.probaction = list(probability.keys())
        self.probvalues = list(probability.values())
    
        self.snr_ranges = snr_ranges
        self.scale_ranges = scale_ranges
        self.index = mgr.list()
        pc_list = []
        p = mp.Process(target=parse_scp, args=(clean_scp, self.clean_list))
        p.start()
        pc_list.append(p)

        p = mp.Process(
            target=parse_scp, args=(regular_noise_scp, self.regular_noise_list)
        )
        p.start()
        pc_list.append(p)

        p = mp.Process(target=parse_scp, args=(rir_scp, self.rir_list))
        p.start()
        pc_list.append(p)

        for p in pc_list:
            p.join()
        # init


        self.len_clean = len(self.clean_list)
        self.len_regular_noise = len(self.regular_noise_list)
        self.len_rir = len(self.rir_list)

        self.index = self.clean_list
        self.index *= repeat
        
        with open(text_scp, "r") as file:
            self.text_list = json.load(file)  
    

        self.mel_spectrogram = MelSpec(
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024,
                    n_mel_channels=100,
                    target_sample_rate=24000,
                    mel_spec_type='vocos',
                )
        self.randstates = [np.random.RandomState(idx) for idx in range(3000)]
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)

    def __len__(self):
        return len(self.index)

    def name(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def __next_probaiblity__(self):
        action = random.choices(self.probaction, self.probvalues)[0]
        return action

    def __select_rand_number__(self, probabilities, randstate):
        ranges = list(probabilities.keys())
        probs = list(probabilities.values())

        selected_range = randstate.choice(ranges, p=probs)
        start, end = map(int, selected_range.split("_to_"))

        rand_number = randstate.uniform(start, end)

        return rand_number

    def get_frame_len(self, index):
        
        return self.index[index]["duration"] * 24000 / 256


    def __getitem__(self, index):
        data_info = self.index[index]
        clean_path = data_info["inputs"]
        # import pdb;pdb.set_trace()
        item_key = os.path.splitext(os.path.basename(clean_path))[0]
        if not item_key in self.text_list.keys():
            text = ''
        else:
            text = self.text_list[item_key]


        clean = get_firstchannel_read(clean_path)

        chunk_length = clean.shape[-1]
      
        randstate = self.randstates[(index + 11) % 3000]

        idx_noise1 = randstate.randint(0, self.len_regular_noise)
        idx_noise2 = randstate.randint(0, self.len_regular_noise)
        # Two different noise samples
        
        while idx_noise2 == idx_noise1:
            idx_noise2 = randstate.randint(0, self.len_regular_noise)
        idx_rir = randstate.randint(0, self.len_rir)

        noise1_path = self.regular_noise_list[idx_noise1]["inputs"]
        noise2_path = self.regular_noise_list[idx_noise2]["inputs"]
        rir_path = self.rir_list[idx_rir]["inputs"]

        choice = randstate.uniform(0, 100)

        snr1 = self.__select_rand_number__(self.snr_ranges, randstate)
        snr2 = self.__select_rand_number__(self.snr_ranges, randstate)
        scale = self.__select_rand_number__(self.scale_ranges, randstate)

        choice = self.__next_probaiblity__()
        
        if choice == 'p1':
            noise1 = get_firstchannel_read(noise1_path)
            noise1 = pad(noise1, chunk_length, randstate)
            inputs, labels = generate_data_one_noise(clean, noise1, snr1, scale)
            desc = f"p1|{self.name(clean_path)}|{self.name(noise1_path)}|{snr1}|{scale}"
        elif choice == 'p2':
            noise1 = get_firstchannel_read(noise1_path)
            noise2 = get_firstchannel_read(noise2_path)
            noise1 = pad(noise1, chunk_length, randstate)
            noise2 = pad(noise2, chunk_length, randstate)
            inputs, labels = generate_data_two_noise(
                clean, noise1, noise2, snr1, snr2, scale
            )
            desc = f"p2|{self.name(clean_path)}|{self.name(noise1_path)}|{self.name(noise2_path)}|{snr1}_{snr2}|{scale}"
        elif choice == 'p3':
            noise1 = get_firstchannel_read(noise1_path)
            noise1 = pad(noise1, chunk_length, randstate)
            rir = audioread(rir_path)
            inputs, labels = generate_reverdata_one_reverb_noise(
                clean, noise1, rir, snr1, scale
            )
            desc = f"p3|{self.name(clean_path)}|{self.name(noise1_path)}|{self.name(rir_path)}|{snr1}|{scale}"
        elif choice == 'p4':
            noise1 = get_firstchannel_read(noise1_path)
            noise2 = get_firstchannel_read(noise2_path)
            noise1 = pad(noise1, chunk_length, randstate)
            noise2 = pad(noise2, chunk_length, randstate)
            rir = audioread(rir_path)
            inputs, labels = generate_reverdata_two_reverb_noise(
                clean, noise1, noise2, rir, snr1, snr2, scale
            )
            desc = f"p4|{self.name(clean_path)}|{self.name(noise1_path)}|{self.name(noise2_path)}|{self.name(rir_path)}|{snr1}_{snr2}|{scale}"
        elif choice == 'p5':
            noise1 = get_firstchannel_read(noise1_path)
            noise1 = pad(noise1, chunk_length, randstate)
            rir = audioread(rir_path)
            inputs, labels = generate_reverdata_one_noise(
                clean, noise1, rir, snr1, scale
            )
            desc = f"p5|{self.name(clean_path)}|{self.name(noise1_path)}|{self.name(rir_path)}|{snr1}|{scale}"
        else:
            noise1 = get_firstchannel_read(noise1_path)
            noise2 = get_firstchannel_read(noise2_path)
            noise1 = pad(noise1, chunk_length, randstate)
            noise2 = pad(noise2, chunk_length, randstate)
            rir = audioread(rir_path)
            inputs, labels = generate_reverdata_one_reverb_noise_one_noise(
                clean, noise1, noise2, rir, snr1, snr2, scale
            )
            desc = f"p6|{self.name(clean_path)}|{self.name(noise1_path)}|{self.name(noise2_path)}|{self.name(rir_path)}|{snr1}_{snr2}|{scale}"


        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        
        inputs = self.resampler(inputs)
        labels = self.resampler(labels)

        noisy_mel_spec = self.mel_spectrogram(inputs)
        noisy_mel_spec = noisy_mel_spec.squeeze(0)  # '1 d t -> d t'
        
        label_mel_spec = self.mel_spectrogram(labels)
        label_mel_spec = label_mel_spec.squeeze(0)  # '1 d t -> d t'
        raw_text = text
        text_list = pinyin(text, style=Style.TONE3)
        text = [''.join(item) for item in text_list]

        egs = {"noisy":inputs,"clean":labels,"label_mel_spec": label_mel_spec, "noisy_mel_spec": noisy_mel_spec, "label_path":clean_path,"text": text ,"raw_text":raw_text}
        return egs


def worker(target_list, result_list, start, end, chunk_length, sample_rate):
    for item in target_list[start:end]:
        duration = item["duration"]
        length = duration * sample_rate
        if length < chunk_length:
            sample_index = -1
            if length * 2 < chunk_length and length * 4 > chunk_length:
                sample_index = -2
            elif length * 2 > chunk_length:
                sample_index = -1
            else:
                continue
            result_list.append([item, sample_index])
        else:
            sample_index = 0
            while sample_index + chunk_length <= length:
                result_list.append([item, sample_index])
                sample_index += chunk_length
            if sample_index < length:
                result_list.append([item, int(length - chunk_length)])


def collate_fn(batch):
    label_mel_specs = [item["label_mel_spec"].squeeze(0) for item in batch]
    label_mel_lengths = torch.LongTensor([spec.shape[-1] for spec in label_mel_specs])
    max_mel_length = label_mel_lengths.amax()

    padded_label_mel_specs = []
    for spec in label_mel_specs:  
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_label_mel_specs.append(padded_spec)

    label_mel_specs = torch.stack(padded_label_mel_specs)


    noisy_mel_specs = [item["noisy_mel_spec"].squeeze(0) for item in batch]
    noisy_mel_lengths = torch.LongTensor([spec.shape[-1] for spec in noisy_mel_specs])
    max_mel_length = noisy_mel_lengths.amax()

    padded_noisy_mel_specs = []
    for spec in noisy_mel_specs:  
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_noisy_mel_specs.append(padded_spec)

    noisy_mel_specs = torch.stack(padded_noisy_mel_specs)
    label_paths = []
    for item in batch:
        label_paths.append(item['label_path'])

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        label_mel=label_mel_specs,
        noisy_mel=noisy_mel_specs,
        label_mel_lengths=label_mel_lengths,
        noisy_mel_lengths=noisy_mel_lengths,
        label_paths=label_paths,
        text=text,
        text_lengths=text_lengths,
    )

        
            
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        
        # The difference between adjacent frames within a batch. 
        # If the difference is too large, the smaller one will be padded with many frames in the collate_fn, which may cause out-of-memory issues.
        self.max_diff = 1.5
        
        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1],reverse=True)

        batch = []
        batch_frames = 0
        
        # import pdb;pdb.set_trace()
        
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if len(batch) > 0:
                prev_frame_len = data_source.get_frame_len(batch[-1])  
                frame_diff = max(prev_frame_len/frame_len,frame_len/prev_frame_len) 
            else:
                frame_diff = float('inf')  # If the current batch is empty, set a large difference to ensure the first frame is added
            
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples) and frame_diff <= self.max_diff:
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

       
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def make_auto_loader(
    clean_scp,
    regular_noise_scp,
    rir_scp,
    text_scp,
    repeat=1,
    batch_size=8,
    max_samples=64,
    num_workers=16,
    sample_rate=16000,
    probability=None,
    snr_ranges=None,
    scale_ranges=None,
):
    dataset = AutoDataset(
        clean_scp=clean_scp,
        text_scp=text_scp,
        regular_noise_scp=regular_noise_scp,
        rir_scp=rir_scp,
        repeat=repeat,
        num_workers=num_workers,
        sample_rate=sample_rate,
        probability=probability,
        snr_ranges=snr_ranges,
        scale_ranges=scale_ranges,
    )
    
    
    sampler = SequentialSampler(dataset)
    batch_sampler = DynamicBatchSampler(
        sampler, batch_size, max_samples=max_samples, random_seed=2102, drop_last=True
    )
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=batch_sampler,
    )
    return sampler,loader

