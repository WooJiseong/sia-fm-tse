import os
import random

import librosa
import numpy as np
import sofar
import torch
import torchaudio
from resemblyzer import VoiceEncoder, preprocess_wav, trim_long_silences
from torch.utils.data import Dataset


def _load_rir_sofar(hrtf: sofar.Sofa, m_idx: int, sample_rate: int) -> torch.Tensor:
    """
    Extract a single impulse response from a sofar Sofa object and resample it.

    Args:
        hrtf:        sofar.Sofa object loaded via sofar.read_sofa().
        m_idx:       Measurement index M to extract.
        sample_rate: Target sample rate.

    Returns:
        rir: [R, sample_rate] float32 tensor (R = number of receivers, typically 2 for binaural).
    """
    _sr = int(getattr(hrtf, "Data_SamplingRate"))  # scalar after sofar loading
    data_ir: np.ndarray = getattr(hrtf, "Data_IR")
    rir = data_irR[m_idx].astype(np.float32)  # [R, N]
    rir = torchaudio.functional.resample(torch.from_numpy(rir), _sr, sample_rate)[
        ..., :sample_rate
    ]  # [R, sample_rate]
    return rir


class CIPIC_simulator:
    def __init__(self, brir_dir, sample_rate):
        self.sample_rate = sample_rate

        CIPIC_rir_scenes = [
            f"{brir_dir}/{f}" for f in os.listdir(brir_dir) if f.endswith(".sofa")
        ]
        self.CIPIC_rir_scenes = [sofar.read_sofa(sid) for sid in CIPIC_rir_scenes]
        self.face_to_face_idx = 608

    def sample_one_scene(self) -> sofar.Sofa:
        return random.sample(self.CIPIC_rir_scenes, 1)[0]

    def sample_one_rir(self, hrtf: sofar.Sofa) -> torch.Tensor:
        m_idx = random.randrange(hrtf.Data_IR.shape[0])
        return _load_rir_sofar(hrtf, m_idx, self.sample_rate)

    def zero_angle_rir(self, hrtf: sofar.Sofa) -> torch.Tensor:
        return _load_rir_sofar(hrtf, self.face_to_face_idx, self.sample_rate)


class RRBRIR_simulator:
    def __init__(self, brir_dir, sample_rate):
        self.sample_rate = sample_rate

        RRBRIR_rir_scenes = [
            f"{brir_dir}/{f}" for f in os.listdir(brir_dir) if f.endswith(".sofa")
        ]
        self.RRBRIR_rir_scenes = [sofar.read_sofa(sid) for sid in RRBRIR_rir_scenes]
        self.face_to_face_idx = 18

    def sample_one_scene(self) -> sofar.Sofa:
        return random.sample(self.RRBRIR_rir_scenes, 1)[0]

    def sample_one_rir(self, hrtf: sofar.Sofa) -> torch.Tensor:
        m_idx = random.randrange(hrtf.Data_IR.shape[0])
        return _load_rir_sofar(hrtf, m_idx, self.sample_rate)

    def zero_angle_rir(self, hrtf: sofar.Sofa) -> torch.Tensor:
        return _load_rir_sofar(hrtf, self.face_to_face_idx, self.sample_rate)


class ASH_simulator:
    def __init__(self, brir_dir, data_split, sample_rate):
        self.sample_rate = sample_rate

        ASH_dir = brir_dir
        if data_split == "tr/":
            self.ASH_rir_scenes = [
                "R06",
                "R07",
                "R09",
                "R12",
                "R13",
                "R17",
                "R18",
                "R19",
                "R20",
                "R21",
                "R22",
                "R23",
                "R24",
                "R25",
                "R26",
                "R27",
                "R28",
                "R31",
                "R32",
                "R33",
                "R34",
            ]  # removed 05 folder
        elif data_split == "cv/":
            self.ASH_rir_scenes = ["R03", "R04", "R08", "R10", "R11", "R30"]
        elif data_split == "tt/":
            self.ASH_rir_scenes = ["R01", "R02", "R14", "R15", "R16", "R29"]
        else:
            raise NotImplementedError(
                f"noise_dir indicate not train/val/text dataset: {data_split}"
            )
        self.ASH_rir_scenes.sort()
        self.ASH_rir_scene_rir_map = {}
        for sid in self.ASH_rir_scenes:
            files = []
            for rir_name in os.listdir(ASH_dir + "/" + sid):
                files.extend([f"{ASH_dir}/{sid}/{rir_name}"])
            self.ASH_rir_scene_rir_map[sid] = files

    def sample_one_scene(self) -> str:
        return random.sample(self.ASH_rir_scenes, 1)[0]

    def sample_one_rir(self, sid: str) -> torch.Tensor:
        rir_id = random.sample(self.ASH_rir_scene_rir_map[sid], 1)[0]
        return self._get_one_rir(rir_id)

    def zero_angle_rir(self, sid: str) -> torch.Tensor:
        rirs = self.ASH_rir_scene_rir_map[sid]
        a0_file = [file for file in rirs if file.endswith("A0.wav")][0]
        return self._get_one_rir(a0_file)

    def _get_one_rir(self, name: str) -> torch.Tensor:
        brir, sr = torchaudio.load(name)
        brir = torchaudio.functional.resample(brir, sr, self.sample_rate)
        if brir.shape[-1] < self.sample_rate:
            brir = torch.nn.functional.pad(
                brir, (0, self.sample_rate - brir.shape[-1]), mode="constant", value=0
            )
        return brir[..., : self.sample_rate]


class CATT_simulator:
    def __init__(self, brir_dir, data_split, sample_rate):
        self.sample_rate = sample_rate

        CATTRIR_dir = brir_dir
        if data_split == "tr/":
            self.CATTRIR_rir_scenes = [
                "0_0s",
                "0_1s",
                "0_2s",
                "0_5s",
                "0_6s",
                "0_7s",
                "1_0s",
            ]
        elif data_split == "cv/":
            self.CATTRIR_rir_scenes = ["0_3s", "0_9s"]
        elif data_split == "tt/":
            self.CATTRIR_rir_scenes = ["0_4s", "0_8s"]
        else:
            raise NotImplementedError(
                f"noise_dir indicate not train/val/text dataset: {data_split}"
            )
        self.CATTRIR_rir_scenes.sort()
        self.CATTRIR_rir_scene_rir_map = {}
        for sid in self.CATTRIR_rir_scenes:
            files = []
            for rir_name in os.listdir(CATTRIR_dir + "/" + sid):
                files.extend([f"{CATTRIR_dir}/{sid}/{rir_name}"])
            self.CATTRIR_rir_scene_rir_map[sid] = files

    def sample_one_scene(self) -> str:
        return random.sample(self.CATTRIR_rir_scenes, 1)[0]

    def sample_one_rir(self, sid: str) -> torch.Tensor:
        rir_id = random.sample(self.CATTRIR_rir_scene_rir_map[sid], 1)[0]
        return self._get_one_rir(rir_id)

    def zero_angle_rir(self, sid: str) -> torch.Tensor:
        rirs = self.CATTRIR_rir_scene_rir_map[sid]
        a0_file = [file for file in rirs if file.endswith("_0.wav")][0]
        return self._get_one_rir(a0_file)

    def _get_one_rir(self, name: str) -> torch.Tensor:
        brir, sr = torchaudio.load(name)
        brir = torchaudio.functional.resample(brir, sr, self.sample_rate)
        if brir.shape[-1] < self.sample_rate:
            brir = torch.nn.functional.pad(
                brir, (0, self.sample_rate - brir.shape[-1]), mode="constant", value=0
            )
        return brir[..., : self.sample_rate]


class LibriDataset_single_emb(Dataset):
    def __init__(
        self,
        root_dir,
        sample_rate=16000,
        wave_length=16000,
        pos_example_length=16000,
        neg_example_length=16000,
        source_num=2,
        min_source_num=2,
        enroll_num=-1,
        min_enroll_num=-1,
        active_num=[-1, 2],
        same_disturb=False,
        verbose=False,
        normalize=True,
        return_dvec=False,
        reproducable=True,
        perturb_speeds=None,
        filling_pattern="repeat",
        zero_in_tgt=False,
        dvec_rate=50,
        binaural=False,
        reverb="none",
        brir_dir=[],
        reverb_cond=True,
        zero_degree_pos=True,
        snr_db_range=[],
        noise_dir="",
        include_silent=False,
        special_spk=[],
        partial_range=[0.5, 0.9],
        neg_partial_range=[0.5, 0.9],
        tgt_pos_l_dec=0,
        tgt_neg_l=0,
        tgt_intensity=-10000000000,
        return_clean_dvec=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.wave_length = wave_length
        self.pos_example_length = pos_example_length
        self.neg_example_length = neg_example_length
        self.source_num = source_num
        self.min_source_num = min_source_num
        if enroll_num < 0:
            self.enroll_num = source_num
        else:
            self.enroll_num = enroll_num
        if min_enroll_num < 0:
            self.min_enroll_num = min_source_num
        else:
            self.min_enroll_num = min_enroll_num

        self.active_num = active_num
        self.verbose = verbose
        self.return_dvec = return_dvec
        self.reproducable = reproducable
        self.filling_pattern = filling_pattern
        self.zero_in_tgt = zero_in_tgt
        self.dvec_rate = dvec_rate
        self.binaural = binaural
        self.reverb_cond = reverb_cond
        self.snr_db_range = snr_db_range
        self.noise_dir = noise_dir
        self.special_spk = special_spk
        self.partial_range = partial_range
        self.neg_partial_range = neg_partial_range
        self.tgt_intensity = tgt_intensity
        self.same_disturb = same_disturb
        self.tgt_pos_l_dec = tgt_pos_l_dec
        self.tgt_neg_l = tgt_neg_l
        self.return_clean_dvec = return_clean_dvec

        if include_silent:
            raise NotImplementedError("include_silent")

        if normalize:
            raise NotImplementedError("normalize")

        if perturb_speeds is not None:
            self.augmentation = torchaudio.transforms.SpeedPerturbation(
                sample_rate, perturb_speeds
            )
        else:
            self.augmentation = None

        if return_dvec or return_clean_dvec:
            self.encoder = VoiceEncoder(device="cpu", verbose=False)

        self.person_ids = [f for f in os.listdir(root_dir)]
        self.person_ids.sort()
        self.person_sound_map = {}
        for pid in self.person_ids:
            files = []
            for chapter_id in os.listdir(root_dir + "/" + pid):
                summary = open(
                    f"{root_dir}/{pid}/{chapter_id}/{pid}-{chapter_id}.trans.txt"
                )
                files.extend(
                    [
                        (
                            f"{chapter_id}/{line.split(' ')[0]}.flac",
                            " ".join(line.split(" ")[1:]),
                        )
                        for line in summary
                    ]
                )
            files.sort()
            self.person_sound_map[pid] = files

        self.zero_degree_pos = zero_degree_pos
        if reverb == "none":
            self.reverb_sims = []
        elif reverb == "all":
            self.reverb_sims = [
                CIPIC_simulator(brir_dir[0], self.sample_rate),
                RRBRIR_simulator(brir_dir[1], self.sample_rate),
                ASH_simulator(brir_dir[2], noise_dir[-3:], self.sample_rate),
                CATT_simulator(brir_dir[3], noise_dir[-3:], self.sample_rate),
            ]
        else:
            raise NotImplementedError(reverb)

        self.noise_names = []
        if self.noise_dir != "":
            self.noise_names = os.listdir(self.noise_dir)

    def __len__(self):
        return len(self.person_ids)

    def rms(self, audio):
        return torch.sqrt(torch.mean(audio**2))

    def normalize_audio(self, target_audio, reference_rms):
        target_rms = self.rms(target_audio)
        normalization_factor = reference_rms / target_rms
        normalized_audio = target_audio * normalization_factor
        return normalized_audio

    def repeat_or_cut_wavefrom(self, wavefrom, desired_length):
        current_length = wavefrom.shape[-1]
        if current_length < desired_length:
            repeat_count = (desired_length + current_length - 1) // current_length
            wavefrom = wavefrom.repeat((1, repeat_count))
        cut_tensor = wavefrom[..., :desired_length]
        return cut_tensor

    def pad_or_cut_waveform(self, wavefrom, desired_length):
        current_length = wavefrom.shape[-1]
        if current_length < desired_length:
            wavefrom = torch.nn.functional.pad(
                wavefrom, (0, desired_length - current_length), "constant"
            )
        cut_tensor = wavefrom[..., :desired_length]
        return cut_tensor

    def load_and_repeat(
        self, sound_name, length, remove_zero=True, filling_pattern="repeat"
    ):
        audio, _ = librosa.load(sound_name, sr=self.sample_rate)
        if remove_zero:
            audio = trim_long_silences(audio)
        audio = torch.from_numpy(audio)
        if self.augmentation is not None:
            audio, _ = self.augmentation(audio)

        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        if filling_pattern == "repeat":
            audio = self.repeat_or_cut_wavefrom(audio, length)
            l = length
        elif filling_pattern == "pad":
            audio = self.pad_or_cut_waveform(audio, length)
            l = length
        elif filling_pattern == "new":
            l = audio.shape[-1]
        else:
            raise NotImplementedError(filling_pattern)
        return audio, l

    def get_dvec_emb(self, audio, emb_length):
        audio = preprocess_wav(audio.numpy())
        embed = self.encoder.embed_utterance(
            audio, return_partials=True, rate=self.dvec_rate
        )[1]
        embed = torch.from_numpy(embed)
        if embed.shape[0] < emb_length:
            embed = torch.nn.functional.pad(
                embed, (0, 0, 0, emb_length - embed.shape[0]), mode="constant", value=0
            )
        return embed

    def get_noise_ratio(self, wave, noise):
        snr_db = random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        snr = 10 ** (snr_db / 10)
        audio_power = torch.mean(wave**2)
        noise_power = torch.mean(noise**2)
        desired_noise_power = audio_power / snr
        noise_scaling_factor = torch.sqrt(desired_noise_power / noise_power)
        return noise_scaling_factor

    def get_noise_ratio_given_snr(self, wave, noise, snr_db):
        snr = 10 ** (snr_db / 10)
        audio_power = torch.mean(wave**2)
        noise_power = torch.mean(noise**2)
        desired_tgt_power = noise_power * snr
        tgt_scaling_factor = torch.sqrt(desired_tgt_power / audio_power)
        return tgt_scaling_factor

    def scale_intensity(self, sample, pos, neg):
        if self.tgt_intensity < -100:
            return sample, pos, neg
        pos_tgt_gain = self.get_noise_ratio_given_snr(
            pos[:1], pos[1:].sum(dim=0), self.tgt_intensity
        )
        pos[:1] = pos[:1] * pos_tgt_gain
        return sample, pos, neg

    def __getitem__(self, idx):
        if self.reproducable:
            random.seed(idx)
            tgt_pid = random.sample(self.person_ids, 1)[0]
        else:
            tgt_pid = self.person_ids[idx]

        tgt_sound_id, _ = random.sample(self.person_sound_map[tgt_pid], 1)[0]
        sound_name = f"{tgt_pid}/{tgt_sound_id}"
        if self.verbose:
            print("gt: ", sound_name)

        source_num = random.randint(self.min_source_num, self.source_num)
        enroll_num = random.randint(self.min_enroll_num, self.enroll_num)
        other_person_ids = list(self.person_sound_map.keys())
        other_person_ids.remove(tgt_pid)
        enroll_noise_pids = random.sample(other_person_ids, enroll_num - 1)
        enroll_noise_pids.sort()
        if self.same_disturb:
            sample_noise_pids = enroll_noise_pids[: source_num - 1]
        else:
            sample_noise_pids = random.sample(other_person_ids, source_num - 1)
        pos_noise_pids = (
            sample_noise_pids[: self.active_num[1] - 1]
            + enroll_noise_pids[self.active_num[1] - 1 :]
        )
        neg_pids = enroll_noise_pids[self.active_num[1] - 1 :]
        if self.verbose:
            print("mix: ", sample_noise_pids)
            print("pos: ", pos_noise_pids)
            print("neg: ", neg_pids)

        # create input sample
        acc_l = 0
        audios = []
        while acc_l < self.wave_length:
            sound_name_, _ = random.sample(self.person_sound_map[tgt_pid], 1)[0]
            sound, l = self.load_and_repeat(
                os.path.join(self.root_dir, tgt_pid + "/" + sound_name_),
                self.wave_length,
                filling_pattern=self.filling_pattern,
            )
            acc_l += l
            audios.append(sound)
        sound = torch.concat(audios, dim=-1)[..., : self.wave_length]
        sample = [sound]
        for i in sample_noise_pids:
            acc_l = 0
            audios = []
            while acc_l < self.wave_length:
                sound_name_, _ = random.sample(self.person_sound_map[i], 1)[0]
                sound, l = self.load_and_repeat(
                    os.path.join(self.root_dir, i + "/" + sound_name_),
                    self.wave_length,
                    remove_zero=(not self.zero_in_tgt),
                    filling_pattern=self.filling_pattern,
                )
                acc_l += l
                audios.append(sound)
            sound = torch.concat(audios, dim=-1)[..., : self.wave_length]
            sample.append(sound)
            if self.verbose:
                print("audio noise", sound_name_)
        sample = torch.stack(sample)

        # create positive enrollment
        acc_l = 0
        audios = []
        while acc_l < self.pos_example_length:
            sound_name_, _ = random.sample(self.person_sound_map[tgt_pid], 1)[0]
            sound, l = self.load_and_repeat(
                os.path.join(self.root_dir, tgt_pid + "/" + sound_name_),
                self.pos_example_length,
                filling_pattern=self.filling_pattern,
            )
            acc_l += l
            audios.append(sound)
        sound = torch.concat(audios, dim=-1)[..., : self.pos_example_length]
        pos_cond_separated = [sound]
        for i in pos_noise_pids:
            acc_l = 0
            audios = []
            while acc_l < self.pos_example_length:
                sound_name_, _ = random.sample(self.person_sound_map[i], 1)[0]
                sound, l = self.load_and_repeat(
                    os.path.join(self.root_dir, i + "/" + sound_name_),
                    self.pos_example_length,
                    filling_pattern=self.filling_pattern,
                )
                acc_l += l
                audios.append(sound)
            sound = torch.concat(audios, dim=-1)[..., : self.pos_example_length]
            pos_cond_separated.append(sound)
            if self.verbose:
                print("pos noise", sound_name_)
        pos_cond_separated = torch.stack(pos_cond_separated)

        # create negative enrollment
        neg_cond = []
        for i in neg_pids:
            acc_l = 0
            negs = []
            while acc_l < self.neg_example_length:
                sound_name_, _ = random.sample(self.person_sound_map[i], 1)[0]
                sound, l = self.load_and_repeat(
                    os.path.join(self.root_dir, i + "/" + sound_name_),
                    self.neg_example_length,
                    filling_pattern=self.filling_pattern,
                )
                acc_l += l
                negs.append(sound)
            sound = torch.concat(negs, dim=-1)[..., : self.neg_example_length]
            neg_cond.append(sound)
            if self.verbose:
                print("neg noise", sound_name_)
        neg_cond = torch.stack(neg_cond)

        neg_rirs = torch.zeros((1,))
        if len(self.reverb_sims) != 0:
            if len(self.reverb_sims) == 1:
                rir_simulator = self.reverb_sims[0]
            else:
                rng = random.Random()
                rir_simulator = rng.sample(self.reverb_sims, 1, counts=[35, 5, 45, 15])[
                    0
                ]
            if self.binaural:
                rir_scene = rir_simulator.sample_one_scene()

                sample = sample.repeat((1, 2, 1))
                pos_cond_separated = pos_cond_separated.repeat((1, 2, 1))
                neg_cond = neg_cond.repeat((1, 2, 1))

                mix_rirs = [
                    rir_simulator.sample_one_rir(rir_scene) for _ in range(source_num)
                ]
                if self.zero_degree_pos:
                    pos_rirs = [rir_simulator.zero_angle_rir(rir_scene)] + [
                        rir_simulator.sample_one_rir(rir_scene)
                        for _ in range(enroll_num - 1)
                    ]
                else:
                    pos_rirs = [
                        rir_simulator.sample_one_rir(rir_scene)
                        for _ in range(enroll_num)
                    ]
                neg_rirs = [
                    rir_simulator.sample_one_rir(rir_scene)
                    for _ in range(enroll_num - self.active_num[1])
                ]

                mix_rirs = torch.stack(mix_rirs)
                pos_rirs = torch.stack(pos_rirs)
                neg_rirs = torch.stack(neg_rirs)

                sample = torchaudio.functional.fftconvolve(sample, mix_rirs)[
                    ..., : self.wave_length
                ]
                if self.reverb_cond:
                    pos_cond_separated = torchaudio.functional.fftconvolve(
                        pos_cond_separated, pos_rirs
                    )[..., : self.pos_example_length]
                    neg_cond = torchaudio.functional.fftconvolve(neg_cond, neg_rirs)[
                        ..., : self.neg_example_length
                    ]
            else:
                rir_scene = rir_simulator.sample_one_scene()

                mix_rirs = [
                    rir_simulator.sample_one_rir(rir_scene) for _ in range(source_num)
                ]
                if self.zero_degree_pos:
                    pos_rirs = [rir_simulator.zero_angle_rir(rir_scene)] + [
                        rir_simulator.sample_one_rir(rir_scene)
                        for _ in range(enroll_num - 1)
                    ]
                else:
                    pos_rirs = [
                        rir_simulator.sample_one_rir(rir_scene)
                        for _ in range(enroll_num)
                    ]
                neg_rirs = [
                    rir_simulator.sample_one_rir(rir_scene)
                    for _ in range(enroll_num - self.active_num[1])
                ]

                mix_rirs = torch.stack(mix_rirs)
                pos_rirs = torch.stack(pos_rirs)
                neg_rirs = torch.stack(neg_rirs)

                sample = torchaudio.functional.fftconvolve(sample, mix_rirs)[
                    ..., : self.wave_length
                ]
                if self.reverb_cond:
                    pos_cond_separated = torchaudio.functional.fftconvolve(
                        pos_cond_separated, pos_rirs
                    )[..., : self.pos_example_length]
                    neg_cond = torchaudio.functional.fftconvolve(neg_cond, neg_rirs)[
                        ..., : self.neg_example_length
                    ]

        if len(self.special_spk) != 0:
            partial_pos_num = 0
            if "Partial_Pos" in self.special_spk:
                partial_pos_num = random.randint(0, enroll_num - self.active_num[1])
                for i in range(partial_pos_num):
                    active_len = int(
                        self.pos_example_length
                        * random.uniform(self.partial_range[0], self.partial_range[1])
                    )
                    active_len = int(
                        min(active_len, self.pos_example_length - self.sample_rate // 2)
                    )
                    start = random.randint(0, self.pos_example_length - active_len)
                    end = start + active_len
                    pos_cond_separated[self.active_num[1] + i, :, :start] = 0
                    pos_cond_separated[self.active_num[1] + i, :, end:] = 0
                    neg_cond[i] = 0

            if "Partial_Neg" in self.special_spk:
                partial_neg_num = random.randint(
                    0, enroll_num - self.active_num[1] - partial_pos_num
                )
                for i in range(partial_neg_num):
                    active_len = int(
                        self.neg_example_length
                        * random.uniform(
                            self.neg_partial_range[0], self.neg_partial_range[1]
                        )
                    )
                    active_len = int(max(active_len, self.sample_rate // 2))
                    start = random.randint(0, self.neg_example_length - active_len)
                    end = start + active_len
                    neg_cond[partial_pos_num + i, :, :start] = 0
                    neg_cond[partial_pos_num + i, :, end:] = 0

        if source_num < self.source_num:
            sample = torch.nn.functional.pad(
                sample,
                (0, 0, 0, 0, 0, self.source_num - source_num),
                mode="constant",
                value=0,
            )

        if enroll_num < self.enroll_num:
            pos_cond_separated = torch.nn.functional.pad(
                pos_cond_separated,
                (0, 0, 0, 0, 0, self.enroll_num - enroll_num),
                mode="constant",
                value=0,
            )
            neg_cond = torch.nn.functional.pad(
                neg_cond,
                (0, 0, 0, 0, 0, self.enroll_num - enroll_num),
                mode="constant",
                value=0,
            )

        if len(self.snr_db_range) > 0:
            sample = torch.nn.functional.pad(
                sample, (0, 0, 0, 0, 0, 1), mode="constant", value=0
            )
            pos_cond_separated = torch.nn.functional.pad(
                pos_cond_separated, (0, 0, 0, 0, 0, 1), mode="constant", value=0
            )
            neg_cond = torch.nn.functional.pad(
                neg_cond, (0, 0, 0, 0, 0, 1), mode="constant", value=0
            )

            noise_name = random.sample(self.noise_names, 1)[0]
            noise, _ = self.load_and_repeat(
                self.noise_dir + noise_name,
                self.wave_length + self.pos_example_length + self.neg_example_length,
                remove_zero=False,
                filling_pattern="repeat",
            )

            noise_scaling_factor = self.get_noise_ratio(sample[0], noise)
            noise = noise_scaling_factor * noise

            sample[-1] += noise[:, : sample.shape[-1]]
            pos_cond_separated[-1] += noise[
                :, sample.shape[-1] : (sample.shape[-1] + pos_cond_separated.shape[-1])
            ]
            neg_cond[-1] += noise[
                :, (sample.shape[-1] + pos_cond_separated.shape[-1]) :
            ]

        sample, pos_cond_separated, neg_cond = self.scale_intensity(
            sample, pos_cond_separated, neg_cond
        )

        return sample, pos_cond_separated, neg_cond
