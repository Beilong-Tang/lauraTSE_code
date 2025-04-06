# Training data loader for urgent-2025
# Start Time: Jan 8 2025
# Author: Beilong Tang

import librosa
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from dataset.augmentation import generate_from_config, generate_augmentations_config

from dataset.helper import log_dic
from dataset.helper import get_scp_dict
from dataset.random_sampler import RandomManager
import random
from utils.audio import read_audio
from utils.load_scp import get_source_list


def pad_audio(audio, win_size, hop_size):
    pad_length = (len(audio) - win_size) % hop_size
    if pad_length > 0:
        pad_length = hop_size - pad_length
    return np.pad(audio, (0, pad_length), mode="constant")


class TrainDMDataset(Dataset):
    def __init__(self, conf, epoch_num=100000, log=print):
        super().__init__()

        #####################
        # Load all the scps #
        #####################
        # speech scp
        speech_fs_scp = conf["speech"]["speech2fs2scp"]
        self.speech_dic = get_scp_dict(
            speech_fs_scp,
            conf["speech"].get("ignore_prefix"),
            base_path=conf["base_path"],
        )
        log_dic(self.speech_dic, log, "Speech")

        # noise scp
        noise_fs_scp = conf["noise"]["scp"]
        noise_ignore_prefix = conf["noise"].get("ignore_prefix")
        # Add wind prefix to noise_ignore_prefix to ignore wind_noise in the noise scp.
        noise_ignore_prefix = (
            ["wind_noise"]
            if noise_ignore_prefix is None
            else noise_ignore_prefix + ["wind_noise"]
        )  # Dont change this line
        self.noise_dic = get_scp_dict(
            noise_fs_scp, noise_ignore_prefix, base_path=conf["base_path"]
        )
        log_dic(self.noise_dic, log, "Noise")

        # wind noise
        wind_noise_scp = conf["wind_noise"]["scp"]
        wind_noise_ignore_prefix = conf["wind_noise"].get("ignore_prefix")
        self.wind_noise_dic = get_scp_dict(
            wind_noise_scp, wind_noise_ignore_prefix, base_path=conf["base_path"]
        )
        log_dic(self.wind_noise_dic, log, "Wind Noise")

        # rir noise
        rir_noise_scp = conf["rir"]["scp"]
        rir_noise_ignore_prefix = conf["rir"].get("ignore_prefix")
        self.rir_noise_dic = get_scp_dict(
            rir_noise_scp, rir_noise_ignore_prefix, base_path=conf["base_path"]
        )
        log_dic(self.rir_noise_dic, log, "rir")

        self.augmentations = list(conf["augmentations"].keys())
        weight_augmentations = [v["weight"] for v in conf["augmentations"].values()]
        conf.weight_augmentations = weight_augmentations / np.sum(weight_augmentations)

        self.sr = conf["sr"]
        self.repeat_per_utt = conf["repeat_per_utt"]
        self.conf = conf
        self.epoch_num = epoch_num
        self.speech_rm = None

    def __len__(self):
        return self.epoch_num

    def init_rm(self, resume_epoch=0):
        self.speech_rm = RandomManager(
            self.speech_dic, self.epoch_num, resume_epoch=resume_epoch
        )

    def set_epoch(self):
        self.speech_rm.set_epoch()

    def __getitem__(self, idx):
        """
        :return mix[T], clean[T], fs
        """
        if self.speech_rm == None:
            raise Exception(f"[TrainDMDataset] call init_rm before the epoch starts")
        fs, speech = self.speech_rm.get_uid_and_freq(idx)  # fs, uid
        speech_path = self.speech_dic[fs][speech]
        audio, fs_speech = read_audio(speech_path, force_1ch=True)  # [1,T]
        assert fs == fs_speech
        meta = generate_augmentations_config(
            self.conf,
            fs,
            audio,
            self.noise_dic,
            self.wind_noise_dic,
            self.rir_noise_dic,
        )
        clean, noisy = generate_from_config(
            meta, self.noise_dic, self.wind_noise_dic, self.rir_noise_dic
        )  # [1,T], [1,T]

        ############################
        # resample it to target sr #
        ############################
        target_sr = self.conf["target_sr"]
        if fs != target_sr:
            clean = librosa.resample(clean, orig_sr=fs, target_sr=target_sr)
            noisy = librosa.resample(noisy, orig_sr=fs, target_sr=target_sr)

        ## Pad audio ##
        clean = pad_audio(clean, self.conf["win_size"], self.conf["hop_size"])
        noisy = pad_audio(noisy, self.conf["win_size"], self.conf["hop_size"])

        return (
            torch.from_numpy(noisy).squeeze(),
            torch.from_numpy(clean).squeeze(),
            target_sr,
            speech,
            speech_path,
        )


def truc_wav_np(*audio: np.ndarray, length):
    """
    Given a list of audio with the same length as arguments, chunk the audio into a given length.
    Note that all the audios will be chunked using the same offset

    Args:
        audio: the list of audios to be chunked, should have the same length with shape [T] (1D)
        length: the length to be chunked into, if length is None, return the original audio
    Returns:
        A list of chuncked audios
    """
    audio_len = audio[0].shape[0]  # [T]
    res = []
    if length == None:
        for a in audio:
            res.append(a)
        return res[0] if len(res) == 1 else res
    if audio_len > length:
        offset = random.randint(0, audio_len - length - 1)
        for a in audio:
            res.append(a[offset : offset + length])
    else:
        for a in audio:
            res.append(np.pad(a, (0, length - a.shape[0]), constant_values=0))
    return res[0] if len(res) == 1 else res


class EvalDataset(Dataset):
    def __init__(
        self, noisy_scp, clean_scp, target_sr, win_size, hop_size, duration=4
    ) -> None:
        """
        Conduct evaluation on the target sample rate.
        """
        super().__init__()
        self.clean_scp = get_source_list(clean_scp)
        self.noisy_scp = get_source_list(noisy_scp)
        self.target_sr = target_sr
        self.audio_len = int(duration * target_sr)
        self.win_size = win_size
        self.hop_size = hop_size

    def __len__(self):
        return len(self.clean_scp)

    def __getitem__(self, idx):
        clean_path = self.clean_scp[idx]
        noisy_path = self.noisy_scp[idx]
        clean_audio = read_audio(clean_path, force_1ch=True, fs=self.target_sr)[0].squeeze(0)
        noisy_audio = read_audio(noisy_path, force_1ch=True, fs=self.target_sr)[0].squeeze(0)
        clean_audio, noisy_audio = truc_wav_np(clean_audio, noisy_audio, length = self.audio_len) # truc it to have the shape of [self.audio_len, ]

        ## Pad audio ##
        clean_audio = pad_audio(clean_audio, self.win_size, self.hop_size)
        noisy_audio = pad_audio(noisy_audio, self.win_size, self.hop_size)

        return torch.from_numpy(noisy_audio), torch.from_numpy(clean_audio)
