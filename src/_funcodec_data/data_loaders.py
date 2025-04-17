# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import collections
import functools
import logging
import numbers
import re
from typing import Mapping
from typing import Union

import numpy as np

from funcodec.fileio.npy_scp import NpyScpReader
from funcodec.datasets.dataset import sound_loader
from funcodec.fileio.read_text import read_2column_text
import random
import librosa
import pickle
from pathlib import Path
from utils.mel_spectrogram import MelSpec

from utils.audio import read_audio
from utils.utils import AttrDict
from utils.hinter import hint_once


def normalize(audio):
    max_value = np.max(np.abs(audio))
    return audio * (1 / (max_value + 1e-8))

class DmMixMelReader:
    def __init__(self, clean_path, spk_dict_path:str, mel_config:dict, snr = 5):
        # snr in [0,5]
        self.clean_scp = read_2column_text(clean_path)
        with open(spk_dict_path, "rb") as f:
            self.spk_dict = pickle.load(f)
        self.snr = snr
        self.mel_proc = MelSpec(**mel_config)
        pass 

    def __len__(self):
        return len(self.clean_scp)
    
    def __iter__(self):
        return iter(self.clean_scp)
    
    def __getitem__(self, uid):
        # load the path
        clean_path = self.clean_scp[uid]
        clean_spk_id = uid.split("-")[0]

        intf_spk = random.choice(list(self.spk_dict.keys()))
        while intf_spk == clean_spk_id:
            intf_spk = random.choice(list(self.spk_dict.keys()))
        intf_path = random.choice(self.spk_dict[intf_spk])

        # load the audio
        clean_audio, _ = librosa.load(clean_path, sr=None)
        intf_audio, _ = librosa.load(intf_path, sr=None)
        ## pad the length 
        if clean_audio.shape[0] > intf_audio[0]:
            ## repeat intf_audio 
            new_intf_audio = np.tile(intf_audio, len(clean_audio) // len(intf_audio) + 1)
            intf_audio = new_intf_audio[:len(clean_audio)]
        elif clean_audio.shape[0] < intf_audio[0]:
            offset = random.randint(0, len(intf_audio) - len(clean_audio) - 1)
            intf_audio = intf_audio[offset: offset + len(clean_audio)]
        assert intf_audio.shape == clean_audio.shape

        ## normalize 
        clean_audio, intf_audio = normalize(clean_audio), normalize(intf_audio)
        ## snr

        _snr = random.random() * self.snr
        intf_audio = intf_audio * 10 ** (-_snr / 20)
        mix = clean_audio + intf_audio
        return self.mel_proc.mel_one_np(mix)
    

class DmRefMelReader:
    def __init__(self, clean_path, spk_dict_path:str, mel_config:dict, ref_ds = 5):
        with open(spk_dict_path, "rb") as f:
            self.spk_dict = pickle.load(f)
        self.clean_scp = read_2column_text(clean_path)
        self.ds = ref_ds
        self.mel_proc = MelSpec(**mel_config)
    
    def __len__(self):
        return len(self.clean_scp)

    def __iter__(self):
        return iter(self.clean_scp)

    def __getitem__(self, uid):
        spk = uid.split("-")[0]

        ref_path = random.choice(self.spk_dict[spk])
        while Path(ref_path).stem == uid:
            ref_path = random.choice(self.spk_dict[spk])
        
        ref_speech, sr = librosa.load(ref_path, sr = None)
        ref_speech = ref_speech[-int(self.ds * sr):]
        ref_speech = normalize(ref_speech)
        return self.mel_proc.mel_one_np(ref_speech)

class MelReader:
    def __init__(self, scp_path, mel_config:dict, ref_ds = None):
        """
        Convert the input audio to mel spectrogram,
        if ref_ds is not None, then clip the audio and only choose the last ds seconds
        """
        self.scp_dict = read_2column_text(scp_path)
        self.mel_proc = MelSpec(**mel_config)
        self.ds =  ref_ds
    
    def __len__(self):
        return len(self.scp_dict)

    def __iter__(self):
        return iter(self.scp_dict)

    def __getitem__(self, uid):
        audio_path = self.scp_dict[uid]
        audio, sr = librosa.load(audio_path, sr = None)
        if self.ds is not None:
            audio = audio[-int(sr * self.ds):]
        audio = normalize(audio)
        return self.mel_proc.mel_one_np(audio)
    


DATA_TYPES = {
    "dm_mix": dict(
        func=DmMixMelReader, 
        kwargs=['spk_dict_path', "mel_config"],
        help="Dynamic Mixing for mixture log-mel"
        ),
    "dm_ref": dict(
        func=DmRefMelReader, 
        kwargs=['spk_dict_path', "mel_config", "ref_ds"],
        help="Dynamic Mixing for reference log-mel for librispeech"
        ),
    "mix_mel": dict(
        func=MelReader, 
        kwargs=["mel_config"],
        help="audio to mel"
        ), ## Mel spectrogram for the audio
    "ref_mel": dict(
        func=MelReader, 
        kwargs=["mel_config", "ref_ds"],
        help="audio to mel"
        ), ## Mel spectrogram for the audio
    "npy": dict(
        func=NpyScpReader,
        kwargs=[],
        help="Npy file format."
        "\n\n"
        "   utterance_id_A /some/where/a.npy\n"
        "   utterance_id_B /some/where/b.npy\n"
        "   ..."
    ),
    "sound": dict(
        func=sound_loader,
        kwargs=["float_dtype"],
        help="Audio format types which supported by sndfile wav, flac, etc."
        "\n\n"
        "   utterance_id_a a.wav\n"
        "   utterance_id_b b.wav\n"
        "   ...",
    )
}

