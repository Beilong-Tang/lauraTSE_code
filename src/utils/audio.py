import soundfile as sf 
import librosa
from pathlib import Path
#############################
# Audio utilities
#############################
def read_audio(filename, force_1ch=False, fs=None):
    """
    Return audio of shape [channel, frame]
    if `force_1ch=True`, it will be mono-channel regardless of the original audio
    """
    audio, fs_ = sf.read(filename, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
        return audio, fs
    return audio, fs_

