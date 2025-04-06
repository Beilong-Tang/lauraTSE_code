import numpy as np 
import librosa

SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

RESAMPLE_METHODS = (
    "kaiser_best",
    "kaiser_fast",
    "scipy",
    "polyphase",
    #    "linear",
    #    "zero_order_hold",
    #    "sinc_best",
    #    "sinc_fastest",
    #    "sinc_medium",
)


def bandwidth_limitation_config(fs: int = 16000, res_type="random"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        fs (int): sampling rate in Hz
        res_type (str): resampling method

    Returns:
        res_type (str): adopted resampling method
        fs_new (int): effective sampling rate in Hz
    """
    # resample to a random sampling rate
    fs_opts = [fs_new for fs_new in SAMPLE_RATES if fs_new < fs]
    if fs_opts:
        if res_type == "random":
            res_type = np.random.choice(RESAMPLE_METHODS)
        fs_new = np.random.choice(fs_opts)
        opts = {"res_type": res_type}
    else:
        res_type = "none"
        fs_new = fs
    return res_type, fs_new

def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    assert fs > fs_new, (fs, fs_new)
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]
