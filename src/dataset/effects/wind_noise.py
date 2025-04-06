import numpy as np
from utils.espnet2.train.preprocesser import detect_non_silence
import io
import soundfile as sf
import subprocess
import tempfile


def add_wind_noise(audio: np.ndarray, noise: np.ndarray, sr: int, params):
    """
    audio:np [1,T] or [T]
    noise: np [1,T] or [T] shape
    Return: noisy speech [1,T] or [T]
    use python subprocess to call ffmpeg to apply wind_noise
    """

    def buildFFmpegCommand(noise_path, output_path, params):
        filter_commands = ""
        filter_commands += "[1:a]asplit=2[sc][mix];"
        filter_commands += (
            "[0:a][sc]sidechaincompress="
            + f"threshold={params['threshold']}:"
            + f"ratio={params['ratio']}:"
            + f"level_sc={params['sc_gain']}"
            + f":release={params['release']}"
            + f":attack={params['attack']}"
            + "[compr];"
        )
        filter_commands += "[compr][mix]amix"

        commands_list = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-i",
            "pipe:0",
            "-i",
            noise_path,
            "-filter_complex",
            filter_commands,
            output_path,
        ]

        return commands_list

    ## Read Numpy audio to stdin
    byte_io = io.BytesIO()
    sf.write(byte_io, audio.squeeze(), sr, format="WAV")
    byte_io.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".wav") as noise_tmp_wav:
        with tempfile.NamedTemporaryFile(suffix=".wav") as output_tmp_wav:
            # noise_tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav")
            sf.write(noise_tmp_wav, noise.squeeze(), samplerate=sr)
            # output_tmp_wav = tempfile.NamedTemporaryFile(suffix= ".wav")
            commands = buildFFmpegCommand(
                noise_tmp_wav.name, output_tmp_wav.name, params
            )
            ffmpeg_process = subprocess.Popen(
                commands,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = ffmpeg_process.communicate(input=byte_io.read())
            res, sr = sf.read(output_tmp_wav.name)
    if len(audio.shape) == 1:
        return np.expand_dims(res, axis=0) # [1,T]
    else:
        return res


def wind_noise(
    speech_sample,
    noise_sample,
    fs,
    threshold,
    ratio,
    attack,
    release,
    sc_gain,
    clipping,
    clipping_threshold,
    snr,
):
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = np.random.randint(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = np.random.randint(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample

    scale = 0.9 / max(
        np.max(np.abs(speech_sample)),
        np.max(np.abs(noise)),
    )
    speech_sample *= scale
    noise *= scale

    params = {
        "threshold": threshold,
        "ratio": ratio,
        "attack": attack,
        "release": release,
        "sc_gain": sc_gain,
    }

    mix = add_wind_noise(speech_sample, noise, sr=fs, params=params)

    mix /= scale
    noise /= scale

    if clipping:
        mix = np.maximum(clipping_threshold * np.min(mix) * np.ones_like(mix), mix)
        mix = np.minimum(clipping_threshold * np.max(mix) * np.ones_like(mix), mix)

    return mix[None], noise[None]
