import numpy as np 
from utils.espnet2.train.preprocesser import detect_non_silence

def mix_noise(speech_sample, noise_sample, snr=5.0):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (Channel, Time)
        noise_sample (np.ndarray): a single noise sample (Channel, Time)
        snr (float): signal-to-nosie ratio (SNR) in dB
        rng (np.random.Generator): random number generator
    Returns:
        noisy_sample (np.ndarray): output noisy sample (Channel, Time)
        noise (np.ndarray): scaled noise sample (Channel, Time)
    """
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
    noisy_speech = speech_sample + noise
    return noisy_speech, noise

