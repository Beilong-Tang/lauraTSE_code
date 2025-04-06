from torchaudio.io import CodecConfig, AudioEffector
import torch
import numpy as np 
def codec_compression(
    speech_sample,
    fs: int,
    format: str,
    encoder: str = None,
    qscale: int = None,
):
    assert format in ["mp3", "ogg"], format
    assert encoder in [None, "None", "vorbis", "opus"], encoder

    encoder = None if encoder == "None" else encoder
    if speech_sample.ndim == 2:
        speech_sample = speech_sample.T  # (channel, sample) -> (sample, channel)
    try:
        module = AudioEffector(
            format=format,
            encoder=encoder,
            codec_config=CodecConfig(qscale=qscale),
            pad_end=True,
        )
        output = module.apply(torch.from_numpy(speech_sample), fs).numpy()
    except Exception as e:
        print(format, encoder, qscale, flush=True)
        print(e, flush=True)

    if output.shape[0] < speech_sample.shape[0]:
        zeros = np.zeros((speech_sample.shape[0] - output.shape[0], output.shape[1]))
        output = np.concatenate((output, zeros), axis=0)
    elif output.shape[0] > speech_sample.shape[0]:
        output = output[: speech_sample.shape[0]]

    assert speech_sample.shape == output.shape, (speech_sample.shape, output.shape)
    return (
        output.T if output.ndim == 2 else output
    )  # (sample, channel) -> (channel, sample)

