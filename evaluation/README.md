# Evaluation

For evalution, we use open-source metrics:

- [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).
- [NISQA](https://github.com/gabrielmittag/NISQA)
- [SpeechBERT](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics)
- [Whisper](https://github.com/openai/whisper). We use the `base` model.
- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
- [jiwer](https://github.com/jitsi/jiwer) for calculating dWER.

You can refer to `src/eval` for evaluation source code. More details can be found in our paper. 

## Installation

`pip install requirements.txt`

## Run 

### Pre-requisite

The folder structure should be:

```
output/
└── wavs
    ├── output1.wav
    ├── output2.wav
    ├── output3.wav
    └── ......
```
The output folder only contains a `wavs` folder where the output wavs are put in. 

After the evaluation is done, the log file will be output to `output/` folder.

### Evaluation

```sh

mix_wav_scp="/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/test/mix.scp"
ref_wav_scp="/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/test/aux_s1.scp"

# LauraTSE config and ckpt
config_path="/DKUdata/tangbl/laura_TSE/ckpt/config_log_mel_aux_5s_finetune_librispeech_epoch_100.yaml"
model_ckpt="/DKUdata/tangbl/laura_TSE/ckpt/laura_tse_librispeech_dm_e_100.pth"

# FunCodec ckpt and config
codec_model_file="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth"
codec_config_file="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml"


output_dir="output/test"

# DDP #
num_proc=4
gpus="cuda:4 cuda:5 cuda:6 cuda:7"

# DNSMOS
dns_model_dir="/DKUdata/tangbl/data/DNS-Challenge/DNSMOS"

# NISQA
nisqa_dir="/DKUdata/tangbl/pkg/evaluation/NISQA"

# Libri2mix Clean Dir
libri2mix_clean_dir="/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/test/s1"

bash evaluation/evaluation.sh 
```