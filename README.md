# LauraTSE

[![Paper](https://img.shields.io/badge/Paper-red?&logo=arxiv)](https://arxiv.org/abs/2504.07402) 
[![Demo](https://img.shields.io/badge/Demo-green?&logo=youtube)](https://beilong-tang.github.io/lauraTSE.demo//)

:warning: This repository is under construction, and it now contains my personal code for operating on my server. But it should contain the key codes for building lauraTSE Model now.


Official repository for [LauraTSE: Target Speaker Extraction using Auto-Regressive Decoder-Only Language Models](https://arxiv.org/abs/2504.07402).

To refer to our LauraTSE model, please see `src/model/laura_model_only_clean.py`. It is adapted from `LauraGPT` model from [FunCodec](https://github.com/modelscope/FunCodec). 

## Installation

Note that our experiments are run in `python3.10`.

1. Install [FunCodec](https://github.com/modelscope/FunCodec) package.
2. Install FunCodec [model](https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch): `audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch`.
3. Install the dependencies `pip install -r requirements.txt`.

## Data Preprocessing

Refer to `data/README.md`. 

## Checkpoints

Our checkpoint for LauraTSE can be found at [here](https://huggingface.co/Beilong/LauraTSE).

We have trained it on LibriSpeech using Dynamic Mixing with SNR 0-5 dB for 100 epochs and then finetune it on Libri2Mix for 20 epochs. 

Results on Libri2Mix clean testset:

|  Model   |  SIG  |  BAK  | OVRL  | NISQA | SpeechBERT | dWER  | WavLM Sim | Wespeaker Sim |
| :------: | :---: | :---: | :---: | :---: | :--------: | :---: | :-------: | :-----------: |
| LauraTSE | 3.609 | 4.084 | 3.336 | 4.333 |   0.908    | 0.159 |   0.974   |     0.876     |

## Inference

```sh
# Input wavs
mix_wav_scp="<Path to mix scp>"
ref_wav_scp="<Path to reference scp>"

# LauraTSE config and ckpt
config_path="<Path to model config>"
model_ckpt="<Path to model ckpt>"

# FunCodec ckpt and config
codec_model_file="<Path to Funcodec model ckpt>"
codec_config_file="<Path to Funcodec model yaml>"

# Output dir. Audio output will be in output/wavs.
output_dir="<Path to output>"

# DDP #
num_proc=4 # How many processes to run in parallel
gpus="cuda:0 cuda:1 cuda:2 cuda:3" # Available GPUs

bash recipes/inference.sh --mix_wav_scp $mix_wav_scp \
 --ref_wav_scp $ref_wav_scp \
 --config_path $config_path \
 --model_ckpt $model_ckpt \
 --codec_model_file $codec_model_file \
 --codec_config_file $codec_config_file \
 --output_dir $output_dir \
 --num_proc $num_proc \
 --gpus $gpus
```

Output audio will be output to `<output_dir>/wavs/*.wav`.


## Training

All the training configs are put in `exp/`. We have provided three training configs:

- Libri2mix Clean training set: `exp/libri2mix/config_log_mel_aux_5s.yaml`.
- LibriSpeech Dynamic Mixing: `exp/librispeech/config_log_mel_aux_5s_e_100_patience.yaml`.
- Libri2mix Finetune: `exp/libri2mix_finetune/config_log_mel_aux_5s_finetune_e_20.yaml`

To train the model:
1. Change the fields in the config.
2. Run one of the following scripts corresponding to the config:
```sh
# 1. Libri2mix Clean
export CUDA_VISIBLE_DEVICES="0,1,2,3"
bash recipes/run_tse_libri2mix.sh

# Or 2. Librispeech Dynamic Mxing
export CUDA_VISIBLE_DEVICES="0,1,2,3"
bash recipes/run_tse_librispeech_dm.sh

# Or 3. Libri2Mix Finetune
export CUDA_VISIBLE_DEVICES="0,1,2,3"
bash recipes/run_tse_libri2mix_finetune.sh --fine_tune <ckpt> 
```

For finetune, you can use the provided model [checkpoint](https://huggingface.co/Beilong/LauraTSE).. 

## Evaluation

Refer `evaluation/README.md`. 