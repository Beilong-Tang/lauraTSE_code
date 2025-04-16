#!/bin/bash


###########
# Setting #
###########
mix_wav_scp="/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/test/mix.scp"
ref_wav_scp="/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/test/aux_s1.scp"

# LauraTSE config and ckpt
config_path="/DKUdata/tangbl/laura_TSE/ckpt/laura_tse_librispeech_dm_e_100_config.yaml"
model_ckpt="/DKUdata/tangbl/laura_TSE/ckpt/laura_tse_librispeech_dm_e_100.pth"

# FunCodec ckpt and config
codec_model_file="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth"
codec_config_file="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml"

output_dir="output/test"

# DDP #
num_proc=4
gpus="cuda:4 cuda:5 cuda:6 cuda:7"

. utils/parse_options.sh

###########
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

mkdir -p $output_dir


echo "[Inference]"
python src/infer.py --mix_wav_scp $mix_wav_scp --ref_wav_scp $ref_wav_scp \
 --config $config_path --model_ckpt $model_ckpt --output_dir "$output_dir/wavs"\
 --num_proc $num_proc --gpus $gpus \
 --codec_model_file $codec_model_file --codec_config_file $codec_config_file

