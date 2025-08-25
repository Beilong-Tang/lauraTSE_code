#!/bin/bash


###########
# Setting #
###########

# Input wavs
mix_wav_scp=
ref_wav_scp=

# LauraTSE config and ckpt
config_path=
model_ckpt=

# FunCodec ckpt and config
codec_model_file=
codec_config_file=

output_dir=

# DDP #
num_proc=
gpus=

# Inference type #
infer="offline"

hop_ds=2

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
 --codec_model_file $codec_model_file --codec_config_file $codec_config_file \
 --infer "$infer" \
 --hop_ds "$hop_ds"

