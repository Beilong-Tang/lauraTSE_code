#!/bin/bash

###########
# Setting #
###########

name="librispeech"
config_path=exp/$name/config_log_mel_aux_5s_e_100_patience.yaml
resume=""

. utils/parse_options.sh

echo "resume is $resume"

###############
# DONT CHANGE #
###############
save_dir=$(dirname "$config_path")/$(basename "$config_path" .yaml)
save_dir=${save_dir/#exp\//}
echo $save_dir
ckpt_path=ckpt/$save_dir
log_path=log/$save_dir
mkdir -p $ckpt_path
mkdir -p $log_path

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python -u src/train.py --config $config_path --log $log_path --ckpt_path $ckpt_path --resume $resume
