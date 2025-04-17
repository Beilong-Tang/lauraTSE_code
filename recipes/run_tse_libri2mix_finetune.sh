#!/bin/bash

###########
# Setting #
###########

name="libri2mix_finetune"
config_path=exp/$name/config_log_mel_aux_5s_finetune_e_20.yaml
resume=""

fine_tune="/DKUdata/tangbl/laura_TSE/ckpt/laura_tse_librispeech_dm_e_100.pth"

. utils/parse_options.sh

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

###############
## Run  DDP  ##
###############
python -u src/train.py --config $config_path --log $log_path --ckpt_path $ckpt_path --resume $resume \
 --fine_tune $fine_tune