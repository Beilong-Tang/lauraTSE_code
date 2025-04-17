#!/bin/bash

set -e
set -u
set -o pipefail

# Libri2Mix Clean Data
###########
# out dir #
###########
out_dir=/DKUdata/tangbl/data/librispeech/funcodec/data

########
# Data #
########
scp_list=("/DKUdata/tangbl/data/librispeech/funcodec/scp/clean.scp")
type=("train")

#######
# DDP #
#######
num_proc=18
gpus="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

#########
# Model #
#########
model="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth"
config="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml"

. utils/parse_options.sh

# Iterate using indices
for ((i=0; i<${#scp_list[@]}; i++)); do
    type=${type[$i]}
    echo "Processing $type"
    scp_file=${scp_list[$i]}
    python utils/export_libri2mix_funcodec.py --scp_file $scp_file \
      --config $config --model $model --output $out_dir/$type \
      --num_proc $num_proc --gpus $gpus \
      --normalize ## Normalize input
done

echo "everything done"
