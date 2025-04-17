#!/bin/bash

set -e
set -u
set -o pipefail

# Libri2Mix Clean Data
###########
# out dir #
###########
out_dir=dump/funcodec

########
# Data #
########
scp_list=("dump/list/train/train_100_360_clean.scp")
type=("train")

#######
# DDP #
#######
num_proc=8
gpus="cuda:0 cuda:1 cuda:2 cuda:3"

#########
# Model #
#########
codec_model_file=
codec_config_file=

. utils/parse_options.sh

# Iterate using indices
for ((i=0; i<${#scp_list[@]}; i++)); do
    type=${type[$i]}
    echo "Processing $type"
    scp_file=${scp_list[$i]}
    python utils/export_libri2mix_funcodec.py --scp_file $scp_file \
      --config $codec_config_file --model $codec_model_file --output $out_dir/$type \
      --num_proc $num_proc --gpus $gpus \
      --normalize ## Normalize input
done

echo "everything done"
