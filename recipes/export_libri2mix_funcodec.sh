#!/bin/bash
#SBATCH -J extract_ilbri2mix_tse_funcodec
#SBATCH -N 1
#SBATCH -o log/extract_ilbri2mix_tse_funcodec.out
#SBATCH -e log/extract_ilbri2mix_tse_funcodec.err
#SBATCH -p kshdnormal
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4


export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2
source /public/software/apps/DeepLearning/whl/env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate urgent2025 # using urgent2025 environment

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/dtk/24.04

################################
#                              #
################################

## This script extracts the funcodec output of libri2mix target data set on ac
set -e
set -u
set -o pipefail


#######
# DDP #
#######
num_proc=16
gpus="cuda:0 cuda:1 cuda:2 cuda:3"

#########
# Model #
#########
model="/public/home/qinxy/bltang/LLM_TTS/egs/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth"
config="/public/home/qinxy/bltang/LLM_TTS/egs/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml"

# Libri2Mix Clean Data
echo "Libri2mix S1 Running..."
###########
# out dir #
###########

out_dir=/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/funcodec/s1

########
# Data #
########
scp_list=("/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/dev/s1.scp" \
          "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/s1.scp" \
          "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/train/all/s1.scp")
type=("dev" "test" "train")

# Iterate using indices
for ((i=0; i<${#scp_list[@]}; i++)); do
    type=${type[$i]}
    echo "Processing $type"
    scp_file=${scp_list[$i]}
    python export_libri2mix_funcodec.py --scp_file $scp_file \
      --config $config --model $model --output $out_dir/$type \
      --num_proc $num_proc --gpus $gpus
    
done

echo "Libri2mix S1 Done"

# Libri2Mix Aux S1 Data
echo "Libri2mix Aux S1 Running..."
###########
# out dir #
###########

out_dir=/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/funcodec/aux_s1

########
# Data #
########
scp_list=("/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/dev/aux_s1.scp" \
          "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/aux_s1.scp" \
          "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/train/all/aux_s1.scp")
type=("dev" "test" "train")

# Iterate using indices
for ((i=0; i<${#scp_list[@]}; i++)); do
    type=${type[$i]}
    echo "Processing $type"
    scp_file=${scp_list[$i]}
    python export_libri2mix_funcodec.py --scp_file $scp_file \
      --config $config --model $model --output $out_dir/$type \
      --num_proc $num_proc --gpus $gpus
    
done

echo "Libri2mix Aux S1 Done"

echo "Everything done..."


