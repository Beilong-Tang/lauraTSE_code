#!/bin/bash
#SBATCH -J inf_tse_libri2mix_log_mel_config_aux_5s_only_clean_finetune_librispeech_epoch_100
#SBATCH -N 4
#SBATCH -o log/inf_tse_libri2mix_log_mel_config_aux_5s_only_clean_finetune_librispeech_epoch_100.out
#SBATCH -e log/inf_tse_libri2mix_log_mel_config_aux_5s_only_clean_finetune_librispeech_epoch_100.err
#SBATCH -p kshdnormal
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4

#### Special Note ####

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.10

###########
# Setting #
###########
stage=0
stop_stage=5 # 0: infer, 1: WER + SPKSIM 2. DNSMOS

# 0. libri2mix mixture without norm
name="libri2mix"
dataset="libri2mix" # Specify which dataset to infer
config_name="config_log_mel_aux_5s_finetune_librispeech_epoch_100.yaml"


mix_wav_scp="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/mix.scp"
ref_wav_scp="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/aux_s1.scp"



libri2mix_clean_dir="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/Libri2Mix/wav16k/min/test/s1" # for wespeaker only

# DDP #
num_proc=8
gpus="cuda:0 cuda:1 cuda:2 cuda:3"

########
# Eval #
########
dns_model_dir="/public/home/qinxy/bltang/ml_framework_slurm/recipes/DNSMOS"
wer_model="base"
wer_num_proc=8

# DONT CHANGE #
wer_reference="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/Libri2Mix/wav16k/min/whisper/whisper_$wer_model.txt"

###############
# DONT CHANGE #
###############
config_path=exp/$name/$config_name
output_dir="output/$name/$dataset/$(basename $config_name .yaml)"
output_dir=$(realpath $output_dir) # Add Change it to absolute path, as NISQA changes the directory
model_ckpt=ckpt/$name/$(basename "$config_path" .yaml)/best.pth

#######
# Run #
#######

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
echo "[Inference]"
python src/infer.py --mix_wav_scp $mix_wav_scp --ref_wav_scp $ref_wav_scp \
 --config $config_path --model_ckpt $model_ckpt --output_dir "$output_dir/wavs" --num_proc $num_proc --gpus $gpus
fi


########
# Eval #
########


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ## WER 
  echo "[WER $wer_model]"
  ## TODO: Change WER TO Large After downloading
  python src/eval/wer.py -t "$output_dir"/wavs -r $wer_reference \
    -o "$output_dir"/transcript_"$wer_model".txt -m $wer_model --num_proc $wer_num_proc
fi 


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  # SPK SIM
  echo "[SPKSIM WeSpeaker]"
  python src/eval/wespeaker_eval.py -t "$output_dir/wavs" \
    -r $libri2mix_clean_dir -o "$output_dir/wespeaker.csv" 

  echo "[Evaluation]"
  # DNSMOS
  echo "[DNSMOS 16k]"
  python src/eval/dnsmos.py --model_dir $dns_model_dir -t "$output_dir/wavs" -o "$output_dir/dnsmos.csv"
fi


##################################
# NISQA, SpeechBert, wavlm_base #
#####################################
nisqa_dir="/public/home/qinxy/bltang/pkg/NISQA" # The path contaitning the NISQA code and ckpts

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # NISQA
  echo "[NISQA]"
  cur_dir=$(pwd)
  cd $nisqa_dir
  python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar \
   --data_dir $output_dir/wavs --num_workers 0 --bs 10 \
   --output_dir $output_dir
  echo "NISQA inference finished"
  cd $cur_dir
  echo "NISQA Merging"
  python recipes_eval/nisqa_merge.py --output_dir $output_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # SpeechBert
  echo "[SpeechBert]"
  python src/eval/speech_bert.py --test_dir "$output_dir/wavs" \
  --ref_dir "$libri2mix_clean_dir" --out_dir "$output_dir"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # WavLM Base Plus SV SpkSim
  echo "[WavLM Base Plus SV SpkSim]"
  python src/eval/wavlm_base_plus_sv_spksim_eval.py --test_dir "$output_dir/wavs" \
  --ref_dir "$libri2mix_clean_dir" --out_dir "$output_dir"
fi
