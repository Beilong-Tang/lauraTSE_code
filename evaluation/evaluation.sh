#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail


###########
# Setting #
###########

# output path (<output_dir>/wavs/*.wav) SEE README
output_dir=

# DDP #
num_proc=4
gpus="cuda:4 cuda:5 cuda:6 cuda:7"

# DNSMOS PATH (../DNS-Challenge/DNSMOS)
dns_model_dir=

# NISQA REPO PATH
nisqa_dir=

# test target wav dir
clean_dir=

stage=1
stop_stage=6

. utils/parse_options.sh

########
# Eval #
########

# WER
wer_model="base"
wer_reference="evaluation/libri2mix_whisper_${wer_model}.txt"
wer_num_proc=$num_proc

####

# WER
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ## WER 
  echo "[WER $wer_model]"
  echo "Preparing Models"
  python -c "import whisper; whisper.load_model('${wer_model}')"
  echo "Model preparation done"
  python src/eval/wer.py -t "$output_dir"/wavs -r $wer_reference \
    -o "$output_dir"/transcript_"$wer_model".txt -m $wer_model --num_proc $wer_num_proc
fi 

# Wespeaker Similarity
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # SPK SIM
  echo "[SPKSIM WeSpeaker]"
  python src/eval/wespeaker_eval.py -t "$output_dir/wavs" \
    -r $clean_dir -o "$output_dir/wespeaker.csv" 
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "[DNSMOS 16k]"
  python src/eval/dnsmos.py --model_dir $dns_model_dir -t "$output_dir/wavs" -o "$output_dir/dnsmos.csv"
fi 


#################################
# NISQA, SpeechBert, wavlm_base #
#################################

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # NISQA
  echo "[NISQA]"
  output_dir=$(realpath "$output_dir")
  cur_dir=$(pwd)
  cd $nisqa_dir
  python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar \
   --data_dir $output_dir/wavs --num_workers 0 --bs 10 \
   --output_dir $output_dir
  echo "NISQA inference finished"
  cd $cur_dir
  echo "NISQA Merging"
  python utils/nisqa_merge.py --output_dir $output_dir
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # SpeechBert
  echo "[SpeechBert]"
  python src/eval/speech_bert.py --test_dir "$output_dir/wavs" \
  --ref_dir "$clean_dir" --out_dir "$output_dir"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # WavLM Base Plus SV SpkSim
  echo "[WavLM Base Plus SV SpkSim]"
  python src/eval/wavlm_base_plus_sv_spksim_eval.py --test_dir "$output_dir/wavs" \
  --ref_dir "$clean_dir" --out_dir "$output_dir"
fi


