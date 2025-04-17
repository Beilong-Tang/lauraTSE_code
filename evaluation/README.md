# Evaluation

For evalution, we use open-source metrics:

- [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).
- [NISQA](https://github.com/gabrielmittag/NISQA)
- [SpeechBERT](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics)
- [Whisper](https://github.com/openai/whisper). We use the `base` model.
- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
- [jiwer](https://github.com/jitsi/jiwer) for calculating dWER.

You can refer to `src/eval` for evaluation source code. More details can be found in our paper. 

## Installation

1. Clone [NISQA](https://github.com/gabrielmittag/NISQA)
2. Clone [DNSMOS](https://github.com/microsoft/DNS-Challenge)
3. `pip install requirements.txt`

## Run 

### Pre-requisite

The audio output folder structure should be:

```
output/
└── wavs
    ├── output1.wav
    ├── output2.wav
    ├── output3.wav
    └── ......
```
The output folder only contains a `wavs` folder where the audio wavs are. 

After the evaluation is done, the log file will be output to `output/` folder.

### Evaluation

1. Modify the setting part in `evaluation.sh`
2. Run
```sh
# output path (<output_dir>/wavs/*.wav) SEE README
output_dir="<Path to output>"

# DDP #
num_proc=4
gpus="cuda:0 cuda:1 cuda:2 cuda:3"

# DNSMOS PATH (../DNS-Challenge/DNSMOS)
dns_model_dir="<PATH to DNSMOS (../DNS-Challenge/DNSMOS)>"

# NISQA REPO PATH
nisqa_dir="<NISQA Repo Path>"

# test target wav dir (the audio wav names have to match the output wav)
clean_dir="<Path to the target clean wav dir>"

# in the root directory
bash evaluation/evaluation.sh --output_dir $output_dir \
  --num_proc $num_proc --gpus $gpus \
  --dns_model_dir $dns_model_dir --nisqa_dir $nisqa_dir \
  --clean_dir $clean_dir
```

