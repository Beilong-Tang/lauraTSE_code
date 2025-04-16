# Evaluation

For evalution, we use open-source metrics:

- [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).
- [NISQA](https://github.com/gabrielmittag/NISQA)
- [SpeechBERT](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics)
- [Whisper](https://github.com/openai/whisper)
- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker). 

You can refer to `src/eval` for evaluation source code. More details can be found in our paper. 

## Installation

`pip install requirements.txt`

## Run 

### Pre-requisite

Note that to use our scripts, you need to format the output folder containing the wavs by creating another folder above it, for example:

```
output/
└── wavs
    ├── output1.wav
    ├── output2.wav
    ├── output3.wav
    └── ......
```
The output folder only contains a `wavs` folder where the output wavs are put in. 

After the evaluation is done, the log file will be output to `output/` folder.

### Evaluation

```sh
## Note that you need to be in the home directory to run the script
bash evaluation/evaluation.sh 
```