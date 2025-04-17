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
The output folder only contains a `wavs` folder where the output wavs are put in. 

After the evaluation is done, the log file will be output to `output/` folder.

### Evaluation

1. Modify the setting part in `evaluation.sh`
2. Run
```sh
## In the root directory.
bash evaluation/evaluation.sh
```

