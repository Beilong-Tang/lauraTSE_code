# Experiment Config

We have provided three configs:

- Libri2mix Clean training set: `exp/libri2mix/config_log_mel_aux_5s.yaml`.
- LibriSpeech Dynamic Mixing: `exp/librispeech/config_log_mel_aux_5s_e_100_patience.yaml`.
- Libri2mix Finetune: `exp/libri2mix_finetune/config_log_mel_aux_5s_finetune_e_20.yaml`

## Instruction

In each config, please change the `FunCodec` part and `Data` part:

```yaml
############
# FunCodec #
############

# For inference need
codec_model_file: <> # path to the Funcodec model ckpt (*/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth)
codec_config_file: <>  path to the Funcodec model file (*/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml)

init_param: ["<path to Funcodec model ckpt>:quantizer.rq.model:quantizer_codebook"]

train_shape_file: ["<training shape scp>"]
valid_shape_file: ["<validation shape scp>"]
train_data_path_and_name_and_type: [
    [
        "/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/train/all/mix.scp",
        "text",
        "mix_mel"
    ],
    [
        "/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/train/all/aux_s1.scp",
        "aux",
        "ref_mel"
    ],
    [
        "/DKUdata/tangbl/laura_gpt_se/libri2mix_tse_data_funcodec/s1/train/all.scp",
        "codec",
        "npy"
    ]
]

valid_data_path_and_name_and_type: [
     [
        "/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/dev/mix.scp",
        "text",
        "mix_mel"
    ],
    [
        "/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/dev/aux_s1.scp",
        "aux",
        "ref_mel"
    ],
    [
        "/DKUdata/tangbl/laura_gpt_se/libri2mix_tse_data_funcodec/s1/dev/all.scp",
        "codec",
        "npy"
    ]
]
```