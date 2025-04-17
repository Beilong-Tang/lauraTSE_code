# Experiment Config

We have provided three configs:

- Libri2mix Clean training set: `exp/libri2mix/config_log_mel_aux_5s.yaml`.
- LibriSpeech Dynamic Mixing: `exp/librispeech/config_log_mel_aux_5s_e_100_patience.yaml`.
- Libri2mix Finetune: `exp/libri2mix_finetune/config_log_mel_aux_5s_finetune_e_20.yaml`

## Key configs

In each config, please change the `FunCodec` part and `Data` part:

- `codec_model_file`: path to the Funcodec model ckpt _(*/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth)_
- `codec_config_file`: path to the Funcodec model config _(*/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml)_
- `train_shape_file`: A list of one string, which is the path to the training shape scp file. E.x. `['/path/to/all_shape.scp']`
- `valid_shape_file`: A list of one string, which is the path to the valid shape scp file. E.x. `['/path/to/all_shape.scp']`. 
- `train_data_path_and_name_and_type`: If not using dynamic mixing, this field should be like 
```yaml
train_data_path_and_name_and_type: [
    [
        "<>", # training mix scp
        "text", # DONT CHANGE
        "mix_mel" # DONT CHANGE
    ],
    [
        "<>", # training aux_s1 scp
        "aux", # DONT CHANGE
        "ref_mel" # DONT CHANGE
    ],
    [
        "<>", # training target s1 codec scp
        "codec", # DONT CHANGE
        "npy" # DONT CHANGE
    ]
]
```

Otherwise, if using dynamic mixing:
```yaml
train_data_path_and_name_and_type: [
    [
        "<>", # Path to s1 target clean.scp
        "text", # DONT CHANGE
        "dm_mix" # DONT CHANGE
    ],
    [
        "<>", # Path to s1 target clean.scp. Note that this is the same as above.
        "aux", # DONT CHANGE
        "dm_ref" # DONT CHANGE
    ],
    [
        "<>", # training target s1 codec scp
        "codec", # DONT CHANGE
        "npy" # DONT CHANGE
    ]
]
```
- spk_dict: (Used in Dynamic Mixing) Path to a pickle file containing a dictionary where each key is a speaker UID and each value is a list of audio paths corresponding to that speaker, used for training.

## Configs about batch size


- `batch_bins`: Controls how many tokens are in one batch.
- `max_mix_ds`: Maximum mixture audio duration.
- `max_aux_ds`: Maximum reference audio duration.

These three controls the model bandwidth. If you encounter CUDA error, you can lower the `batch_bins` as well as restricting the `max_mix_ds` and `max_aux_ds`. 
