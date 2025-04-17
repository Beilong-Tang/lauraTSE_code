# Experiment Config

We have provided the librispeech dynamic mixing config:

- Libri2mix Clean training set: `exp/libri2mix/config_log_mel_aux_5s.yaml`.
- LibriSpeech Dynamic Mixing: `exp/librispeech/config_log_mel_aux_5s_e_100_patience.yaml`.
- Libri2mix Finetune: `exp/libri2mix_finetune/config_log_mel_aux_5s_finetune_e_20.yaml`


## Key configs

In each config, please change the `FunCodec` part and `Data` part:

- `codec_model_file`: path to the Funcodec model ckpt _(*/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth)_
- `codec_config_file`: path to the Funcodec model config _(*/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml)_

## Configs about batch size


- `batch_bins`: Controls how many tokens are in one batch.
- `max_mix_ds`: Maximum mixture audio duration.
- `max_aux_ds`: Maximum reference audio duration.

These three controls the model bandwidth. If you encounter CUDA error, you can lower the `batch_bins` as well as restricting the `max_mix_ds` and `max_aux_ds`. 
