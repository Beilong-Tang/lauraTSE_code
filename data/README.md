# Data

We provide our data processing for dynamic mixing for LibriSpeech and evaluation on Libri2Mix. 

For Libri2Mix training generation, I am still working on it. 

## Training Data

We use train-clean-100, and train-clean-360 of LibriSpeech for training. The data is available at https://www.openslr.org/12.

## Evaluation and Test data

For evaluation and testing, our reference speech is randomly selected. 
To enhance reproducibility and lower the complexity to generate the data, we directly upload
our data to huggingface:

`libri2mix_dev`: https://huggingface.co/datasets/Beilong/libri2mix_clean_target/resolve/main/libri2mix_dev.tar.gz?download=true


`libri2mix_test`: https://huggingface.co/datasets/Beilong/libri2mix_clean_target/resolve/main/libri2mix_test.tar.gz?download=true

After downloading and extracting, you get `libri2mix_dev` and `libri2mix_test` where each 
folder has:
- s1: the clean speech for target speaker
- aux_s1: reference speech for s1
- mix_clean: the mixture. 


## SCP generation

Make sure you have all the data following the [Data Preparation](#data-preparation) step. 

Create an output folder which does not have subfoler `list` in it.

Run
```
python generate_list.py --librispeech_train_100 <path_to_train-clean-100> \
 --librispeech_train_360 <path_to_train-clean-360> \
 --libri2mix_dev <path_to_libri2mix_dev> \
 --libri2mix_test <path_to_libri2mix_test> \
 --output dump/
```

Scp files will be generated under the `list` folder of your output path.

You list folder will look like:
```
.
├── libri2mix_dev
│   ├── aux_s1.scp 
│   ├── mix_clean.scp
│   └── s1.scp
├── libri2mix_test
│   ├── aux_s1.scp
│   ├── mix_clean.scp
│   └── s1.scp
└── train
    ├── train_100_360_clean.scp
    └── train_100_360_spk_dict.pkl # Dict[str, list[str]] mapping a speaker to all its utterances
```

`train_100_360_spk_dict.pkl` is nothing but a `Dict[str, list[str]]` which maps a 
speaker to all its utterances. 

## FunCodec Codec Generation

### Libri2Mix Dev and Test

```sh
# FunCodec ckpt and config
codec_model_file="<Path to Funcodec model ckpt>"
codec_config_file="<Path to Funcodec model yaml>"

bash export_libri2mix_funcodec.sh --codec_model_file $codec_model_file --codec_config_file $codec_config_file
```

The output codec scp will be `dump/funcodec/libri2mix/*/all.scp`, and the output shape file will be `dump/funcodec/libri2mix/*/all_shape.scp`

### LibriSpeech

```sh
# FunCodec ckpt and config
codec_model_file="<Path to Funcodec model ckpt>"
codec_config_file="<Path to Funcodec model yaml>"

bash export_librispeech_funcodec_normalize.sh --codec_model_file $codec_model_file --codec_config_file $codec_config_file
```
