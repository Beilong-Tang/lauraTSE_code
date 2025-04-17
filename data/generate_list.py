import os.path as op
import os
import glob
import pickle
import tqdm
import argparse
from pathlib import Path

BASE_PATH = "."


def p(*args):
    return op.join(BASE_PATH, *args)


def generate_librispeech_training_spk_dict(train_100: str, train_360: str):
    print("generate training scp")
    spk_dict = {}
    train_audio = [train_100, train_360]
    for t in train_audio:
        audio_files = glob.glob(op.join(t, "*", "*", "*.flac"))
        for a in tqdm.tqdm(audio_files):
            spk = a.split("/")[-3]
            if spk in spk_dict:
                spk_dict[spk] = spk_dict[spk] + [a]
            else:
                spk_dict[spk] = [a]
    with open(p("list", "librispeech_train", "train_100_360_spk_dict.pkl"), "wb") as f:
        pickle.dump(spk_dict, f)
    
    ## Generate the speaker scp
    res = []
    for _, values in spk_dict.items():
        for path in values:
            name = Path(path).stem
            res.append(f"{name} {path}\n")
    with open(p("list", "librispeech_train", "train_100_360_clean.scp"), "w") as f:
        f.writelines(res)
    print("done!")


def generate_scp(dataset_name: str, type: str, name: str):
    print(f"generating scp for {name} of type {type}")
    files = sorted(glob.glob(p(dataset_name, type, "*.wav")))
    res = [f"{Path(i).stem} {i}\n" for i in files]
    with open(p("list", name, f"{type}.scp"), "w") as f:
        for r in res:
            f.write(r)
    print("done")

def generate_libri2mix_train(lm_100, lm_360, lm_aux, ls_spk_dict):
    
    mix_path = glob.glob(str(Path(lm_100)/'mix_clean'/'*.wav'))
    mix_path +=  glob.glob(str(Path(lm_360)/'mix_clean'/'*.wav'))

    res = [ f"{Path(p).stem} {p}\n" for p in mix_path]
    with open(p("list", "libri2mix_train", "mix_clean.scp"), "w") as file:
        file.writelines(res)

    s1_path = glob.glob(str(Path(lm_100)/'s1'/'*.wav'))
    s1_path += glob.glob(str(Path(lm_360)/'s1'/'*.wav'))

    res = [ f"{Path(p).stem} {p}\n" for p in mix_path]
    with open(p("list", "libri2mix_train", "s1.scp"), "w") as file:
        file.writelines(res)

    ## Generate lm_aux



    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ls100", "--librispeech_train_100", type=str, required=True)
    parser.add_argument("-ls360", "--librispeech_train_360", type=str, required=True)
    parser.add_argument("-lm_dev", "--libri2mix_dev", type=str, required=True)
    parser.add_argument("-lm_test", "--libri2mix_test", type=str, required=True)
    parser.add_argument('--libri2mix_train_clean_360', type=str, required=True)
    parser.add_argument('--libri2mix_train_clean_100', type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    BASE_PATH = args.output

    if op.exists(p("list")):
        raise FileExistsError(
            "Please choose another folder that does not have folder 'list' as output folder."
        )
    os.makedirs(p("list"), exist_ok=True)
    os.makedirs(p("list", "librispeech_train"), exist_ok=True)
    os.makedirs(p("list", "libri2mix_dev"), exist_ok=True)
    os.makedirs(p("list", "libri2mix_test"), exist_ok=True)
    os.makedirs(p("list", "libri2mix_train"), exist_ok=True)

    dev_audio = args.libri2mix_dev
    test_audio = args.libri2mix_test

    generate_librispeech_training_spk_dict(args.librispeech_train_100, args.librispeech_train_360)

    generate_libri2mix_train(args.libri2mix_train_clean_100, args.libri2mix_train_clean_360, 'libri2mix_100_360_aux_s1.txt' )

    for t in ["aux_s1", "mix_clean", "s1"]:
        for d, n in [(dev_audio, "libri2mix_dev"), (test_audio, "libri2mix_test")]:
            generate_scp(d, t, n)
    print("All scp files are generated successfully!")