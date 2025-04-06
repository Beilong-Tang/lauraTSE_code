#
# Evaluating Speaker Similarity using WavLM base plus https://huggingface.co/microsoft/wavlm-base-plus-sv
#

#################
# Global Config #
#################
WAVLM_BASE_PLUS_SV = "/public/home/qinxy/bltang/pkg/wavlm_base_plus_sv"
#===============#
import os 
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import os.path as op

import argparse
import tqdm
import glob
import librosa
import pandas as pd
from pathlib import Path

print("Importing")
import torch
import torch.multiprocessing as mp
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
print("Done")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir', type = str, required=True)
    p.add_argument('--ref_dir', type = str, required=True)
    p.add_argument('--ref_suffix', type = str, default = 'wav')

    # Output specific
    p.add_argument('--out_dir',type=str, required=True, help = "the directory to put the log results into")

    # ddp 
    p.add_argument('--gpus', nargs="+", default= ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    p.add_argument('--num_proc', type = int, default=8)
    return p.parse_args()

def main(args):
    os.makedirs(str(Path(args.out_dir) / '.temp'), exist_ok=True)
    mp.spawn(run_eval, args=(args,), nprocs=args.num_proc, join = True)
    ## Merge the results
    res = None
    for _r in range(args.num_proc):
        _csv_path = str(Path(args.out_dir) / '.temp' / f"wavlm_base_plus_sv_spksim_temp_{_r}.csv")
        df = pd.read_csv(_csv_path)
        if res is None:
            res = df
        else:
            res = pd.concat([res, df], axis = 0)
    print(f"WavLM Base Plus SV SpkSim:\n{res.describe()}")
    with open(str(Path(args.out_dir)  / "wavlm_base_plus_sv_spksim.txt"), "w") as f:
        print(f"WavLM Base Plus SV SpkSim:\n{res.describe()}", file = f)
    res.to_csv(str(Path(args.out_dir) / f"wavlm_base_plus_sv_spksim.csv"))
    print("Done!")


def run_eval(rank, args):
    suffix = args.ref_suffix
    ## Dataset
    ref_audio_paths = glob.glob(op.join(args.ref_dir, f'*.{suffix}'))
    ref_audio_path_dict = dict([(Path(p).stem, p) for p in ref_audio_paths])

    out_audio_paths = sorted(glob.glob(op.join(args.test_dir, "*.wav")))
    out_audio_paths = out_audio_paths[rank::args.num_proc]
    out_audio_path_dict = dict([(Path(p).stem, p) for p in out_audio_paths])
    print(f"Evaluation on len {len(out_audio_path_dict)} at rank {rank}")

    ## Model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_BASE_PLUS_SV)
    model = WavLMForXVector.from_pretrained(WAVLM_BASE_PLUS_SV)
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    device = args.gpus[rank % len(args.gpus)]
    model.to(device)

    res = []
    for _k, _out_path in tqdm.tqdm(out_audio_path_dict.items(), desc=f"[rank {rank}]"):
        _ref_path = ref_audio_path_dict.get(_k)
        assert _ref_path is not None
        _out_audio, _ = librosa.load(_out_path, sr=None)
        _ref_audio, _ = librosa.load(_ref_path, sr=None)
        audios = [_out_audio, _ref_audio]
        inputs = feature_extractor(audios, sampling_rate = 16000,  padding=True, return_tensors="pt").to(device)
        embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        similarity = cosine_sim(embeddings[0], embeddings[1]).item()
        res.append({"name": Path(_out_path).stem ,"similarity": similarity})

    df = pd.DataFrame(res)
    # print(df.describe())
    # with open(str(Path(args.out_dir) / '.temp' / "wavlm_base_plus_sv_spksim.txt"), "w") as f:
    #     print(f"WavLM Base Plus SV SpkSim:\n{df.describe()}", file = f)
    df.to_csv(str(Path(args.out_dir) / '.temp' / f"wavlm_base_plus_sv_spksim_temp_{rank}.csv"))

    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)