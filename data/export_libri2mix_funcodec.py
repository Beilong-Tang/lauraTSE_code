from funcodec.bin.codec_inference import Speech2Token
import torch
from typing import List
import librosa
import tqdm
import numpy as np
import os
import torch.multiprocessing as mp
import argparse
from pathlib import Path
import re

SEED = 1234

def get_source_list(file_path: str, ret_name=False):
    files = []
    names = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            l = line.replace("\n", "").split(" ")
            name = l[0]
            path = l[-1]
            files.append(path)
            names.append(name)
    if ret_name:
        return names, files
    return files

def list_to_files(arr: list, file_path):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    with open(file_path, "w") as f:
        for e in arr:
            if e.endswith("\n"):
                f.write(e)
            else:
                f.write(e + "\n")

def match_files(path):
    base_name = os.path.basename(path)
    pattern = re.compile(base_name)
    matching_files = [f for f in os.listdir(os.path.dirname(path)) if pattern.fullmatch(f)]
    return [os.path.join(os.path.dirname(path), f) for f in matching_files]

# def rms_normalize(audio):
#     rms = np.sqrt(np.mean(np.square(audio)))  # Calculate the RMS value
#     normalized_audio = audio / rms  # Normalize audio to unit RMS
#     return normalized_audio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scp_file',type=str, required = True)
    p.add_argument('--base_path', type = str, default = None)
    p.add_argument('--config', type=str, required = True)
    p.add_argument('--model', type=str, required= True)
    p.add_argument('--output', type=str, required= True)
    ##############
    # DDP config #
    ##############
    p.add_argument('--num_proc', type = int, default = 8)
    p.add_argument('--gpus', nargs="+", default = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok= True)
    mp.spawn(run, args=(args,), nprocs=args.num_proc, join=True)
    merge_scp(args)
    pass

def run(rank, args):
    names, paths = get_source_list(args.scp_file, ret_name= True)
    device = args.gpus[rank % len(args.gpus)]
    names = names[rank::args.num_proc]
    paths = paths[rank::args.num_proc]
    torch.cuda.set_device(torch.device(device))

    # Output path # ex. out/{rank}
    out_path = Path(args.output)
    if args.base_path is None:
        os.makedirs(out_path /str(rank), exist_ok= True)
    else:
        os.makedirs(out_path / 'utt', exist_ok = True)
    shape_file = open(out_path / f"{rank}_shape.scp", "w")
    scp_file = open(out_path / f"{rank}.scp", "w")
    

    # Initialize Model #
    model = Speech2Token(config_file =args.config, model_file = args.model, device='cuda')
    model.eval()
    print("[=== Start outputing to codec ===]")
    with torch.no_grad():
        for n, p in tqdm.tqdm(list(zip(names, paths)), desc= f"[rank{rank}]"):
            
            if args.base_path is not None:
                abs_path = str(Path(args.base_path) / p)
            else:
                abs_path = p

            # load audio 
            audio, _sr = librosa.load(abs_path, sr = None) # [T]
            assert _sr == 16000
            # audio = rms_normalize(audio) # Remove RMS normalization
            audio = torch.from_numpy(audio)
            audio = audio.to('cuda')
            audio = audio.unsqueeze(0).unsqueeze(0) # [1,1,T]

            # modeling
            codes = model(audio, run_mod = "encode")[0][0].permute(1,2,0).squeeze(0) #[T, n_q(32)]

            # saving
            file_name = Path(abs_path).stem + ".npy"
            if args.base_path is not None: 
                save_path = out_path / 'utt' / Path(p).parent / file_name
            else:
                save_path = out_path / str(rank) / file_name
                pass

            os.makedirs(save_path.parent, exist_ok=True)

            np.save(save_path, codes.cpu().numpy())
            scp_file.write(f"{n} {save_path}\n")

            codes_len = codes.size(0)
            shape_file.write(f"{n} {codes_len}\n")
    shape_file.close()
    shape_file.close()
    torch.cuda.empty_cache()

def _get(files)-> List[str]:
    res = []
    for p in files:
        names, paths = get_source_list(p, ret_name= True)
        for _n, _p in list(zip(names, paths)):
            res.append(f"{_n} {_p}\n")
    return res

def merge_scp(args):
    print("[ Merging scp outputs ]")
    out_path = Path(args.output)
    shape_files = match_files(f"{str(out_path)}/[0-9]*_shape.scp")
    scp_files = match_files(f"{str(out_path)}/[0-9]*.scp")
    shape_all = _get(shape_files)
    scp_all = _get(scp_files)
    list_to_files(shape_all, str(out_path / "all_shape.scp"))
    list_to_files(scp_all, str(out_path / "all.scp"))
    

if __name__ == "__main__":
    main()
    pass
