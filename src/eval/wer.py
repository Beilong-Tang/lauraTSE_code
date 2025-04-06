"""
Command line tool to output the transcription of a list of audio files.
"""

import whisper
import argparse
import os.path as op
import glob
import tqdm
import os
import sys
import jiwer
import torch.multiprocessing as mp
import torch
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import multiprocessing as mp

from utils.utils import merge_content


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test_file",
        help="The path of audio files ending with .wav to be transcripted",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        default=None,
        help="If None, just run the whisper inference on test_file without evaluating wer. The path to the output transcript, the content should be name|sentence\n name2|sentence ",
    )
    parser.add_argument(
        "-o", "--output", help="The output file containing the transcript"
    )
    parser.add_argument(
        "-m", "--model", help="The model to use, default: base", default="base"
    )
    # DDP #
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"wer args: {args}")
    os.makedirs(op.dirname(args.output), exist_ok=True)
    transcribe(args)
    if args.reference is not None:
        print("running wer calculation")
        wer(args)

def _transcribe_single_process(rank, args) -> str:
    device = args.gpus[rank % len(args.gpus)]
    torch.cuda.set_device(device)
    model = whisper.load_model(args.model, device=device)
    model.to(device)
    audio_path_list = sorted(glob.glob(op.join(args.test_file, "*.wav")))
    audio_path_list = audio_path_list[rank :: args.num_proc]
    res = []
    output_file = f"{args.output}.temp_{rank}.txt"
    for audio in tqdm.tqdm(audio_path_list, desc=f"[rank {rank}/{args.num_proc}]"):
        result = model.transcribe(audio)
        res.append(f"{os.path.basename(audio)}|{result['text']}\n")
    with open(output_file, "w") as f:
        f.writelines(res)
    return output_file

def transcribe(args):

    inputs = []
    for i in range(0, args.num_proc):
        inputs.append((i, args))
    with mp.Pool(processes=args.num_proc) as pool:
        results = pool.starmap(_transcribe_single_process, inputs)
    print(f"results {results}")
    merge_content(results, args.output)

def wer(args):
    def _get_result(path, separator="|"):
        """
        Args:
            path: transcript txt path
            separator: the separator for each line to separate name and text value
        Returns:
            dict where key is name and value is the transcript
        """
        res_dict = {}
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                _idx = line.find(separator)
                name = line[:_idx]
                text = line[_idx+1:]
                res_dict[name] = text
        return res_dict
    output_dict = _get_result(args.output)
    ref_dict = _get_result(args.reference)
    total_wer = 0.0
    bad_audio=[]
    res = []
    for key, value in tqdm.tqdm(ref_dict.items()):
        if output_dict.get(key) is None:
            bad_audio.append(key)
        else:
            value_hat = output_dict.get(key)
            wer = jiwer.wer(value, value_hat)
            res.append({"id": key, "wer": wer})
            total_wer += wer
    avg_wer = total_wer / len(ref_dict)
    info = f"average wer: {avg_wer}, missing audio is {bad_audio}"
    print(info)
    with open(Path(args.output).parent / "wer.txt", "w") as f:
        print(info, file = f)
    pd.DataFrame(res).to_csv(str(Path(args.output).parent / "wer.csv"), index=False)

    
if __name__ == "__main__":
    main()

