import os
import os.path as op
import tqdm
import argparse
import wespeaker

import glob
import pandas as pd
from pathlib import Path


def main(args):
    model = wespeaker.load_model(args.language)
    model.set_gpu(args.device_id)
    audio_path_list = sorted(glob.glob(op.join(args.test_file, "*.wav")))
    ref_path_list = sorted(glob.glob(op.join(args.reference_file, "*.wav")))
    print("total audio len: ", len(audio_path_list))
    os.makedirs(op.dirname(args.output), exist_ok=True)
    res = []
    total = 0
    pbar = tqdm.tqdm(list(zip(audio_path_list, ref_path_list)))
    for idx, (output, ref) in enumerate(pbar):
        sim = model.compute_similarity(output, ref)
        res.append({"sim": sim})
        total += sim
        pbar.set_description(f"sim: {total/(idx+1):.2f}")
    df = pd.DataFrame(res)
    df.to_csv(args.output)
    print(df.describe())
    with open(str(Path(args.output).parent / "wespeaker_sim.log"), "w") as f:
        print(df.describe(), file= f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test_file",
        help="The path of directory of audio files ending with .wav to be transcripted",
    )
    parser.add_argument(
        "-r",
        "--reference_file",
        help="The path of directory of audio files ending with .wav to which are reference audio",
    )
    parser.add_argument(
        "-o", "--output", help="The output file containing the similarity "
    )
    parser.add_argument(
        "-l", "--language", help="The language, [english, chinese]", default="english"
    )
    parser.add_argument("-d", "--device_id", help="device id", type=int, default=0)
    args = parser.parse_args()
    main(args)
    pass

