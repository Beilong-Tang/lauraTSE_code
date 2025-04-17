import jiwer 
import tqdm 
from pathlib import Path
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()

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
    


args = parse_args()
wer(args)