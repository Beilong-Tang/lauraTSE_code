#
# Wrapper for SpeechBert Score: https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics 
#

import os 
import sys
import os.path as op
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import glob
from pathlib import Path
import argparse
import tqdm
import librosa
print("Importing")
from discrete_speech_metrics import SpeechBERTScore
print("Finished Importing")


def parse_argss():
    p = argparse.ArgumentParser()
    # Model specific
    p.add_argument("--model", default='hubert-base')
    p.add_argument("--layer", default = 11)

    # Data specific
    p.add_argument('--test_dir', type = str, required=True)
    p.add_argument('--ref_dir', type = str, required=True)

    p.add_argument('--ref_suffix', type = str, default = 'wav')

    # Output specific
    p.add_argument('--out_dir',type=str, required=True, help = "the directory to put the log results into")

    return p.parse_args()

def main(args):
    metrics = SpeechBERTScore(sr=16000,
                              model_type=args.model,
                              layer=args.layer,
                              use_gpu=True)
    
    suffix = args.ref_suffix

    ref_audio_paths = glob.glob(op.join(args.ref_dir, f'*.{suffix}'))
    ref_audio_path_dict = dict([(Path(p).stem, p) for p in ref_audio_paths])

    out_audio_paths = glob.glob(op.join(args.test_dir, "*.wav"))
    out_audio_path_dict = dict([(Path(p).stem, p) for p in out_audio_paths])
    print(f"Evaluation on len {len(out_audio_path_dict)}")

    res = []
    for _k, _out_path in tqdm.tqdm(out_audio_path_dict.items()):
        _ref_path = ref_audio_path_dict.get(_k)
        assert _ref_path is not None
        _out_audio, _ = librosa.load(_out_path, sr=None)
        _ref_audio, _ = librosa.load(_ref_path, sr=None)
        precision, _, _ = metrics.score(_ref_audio, _out_audio)
        res.append({"name": Path(_out_path).stem ,"precision": precision})
    
    df = pd.DataFrame(res)
    print(df.describe())
    with open(str(Path(args.out_dir) / "speech_bert.txt"), "w") as f:
        print(f"Speech Bert Score:\n{df.describe()}", file = f)
    df.to_csv(str(Path(args.out_dir) / "speech_bert_results.csv"))
if __name__ == "__main__":
    args = parse_argss()
    main(args)
    pass