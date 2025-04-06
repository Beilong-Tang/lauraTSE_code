from typing import Dict, List, Union
from collections import defaultdict
import numpy as np 
from pathlib import Path
def get_scp_dict(scp:str, ignore_prefix: Union[List, None]=None, base_path = None) -> Dict[int, Dict[str, str]]:
    """
    scp: path to scp path where scp contains three columns: uid, sr, path
    ignore_prefix: uid prefix to be ignored
    base_path is specified when the scp contains relative path
    """
    speech_dic = defaultdict(dict)
    with open(scp, "r") as f:
        for line in f:
            uid, fs, audio_path = line.strip().split()
            ## add base_path to audio_path
            if base_path is not None:
                audio_path = str(Path(base_path) / audio_path)
            ## ignore some components
            if ignore_prefix is not None:
                if any(uid.startswith(pref) for pref in ignore_prefix):
                    continue
            speech_dic[int(fs)][uid] = audio_path
    return speech_dic

def random_choice(arr: List, p: List[float] = None, size: int=None, replace=False):
    """
    Randomly choose one element from the array.
    """
    return np.random.choice(arr, p=p, size=size, replace=replace)

def log_dic(audio_dic, log, label):
    info = ""
    info += (
        f"[{label}] Total length: {sum(len(value) for value in audio_dic.values())}\n"
    )
    for k, v in audio_dic.items():
        info += f"[{label}] {k} - len: {len(v)}\n"
    log(info)
