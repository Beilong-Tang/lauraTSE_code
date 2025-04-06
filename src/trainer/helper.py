import os
import os.path as op
import re
import torch
from typing import Union
from pathlib import Path
import pickle


def dict_to_str(dictionary):
    res = ""
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            value = value.item()

        if isinstance(value, float):
            res += f"{key} : {value:.3f} | "
        else:
            res += f"{key} : {value} | "

    return res


def add_result(result: dict, res: dict):
    for key in res.keys():
        if result.get(key) == None:
            result[key] = res[key]
        elif isinstance(res[key], str):
            result[key] = res[key]
        else:
            result[key] = result[key] + res[key]
    return result


def normalize_result(result: dict, length: int):
    for key in result.keys():
        if isinstance(result[key], str):
            pass
        else:
            result[key] = result[key] / length
    return result


def save_stats(path, content):
    ## TODO: Finish Saving the stats
    dirname = op.dirname(path)
    files = [f for f in os.listdir(dirname) if (f.endswith(".pkl") and f.startswith("stats_epoch"))]
    for f in files:
        os.remove(str(Path(dirname) / f))
    with open(path, "wb") as f:
        pickle.dump(content, f)

def save(path, content, epoch, max_ckpt=1):
    # if len(files_path) >= max_ckpt:
    if max_ckpt == -1:
        ##save
        torch.save(content, path)
        return
    if max_ckpt == None:
        max_ckpt = 1
    dirname = op.dirname(path)
    files = sorted(
        [f for f in os.listdir(dirname) if (f.endswith(".pth") and "best" not in f)],
        key=lambda x: int(re.search(r"[0-9]+", x).group()),
    )
    files = list(filter(lambda x: int(re.search(r"[0-9]+", x).group()) < epoch, files))
    files_path = [op.join(dirname, f) for f in files]
    if len(files_path) >= max_ckpt:
        try:
            os.remove(files_path[0])
        except FileNotFoundError as e:
            print("saving error")
            print(e)
    torch.save(content, path)


def load_ckpt(ckpt_dir: str):
    """
    ckpt dir: the directory to the checkpoint folder
    continue_from: if string, return the string (path)
        if boolean, if true: return the latest epoch ckpt path in ckpt_dir or None if there is no ckpt available
                    if false: return None
        if None: return the latest epoch ckpt path in ckpt_dir or None if there is no ckpt available
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        return None

    files = sorted(
        [f for f in os.listdir(ckpt_dir) if (f.endswith(".pth") and "best" not in f)],
        key=lambda x: int(re.search(r"[0-9]+", x).group()),
    )
    if len(files) == 0:
        return None
    else:
        return op.join(ckpt_dir, files[-1])
