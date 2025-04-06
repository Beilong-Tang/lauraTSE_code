import datetime
import os
import logging
from typing import List
import yaml
import random
import numpy as np
import torch

from argparse import Namespace


def init(module, config, *args, **kwargs):
    return getattr(module, config["type"])(*args, **kwargs, **config["args"])


def setup_logger(log_dir: str, rank: int, out=True):
    if out:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
            handlers=[
                logging.FileHandler(f"{log_dir}/{now}.log"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
            handlers=[logging.StreamHandler()],
        )
    logger = logging.getLogger()
    logger.info("logger initialized")
    return Logger(logger, rank)


def update_args(args: Namespace, config_file_path: str):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        args.__setattr__(k, v)
    return args


class Logger:
    def __init__(self, log: logging.Logger, rank: int):
        self.log = log
        self.rank = rank

    def info(self, msg: str):
        if self.rank == 0:
            self.log.info(msg)

    def debug(self, msg: str):
        if self.rank == 0:
            self.log.debug(msg)
        pass

    def warning(self, msg: str):
        self.log.warning(f"rank {self.rank} - {msg}")
        pass

    def error(self, msg: str):
        self.log.error(f"rank {self.rank} - {msg}")
        pass

    def critical(self, msg: str):
        self.log.critical(f"rank {self.rank} - {msg}")

        pass


def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED


def get_env(config_path: str):
    """
    config_path: str to yaml config
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(**config)
    return config


class AttrDict(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None

    def __getitem__(self, key):
        return self.__getattribute__(key)


def get_source_list(file_path: str, ret_name=False):
    """
    the file content should be uid, (fs), path
    """
    files = []
    names = []
    with open(file_path, "r") as f:
        for line in f.readlines():
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



def merge_content(files: List[str], save_path: str):
    """
    Merge contents from each file in files to save_path
    It drops empty lines.
    """
    def _get(files)-> List[str]:
        res = []
        for p in files:
            with open(p, "r") as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue 
                    res.append(line)
        return res
    res = _get(files)
    with open(save_path, "w") as f:
        f.writelines(res)
