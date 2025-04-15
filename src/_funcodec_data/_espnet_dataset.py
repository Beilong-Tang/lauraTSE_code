# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import collections
import functools
import logging
import numbers
import re
from typing import Mapping
from typing import Union

import h5py
import kaldiio
import numpy as np
import torch
from typeguard import check_argument_types

from funcodec.fileio.npy_scp import NpyScpReader
from funcodec.fileio.rand_gen_dataset import FloatRandomGenerateDataset
from funcodec.fileio.rand_gen_dataset import IntRandomGenerateDataset
from funcodec.fileio.read_text import load_num_sequence_text
from funcodec.fileio.read_text import read_2column_text
from funcodec.fileio.sound_scp import SoundScpReader
from funcodec.fileio.read_text import read_2column_text
from funcodec.datasets.dataset import ESPnetDataset
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import Union

from utils.hinter import hint_once

from .data_loaders import DATA_TYPES


class DMESPnetDataset(ESPnetDataset):
    """Pytorch Dataset class for ESPNet.

    Added only the DATA_TYPES the mix standing for mixing

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        ... uttid, data = dataset['uttid']
        {'input': per_utt_array, 'output': per_utt_array}
    """
    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
        max_cache_fd: int = 0,
        conf_dm_noise='conf_dm_noise/simulation_train.yaml',
    ):
        self.conf_dm_noise = conf_dm_noise # This line has to be written before the super().__init__() method
        super().__init__(path_name_type_list, preprocess, float_dtype, int_dtype, max_cache_size, max_cache_fd)
        
    

    def _build_loader(
        self, path: str, loader_type: str
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
        """
        for key, dic in DATA_TYPES.items():
            # e.g. loader_type="sound"
            # -> return DATA_TYPES["sound"]["func"](path)
            if re.match(key, loader_type):
                kwargs = {}
                for key2 in dic["kwargs"]:
                    if key2 == "loader_type":
                        kwargs["loader_type"] = loader_type
                    elif key2 == "float_dtype":
                        kwargs["float_dtype"] = self.float_dtype
                    elif key2 == "int_dtype":
                        kwargs["int_dtype"] = self.int_dtype
                    elif key2 == "max_cache_fd":
                        kwargs["max_cache_fd"] = self.max_cache_fd
                    
                    ## Add the dynamic mixing here
                    elif key2 == "spk_dict_path":
                        kwargs['spk_dict_path'] = self.spk_dict_path 
                    elif key2 == "mel_config":
                        kwargs['mel_config'] = self.mel_config
                    elif key2 =="ref_ds":
                        kwargs['ref_ds'] = self.ref_ds
                    else:
                        raise RuntimeError(f"Not implemented keyword argument: {key2}")

                func = dic["func"]
                try:
                    return func(path, **kwargs) # returns a value
                except Exception:
                    if hasattr(func, "__name__"):
                        name = func.__name__
                    else:
                        name = str(func)
                    logging.error(f"An error happened with {name}({path})")
                    raise
        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")