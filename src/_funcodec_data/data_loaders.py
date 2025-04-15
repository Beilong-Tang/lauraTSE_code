# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import collections
import functools
import logging
import numbers
import re
from typing import Mapping
from typing import Union

import numpy as np

from funcodec.fileio.npy_scp import NpyScpReader
from funcodec.datasets.dataset import sound_loader

from utils.audio import read_audio
from utils.utils import AttrDict

from utils.hinter import hint_once

DATA_TYPES = {
    "sound": dict(
        func=sound_loader,
        kwargs=["float_dtype"],
        help="Audio format types which supported by sndfile wav, flac, etc."
        "\n\n"
        "   utterance_id_a a.wav\n"
        "   utterance_id_b b.wav\n"
        "   ...",
    ),
    "npy": dict(
        func=NpyScpReader,
        kwargs=[],
        help="Npy file format."
        "\n\n"
        "   utterance_id_A /some/where/a.npy\n"
        "   utterance_id_B /some/where/b.npy\n"
        "   ...",
    )
}