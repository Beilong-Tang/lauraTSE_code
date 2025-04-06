# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Abstract task module."""
import argparse
import logging
from dataclasses import dataclass
from distutils.version import LooseVersion
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.multiprocessing
import torch.nn
import torch.optim
from typeguard import check_argument_types

from funcodec import __version__
from funcodec.datasets.dataset import AbsDataset
from funcodec.datasets.dataset import DATA_TYPES
# from funcodec.datasets.dataset import ESPnetDataset
from _funcodec_data._espnet_dataset import DMESPnetDataset
from funcodec.datasets.iterable_dataset import IterableESPnetDataset
from funcodec.iterators.abs_iter_factory import AbsIterFactory
from funcodec.iterators.chunk_iter_factory import ChunkIterFactory
from funcodec.iterators.multiple_iter_factory import MultipleIterFactory
from funcodec.iterators.sequence_iter_factory import SequenceIterFactory
from funcodec.util_funcs.collect_stats import collect_stats
from funcodec.optimizers.sgd import SGD
from funcodec.optimizers.fairseq_adam import FairseqAdam
from funcodec.samplers.build_batch_sampler import BATCH_TYPES
from funcodec.samplers.build_batch_sampler import build_batch_sampler
from funcodec.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from funcodec.schedulers.noam_lr import NoamLR
from funcodec.schedulers.warmup_lr import WarmupLR
from funcodec.schedulers.tri_stage_scheduler import TriStageLR
from funcodec.torch_utils.load_pretrained_model import load_pretrained_model
from funcodec.torch_utils.model_summary import model_summary
from funcodec.torch_utils.pytorch_version import pytorch_cudnn_version
from funcodec.torch_utils.set_all_random_seed import set_all_random_seed
from funcodec.train.abs_espnet_model import AbsESPnetModel
from funcodec.train.class_choices import ClassChoices
from funcodec.train.distributed_utils import DistributedOption
from funcodec.train.trainer import Trainer
from funcodec.utils import config_argparse
from funcodec.utils.build_dataclass import build_dataclass
from funcodec.utils.cli_utils import get_commandline_args
from funcodec.utils.get_default_kwargs import get_default_kwargs
from funcodec.utils.nested_dict_action import NestedDictAction
from funcodec.utils.types import humanfriendly_parse_size_or_none
from funcodec.utils.types import int_or_none
from funcodec.utils.types import str2bool
from funcodec.utils.types import str2triple_str
from funcodec.utils.types import str_or_int
from funcodec.utils.types import str_or_none
from funcodec.utils.wav_utils import calc_shape, generate_data_list
from funcodec.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

try:
    import wandb
except Exception:
    wandb = None

if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
    pass
else:
    pass

optim_classes = dict(
    adam=torch.optim.Adam,
    fairseq_adam=FairseqAdam,
    adamw=torch.optim.AdamW,
    sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)
if LooseVersion(torch.__version__) >= LooseVersion("1.10.0"):
    # From 1.10.0, RAdam is officially supported
    optim_classes.update(
        radam=torch.optim.RAdam,
    )
try:
    import torch_optimizer

    optim_classes.update(
        accagd=torch_optimizer.AccSGD,
        adabound=torch_optimizer.AdaBound,
        adamod=torch_optimizer.AdaMod,
        diffgrad=torch_optimizer.DiffGrad,
        lamb=torch_optimizer.Lamb,
        novograd=torch_optimizer.NovoGrad,
        pid=torch_optimizer.PID,
        # torch_optimizer<=0.0.1a10 doesn't support
        # qhadam=torch_optimizer.QHAdam,
        qhm=torch_optimizer.QHM,
        sgdw=torch_optimizer.SGDW,
        yogi=torch_optimizer.Yogi,
    )
    if LooseVersion(torch_optimizer.__version__) < LooseVersion("0.2.0"):
        # From 0.2.0, RAdam is dropped
        optim_classes.update(
            radam=torch_optimizer.RAdam,
        )
    del torch_optimizer
except ImportError:
    pass
try:
    import apex

    optim_classes.update( 
        fusedadam=apex.optimizers.FusedAdam,
        fusedlamb=apex.optimizers.FusedLAMB,
        fusednovograd=apex.optimizers.FusedNovoGrad,
        fusedsgd=apex.optimizers.FusedSGD,
    )
    del apex
except ImportError:
    pass
try:
    import fairscale
except ImportError:
    fairscale = None

try:
    from funcodec.optimizers.lazy_adam import LazyAdamW
    optim_classes.update(
        lazy_adamw=LazyAdamW
    )
except ImportError:
    LazyAdamW = None

scheduler_classes = dict(
    ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
    noamlr=NoamLR,
    warmuplr=WarmupLR,
    tri_stage=TriStageLR,
    cycliclr=torch.optim.lr_scheduler.CyclicLR,
    onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
    CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
)
# To lower keys
optim_classes = {k.lower(): v for k, v in optim_classes.items()}
scheduler_classes = {k.lower(): v for k, v in scheduler_classes.items()}


@dataclass
class IteratorOptions:
    preprocess_fn: callable
    collate_fn: callable
    data_path_and_name_and_type: list
    shape_files: list
    batch_size: int
    batch_bins: int
    batch_type: str
    max_cache_size: float
    max_cache_fd: int
    distributed: bool
    num_batches: Optional[int]
    num_iters_per_epoch: Optional[int]
    train: bool


def build_sequence_iter_factory(
            args: argparse.Namespace, iter_options: IteratorOptions, mode: str
    ) -> AbsIterFactory:    
        # assert check_argument_types()

        dataset = DMESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
            conf_dm_noise=args.conf_dm_noise
        )

        if Path(
                Path(iter_options.data_path_and_name_and_type[0][0]).parent, "utt2category"
        ).exists():
            utt2category_file = str(
                Path(
                    Path(iter_options.data_path_and_name_and_type[0][0]).parent,
                    "utt2category",
                )
            )
        else:
            utt2category_file = None
        batch_sampler = build_batch_sampler(
            type=iter_options.batch_type,
            shape_files=iter_options.shape_files,
            fold_lengths=args.fold_length,
            batch_size=iter_options.batch_size,
            batch_bins=iter_options.batch_bins,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
            drop_last=args.drop_last,
            min_batch_size=torch.distributed.get_world_size()
            if iter_options.distributed
            else 1,
            utt2category_file=utt2category_file, # Does not matter here
        )

        batches = list(batch_sampler) # [("a","b"), ("c","d","e"), ("x","y")]
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]

        bs_list = [len(batch) for batch in batches]

        logging.info(f"[{mode}] dataset:\n{dataset}")
        logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
        logging.info(
            f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
            f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
        )

        if args.scheduler == "tri_stage" and mode == "train":
            args.max_update = len(bs_list) * args.max_epoch
            logging.info("Max update: {}".format(args.max_update))

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]


        return SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
        )
