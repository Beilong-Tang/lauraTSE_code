import argparse
import os
import sys 
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import random
import yaml
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.schedulers.warmup_lr import WarmupLR
from funcodec.torch_utils.load_pretrained_model import load_pretrained_model

from _funcodec import init_sequence_iter_factory, init_dm_sequence_iter_factory, build_model

from trainer.abs_trainer import Trainer
from utils.utils import setup_logger, init, AttrDict

def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED


## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, args):
    args = AttrDict(**vars(args))
    #########
    ## DDP ##
    #########
    if "SLURM_PROCID" in os.environ:
        rank = args.rank
        device = args.gpu
    else:
        setup(rank, args.world_size, args.dist_backend)
        device = rank % torch.cuda.device_count()
        pass
    torch.cuda.set_device(device)
    setup_seed(args.seed, rank)

    #####################
    # LauraGPT Specific #
    #####################
    args.ngpu = args.world_size
    
    l = setup_logger(args.log, rank)
    l.info("logging initialized succesully")
    l.info(args)
    l.info(f"rank {rank} of world_size {args.world_size} started...")
    l.info("setup model")
    ## load laura gpt model
    model: nn.Module = build_model(args)
    model.cuda()
    l.info(f"model {model} is intialized")
    l.info(f"model parameters: {sum(p.numel() for p in model.parameters())}")
    l.info(f"Decoder LM parameters: {sum(p.numel() for p in model.codec_lm.parameters())}")
    for p in args.init_param:
        l.info(f"Loading pretrained params from {p}")
        load_pretrained_model(
            model=model,
            init_param=p,
            ignore_init_mismatch=True,
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            map_location=f"cuda:{torch.cuda.current_device()}",
        )
    ## If finetune or not, if we do, load the ckpt
    if args.fine_tune != "":
        l.info(f"# FINE TUNING # from ckpt {args.fine_tune}")
        _ckpt = torch.load(args.fine_tune, map_location="cuda")
        l.info(f"pretrained for epochs {_ckpt['epoch']}")
        model.load_state_dict(_ckpt['model_state_dict'])
        pass

    model = DDP(model, device_ids=[args.gpu])
    ## optimizer
    optim = init(torch.optim, args.optim, model.parameters())
    ## scheduler
    assert args.scheduler == "warmuplr"
    scheduler = WarmupLR(optim, **args.scheduler_conf)
    l.info(f"scheduler {scheduler} and optim {optim} is initialized")
    ## setup dataloader
    ### Initialized iter factory

    ## Check if the conf_dm_noise config is specified
    ## If specified, use dynamic mixing for the noise
    if args.conf_dm_noise is None:
        train_iter = init_sequence_iter_factory(args, rank, "train")
    else:
        print("init training dataset!")
        train_iter = init_dm_sequence_iter_factory(args, rank, 'train')
    print("init validation dataset")
    val_iter = init_sequence_iter_factory(args, rank, "valid")
    print("done!!")

    print("start training")
    ## ckpt_dir
    trainer = Trainer(
        model,
        train_iter,
        val_iter,
        optim,
        scheduler,
        config=args,
        ckpt_dir=args.ckpt_path,
        rank=rank,
        logger=l,
        resume = args.resume
    )
    l.info("starting training!")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=str, help="Output of the log")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--resume", type=str, nargs="?", const="")
    parser.add_argument("--fine_tune", type=str, nargs="?", const="") ## Determines if we finetue or not. 
    ##############
    # DDP Config #
    ##############
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    args = parser.parse_args()
    print(f"DEBUG: resume path {args.resume}")
    if args.config is not None:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        for k, v in config.items():
            args.__setattr__(k, v)
    ###################
    ## Running Slurm ##
    ###################
    if "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = args.rank % torch.cuda.device_count()
        print(f"running slurm on world size {args.world_size} and device num: {torch.cuda.device_count()}")
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        main(-1, args)
        pass
    #################
    ## Running DDP ##
    #################
    else:
        print("running ddp")
        args.world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
