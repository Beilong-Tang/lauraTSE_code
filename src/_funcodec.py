import torch

from funcodec.train.distributed_utils import DistributedOption
from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.iterators.sequence_iter_factory import SequenceIterFactory



def init_sequence_iter_factory(args, rank, mode) -> SequenceIterFactory:
    distributed_option = DistributedOption()
    distributed_option.distributed = True
    distributed_option.dist_rank = rank
    distributed_option.local_rank = rank

    distributed_option.dist_world_size = torch.distributed.get_world_size()
    iter_option = Text2AudioGenTask.build_iter_options(args, distributed_option, mode)
    return Text2AudioGenTask.build_sequence_iter_factory(args, iter_option, mode)


def init_dm_sequence_iter_factory(args, rank, mode) -> SequenceIterFactory:
    """
    Dynamic Sequence Iterator Factory
    If args.preprocess is speficified, we use the preprocess_fn
    """
    distributed_option = DistributedOption()
    distributed_option.distributed = True
    distributed_option.dist_rank = rank
    distributed_option.local_rank = rank

    distributed_option.dist_world_size = torch.distributed.get_world_size()
    iter_option = Text2AudioGenTask.build_iter_options(args, distributed_option, mode)
    # iter_option.preprocess_fn = 
    from _funcodec_data.build_sequence_iter import build_sequence_iter_factory
    return build_sequence_iter_factory(args, iter_option, mode)

from funcodec.tasks.text2audio_generation import text_encoder_choices, codec_encoder_choices
from model.laura_model_only_clean import LauraGenModelOnlyClean
from funcodec.torch_utils.initialize import initialize
import os

def build_model(args):
    input_size = args.input_size

    # 1. Text Encoder
    if args.text_encoder is not None:
        text_encoder_class = text_encoder_choices.get_class(args.text_encoder)
        text_encoder = text_encoder_class(input_size=input_size, **args.text_encoder_conf)
    else:
        text_encoder = None

    # 2. Codec Encoder
    if args.codec_encoder is not None:
        codec_encoder_class = codec_encoder_choices.get_class(args.codec_encoder)
        codec_encoder = codec_encoder_class(
            input_size=args.model_conf["codec_conf"]["codebook_dim"],
            **args.codec_encoder_conf
        )
    else:
        codec_encoder = None

    # 3. Build model
    token_list = []
    if args.token_list is not None:
        if isinstance(args.token_list, list):
            token_list = args.token_list
        elif os.path.exists(args.token_list):
            for line in open(args.token_list, "rt"):
                token = line.strip()
                token_list.append(token)
        else:
            raise TypeError("If token_list is not None, it must be list or str.")
    # model_class = model_choices.get_class(args.model)
    model = LauraGenModelOnlyClean(
        input_size=input_size,
        vocab_size=len(token_list),
        token_list=token_list,
        text_encoder=text_encoder,
        codec_encoder=codec_encoder,
        **args.model_conf,
    )

    # 10. Initialize
    if args.init is not None:
        initialize(model, args.init)

    return model