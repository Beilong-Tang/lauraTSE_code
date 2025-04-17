import torch.distributed
import logging

HINTED = set()


def hint_once(content, uid, rank=None):
    """
    ranks: which rank to output log
    """
    _cur_rank = None 
    if torch.distributed.is_initialized():
        _cur_rank = torch.distributed.get_rank()

    if (rank is None) or (_cur_rank is None) or _cur_rank == rank:
        if uid not in HINTED:
            if _cur_rank is not None:
                print(f"[HINT_ONCE] Rank {_cur_rank}: {content}")
            else:
                print(f"[HINT_ONCE] {content}")
            HINTED.add(uid)

def check_hint(uid) -> bool:
    """
    Check if uid is in already hinted
    """
    return uid in HINTED

