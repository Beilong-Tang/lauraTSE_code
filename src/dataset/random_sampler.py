from typing import List, Dict 
import numpy as np

class RandomSampler:
    def __init__(
        self, chunk_size: int, arr: List[str], resume_epoch=0, seed=0, shuffle=True
    ) -> None:
        self.random = np.random.default_rng(seed)  # Control the seed
        if shuffle:
            self.random.shuffle(arr)  # Randomly shuffle the array
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.arr = arr
        self.resume_epoch = resume_epoch

        self._start_idx = (resume_epoch * chunk_size) % len(arr)

    def get_chunk(self) -> List[str]:
        """
        This method should be called after each epoch to get the chunk
        """
        res = self.arr[self._start_idx : self._start_idx + self.chunk_size]
        remain = self.chunk_size - len(res)
        if remain > 0:
            for i in range(remain):
                res.append(self.arr[i % len(self.arr)])
            self._start_idx = remain % len(self.arr)
        elif remain == 0:
            self._start_idx = self._start_idx + self.chunk_size
        else:
            raise Exception(f"[RandomSampler] remain cannot be smaller than 0!")
        if self.shuffle:
            self.random.shuffle(res)
        assert len(res) == self.chunk_size
        return res


class RandomManager:
    def __init__(
        self,
        utt2fs2scp: Dict[int, Dict[str, str]],
        epoch_num: int,
        seed=0,
        resume_epoch=0,
    ) -> None:
        chunk_sizes = self._init_chunk_size(len(utt2fs2scp.keys()), epoch_num)
        self.random_chunk:Dict[int, RandomSampler] = {} # List[int, RandomSampler]
        for i, (fs, map) in enumerate(utt2fs2scp.items()):
            self.random_chunk[fs] = RandomSampler(
                chunk_sizes[i],
                list(map.keys()),
                resume_epoch=resume_epoch,
                seed=seed,
                shuffle=True,
            )
        self.set_epoch()

    def _init_chunk_size(self, world_size: int, total_num: int):
        split_size = total_num // world_size
        remainder = total_num % world_size
        split_sizes = [split_size] * world_size  # List[int]
        for i in range(remainder):
            split_sizes[i] += 1
        assert sum(split_sizes) == total_num
        return split_sizes

    def set_epoch(self) -> Dict[int, List[str]]:
        result = {}
        for fs, rand_sampler in self.random_chunk.items():
            result[fs] = rand_sampler.get_chunk()
        self.result = result
    
    def get_uid_and_freq(self, idx):
        fs = None
        uid = None
        for key, value in self.result.items():
            if len(value) <= idx:
                idx -= len(value)
            else:
                uid = value[idx]
                fs = key
                return fs, uid
        raise Exception(f"[get_fs_and_uid] index error, cannot reach here!")
