import random

from torchaudio._extension.utils import torch


def pad_list(xs, pad_value=0):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


class Collate:
    """
    Collator for dynamic mixing and padding
    """

    def __init__(self, max_len, max_hist=20) -> None:
        """
        max_hist: saves only the previous <max_hist> unused datas
        """
        self.max_len = max_len
        ## save the previous unused samples
        self._hist = []
        self.max_hist = max_hist

    def __call__(self, batch):
        """ """
        max_len = self.max_len
        mix_speech_arr = [a[0] for a in batch]  # [T1, T2, ...]
        clean_speech_arr = [a[1] for a in batch]  # [T1, T2, ...]
        mix_clean_speech = list(zip(mix_speech_arr, clean_speech_arr))
        random.shuffle(mix_clean_speech)  # shuffle batch

        total_len = 0
        res_clean = []
        res_mix = []

        for i, (mix, clean) in enumerate(mix_clean_speech):
            if total_len + len(clean) > max_len:
                res_mix.append(mix[: max_len - total_len])
                res_clean.append(clean[: max_len - total_len])

                # append unused samples to history #
                self._hist = self._hist + batch[i + 1 :]
                self._hist = self._hist[-self.max_hist :]
                break
            total_len += len(clean)
            res_mix.append(mix)
            res_clean.append(clean)

        # if the current batch data is not enough to fill in the maximum len, check the hist
        if total_len < max_len:
            if len(self._hist) > 0:
                for i, hist_item in enumerate(self._hist):
                    mix = hist_item[0]
                    clean = hist_item[1]
                    if total_len + len(mix) > max_len:
                        res_mix.append(mix[: max_len - total_len])
                        res_clean.append(clean[: max_len - total_len])
                        break
                    total_len += len(clean)
                    res_mix.append(mix)
                    res_clean.append(clean)
                pass
                self._hist = self._hist[i + 1 :]
        ## TODO: pad them into a the same length
        assert len(res_mix) == len(res_clean)
        lens = [len(r) for r in res_mix]

        return (
            pad_list(res_mix),
            pad_list(res_clean),
            torch.tensor(lens, dtype=torch.long),
        )
