
from funcodec.modules.nets_utils import pad_list
import torch

class MaxLength():


    def __init__(self, field_list:list, max_len: 1000):
        """
        Data class to make sure that each data is under the maximum length 
        """
        self.field_list = field_list
        self.max_len = max_len
        pass 

    def __call__(self, data:dict):
        """
        data should be {name: [B,T,*], name_lens: [B], ...}
        """
        _constrain = False
        ## Iterate it to see if we need to apply max length
        for _key in self.field_list:
            if data[_key].size(1) > self.max_len:
                _constrain = True 
                break
        if _constrain is False:
            return data 
        else:
            _res_dict = {}
            for _key in self.field_list:
                res= []
                res_len = []
                batch = data[_key]
                batch_lens = data[_key+"_lengths"]
                for item, item_len in zip(batch, batch_lens): # [T,*]
                    item = item[:item_len.item()]
                    ## Apply maximum here
                    item = item[:self.max_len]
                    res.append(item)
                    res_len.append(item.size(0))
                res = pad_list(res, 0.0)
                _res_dict[_key] = res 
                _res_dict[_key+"_lengths"] = torch.tensor(res_len, dtype = torch.long)
            return _res_dict

# class CleanNoisyFilter():

#     def __init__(self):
#         """
#         Post process for data class that combines clean, noisy into data where each of them have the same length
#         """
#         pass 

#     def __call__(self, data:torch.Tensor, data_lengths: torch.Tensor):
#         res_clean_len = []
#         res_clean = []
#         res_noisy_len = []
#         res_noisy = []
#         for item, item_len in zip(data, data_lengths): # [T], 1'
#             _clean_noisy = item[:item_len.item()]
#             _each_length = int(item_len.item() / 2)
#             _clean = _clean_noisy[:_each_length]
#             _noisy = _clean_noisy[_each_length:]
#             assert len(_clean) == len(_noisy) 
#             res_clean_len.append(_each_length)
#             res_noisy_len.append(_each_length)
#             res_clean.append(_clean)
#             res_noisy.append(_noisy)
#         res_clean = pad_list(res_clean, 0.0)
#         res_noisy = pad_list(res_noisy, 0.0)
#         res_clean_len = torch.tensor(res_clean_len, dtype = torch.long)
#         res_noisy_len = torch.tensor(res_noisy_len, dtype = torch.long)
#         return res_clean, res_clean_len, res_noisy, res_noisy_len