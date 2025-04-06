# This script generates the noise
from typing import Dict
import numpy as np
from copy import deepcopy
import re
import ast

from dataset.helper import random_choice
from dataset.effects.bandwidth import bandwidth_limitation_config, bandwidth_limitation
from dataset.effects.packet_loss import packet_loss_config, packet_loss
from dataset.effects.clipping import clipping
from dataset.effects.codec import codec_compression
from dataset.effects.mix_noise import mix_noise
from dataset.effects.wind_noise import wind_noise
from dataset.effects.reverberation import add_reverberation
from utils.audio import read_audio
from utils.hinter import hint_once

from utils.rir_utils import estimate_early_rir
# Avaiable sampling rates for bandwidth limitation

AUGMENTATIONS = ("bandwidth_limitation", "clipping")


def select_sample(fs, utt2fs2scp: Dict[int, Dict[str, str]]):
    keys = list(filter(lambda x: x >= fs, list(utt2fs2scp.keys())))
    fs_ = random_choice(keys)
    uid = random_choice(list(utt2fs2scp[fs_].keys()))
    return uid, utt2fs2scp[fs_][uid], fs_  # uid, path, fs

##############################################
# Generate Configuration of different noises #
##############################################


def _process_one_sample(
    conf,
    fs,
    speech_length,
    use_wind_noise,
    wind_noise_dic,
    wind_noise_snr_range,
    noise_dic,
    snr_range,
    rir_dic,
    augmentations="none",
):
    if use_wind_noise:
        noise_uid, _, noise_fs = select_sample(
            fs,
            wind_noise_dic,
        )
        # wind-noise simulation config
        wn_conf = conf.wind_noise_config
        threshold = np.random.uniform(*wn_conf["threshold"])
        #
        hint_once(f"threshold {threshold}", "threshold")
        ratio = np.random.uniform(*wn_conf["ratio"])
        attack = np.random.uniform(*wn_conf["attack"])
        release = np.random.uniform(*wn_conf["release"])
        sc_gain = np.random.uniform(*wn_conf["sc_gain"])
        clipping_threshold = np.random.uniform(*wn_conf["clipping_threshold"])
        clipping = np.random.random() < wn_conf["clipping_chance"]
        augmentation_config = (
            "wind_noise("
            f"threshold={threshold},ratio={ratio},"
            f"attack={attack},release={release},"
            f"sc_gain={sc_gain},clipping={clipping},"
            f"clipping_threshold={clipping_threshold})/"
        )
        snr = np.random.uniform(*wind_noise_snr_range)
    else:
        noise_uid, noise, noise_fs = select_sample(
            fs,
            noise_dic,
        )
        augmentation_config = ""
        snr = np.random.uniform(*snr_range)
    if noise_uid is None:
        raise ValueError(f"Noise sample not found for fs={fs}+ Hz")

    # select a room impulse response (RIR)
    if (
        rir_dic is None
        or conf.prob_reverberation <= 0.0
        or np.random.rand() <= conf.prob_reverberation
    ):
        rir_uid, rir = None, None
    else:
        rir_uid, rir, rir_fs = select_sample(fs, rir_dic)

    # apply an additional augmentation
    if isinstance(augmentations, str) and augmentations == "none":
        if not use_wind_noise:
            augmentation_config = "none"
    else:
        for i, augmentation in enumerate(augmentations):
            this_aug = conf.augmentations[augmentation]
            if augmentation == "bandwidth_limitation":
                res_type, fs_new = bandwidth_limitation_config(fs=fs, res_type="random")
                augmentation_config += f"{augmentation}-{res_type}->{fs_new}"
            elif augmentation == "clipping":
                min_quantile = np.random.uniform(*this_aug["clipping_min_quantile"])
                max_quantile = np.random.uniform(*this_aug["clipping_max_quantile"])
                augmentation_config += (
                    f"{augmentation}(min={min_quantile},max={max_quantile})"
                )
            elif augmentation == "codec":
                # vbr_quality = np.random.uniform(*this_aug["vbr_quality"])
                # augmentation_config += f"{augmentation}(vbr_quality={vbr_quality})"
                codec_config = np.random.choice(this_aug["config"], 1)[0]
                format, encoder, qscale = (
                    codec_config["format"],
                    codec_config["encoder"],
                    codec_config["qscale"],
                )
                if encoder is not None and isinstance(encoder, list):
                    encoder = np.random.choice(encoder, 1)[0]
                if qscale is not None and isinstance(qscale, list):
                    qscale = np.random.randint(*qscale)
                augmentation_config += (
                    f"{augmentation}"
                    f"(format={format},encoder={encoder},qscale={qscale})"
                )

            elif augmentation == "packet_loss":
                packet_duration_ms = this_aug["packet_duration_ms"]
                packet_loss_indices = packet_loss_config(
                    speech_length,
                    fs,
                    packet_duration_ms,
                    this_aug["packet_loss_rate"],
                    this_aug["max_continuous_packet_loss"],
                )
                augmentation_config += (
                    f"{augmentation}"
                    f"(packet_loss_indices={packet_loss_indices},"
                    f"packet_duration_ms={packet_duration_ms})"
                )
            else:
                raise NotImplementedError(augmentation)

            # / is used for splitting multiple augmentation configuration
            if i < len(augmentations) - 1:
                augmentation_config += "/"
    meta = {
        "noise_uid": (noise_uid, noise_fs),
        "rir_uid": (None, None) if rir_uid is None else (rir_uid, rir_fs),
        "snr": snr,
        "augmentation": augmentation_config,
        "fs": fs,
        "length": speech_length,
    }
    return meta


def generate_augmentations_config(
    conf, fs: int, speech: np.ndarray, noise_dic, wind_noise_dic, rir_dic
) -> Dict[str, Dict]:
    """
    conf: config
    speech: [1,T]
    """
    speech_length = speech.shape[1]
    # get wind noise snr range and snr range
    wind_noise_snr_range = (
        conf.wind_noise_snr_low_bound,
        conf.wind_noise_snr_high_bound,
    )
    snr_range = (conf.snr_low_bound, conf.snr_high_bound)

    # Repeatly add augmentations
    use_wind_noise: bool = np.random.random() < conf.prob_wind_noise

    num_aug = random_choice(
        list(conf.num_augmentations.keys()),
        p=list(conf.num_augmentations.values()),
    )  # augmentation number

    if num_aug == 0:
        ## No augmentations
        aug = "none"
    else:
        
        aug = random_choice(
            list(conf.augmentations.keys()),
            p=conf.weight_augmentations,
            size=num_aug,
            replace=False,
        )  # list of augmentations
        while use_wind_noise and "clipping" in aug:
            aug = np.random.choice(
                list(conf.augmentations.keys()),
                p=conf.weight_augmentations,
                size=num_aug,
                replace=False,
            )

    info = _process_one_sample(
        conf,
        fs,
        speech_length,
        use_wind_noise=use_wind_noise,
        wind_noise_dic=wind_noise_dic,
        wind_noise_snr_range=wind_noise_snr_range,
        noise_dic=noise_dic,
        snr_range=snr_range,
        rir_dic=rir_dic,
        augmentations=aug,
    )
    info["speech"] = speech
    return info

#############################
# Augmentations per sample
#############################
def generate_from_config(info: dict, noise_dic, wind_noise_dic, rir_dic, force_1ch=True):
    fs = int(info["fs"])
    snr = float(info["snr"])
    noise_uid, noise_fs = info["noise_uid"]
    if noise_uid.startswith("wind_noise"):
        noise_path = wind_noise_dic[noise_fs][noise_uid]
    else:
        noise_path = noise_dic[noise_fs][noise_uid]
    speech_sample = info["speech"] # np [1,T]

    noise_sample = read_audio(noise_path, force_1ch=force_1ch, fs=fs)[0] # resampled noise [1,T]
    noisy_speech = deepcopy(speech_sample)

    # augmentation information, split by /
    augmentations = info["augmentation"].split("/")
    rir_uid, rir_fs = info["rir_uid"]
    if rir_uid != None:
        rir = rir_dic[rir_fs][rir_uid]
        rir_sample = read_audio(rir, force_1ch=force_1ch, fs=fs)[0]
        noisy_speech = add_reverberation(speech_sample, rir_sample)
        # make sure the clean speech is aligned with the input noisy speech
        early_rir_sample = estimate_early_rir(rir_sample, fs=fs)
        speech_sample = add_reverberation(speech_sample, early_rir_sample)
    else:
        noisy_speech = speech_sample
    pass
    
    # simulation with non-linear wind-noise mixing
    if noise_uid.startswith("wind_noise"):
        nuid, _ = info["noise_uid"]
        augmentation = [a for a in augmentations if a.startswith("wind_noise")]
        assert (
            len(augmentation) == 1
        ), f"Configuration for the wind-noise simulation is necessary: {augmentation} {nuid}"

        # threshold, ratio, attack, release, sc_gain, snr, clipping, clipping_threshold
        match = re.fullmatch(
            f"wind_noise\(threshold=(.*),ratio=(.*),attack=(.*),release=(.*),sc_gain=(.*),clipping=(.*),clipping_threshold=(.*)\)",
            augmentation[0],
        )
        (
            threshold_,
            ratio_,
            attack_,
            release_,
            sc_gain_,
            clipping_,
            clipping_threshold_,
        ) = match.groups()
        noisy_speech, noise_sample = wind_noise(
            noisy_speech,
            noise_sample,
            fs,
            float(threshold_),
            float(ratio_),
            float(attack_),
            float(release_),
            float(sc_gain_),
            bool(clipping_),
            float(clipping_threshold_),
            float(snr),
        )
    else:
        noisy_speech, noise_sample = mix_noise(
            noisy_speech, noise_sample, snr=snr
        )
     # apply an additional augmentation
    for augmentation in augmentations:
        if augmentation == "none" or augmentation == "":
            pass
        elif augmentation.startswith("wind_noise"):
            pass
        elif augmentation.startswith("bandwidth_limitation"):
            match = re.fullmatch(f"bandwidth_limitation-(.*)->(\d+)", augmentation)
            res_type, fs_new = match.groups()
            noisy_speech = bandwidth_limitation(
                noisy_speech, fs=fs, fs_new=int(fs_new), res_type=res_type
            )
        elif augmentation.startswith("clipping"):
            match = re.fullmatch(f"clipping\(min=(.*),max=(.*)\)", augmentation)
            min_, max_ = map(float, match.groups())
            noisy_speech = clipping(noisy_speech, min_quantile=min_, max_quantile=max_)
        elif augmentation.startswith("codec"):
            # match = re.fullmatch(f"codec\(vbr_quality=(.*)\)", augmentation)
            # vbr_quality_ = match.groups()[0]
            # noisy_speech = codec_compression(noisy_speech, fs, float(vbr_quality_))
            match = re.fullmatch(
                f"codec\(format=(.*),encoder=(.*),qscale=(.*)\)", augmentation
            )
            format, encoder, qscale = match.groups()
            noisy_speech = codec_compression(
                noisy_speech, fs, format=format, encoder=encoder, qscale=int(qscale)
            )

        elif augmentation.startswith("packet_loss"):
            match = re.fullmatch(
                f"packet_loss\(packet_loss_indices=(.*),packet_duration_ms=(.*)\)",
                augmentation,
            )
            packet_loss_indices_, packet_duration_ms_ = match.groups()
            packet_loss_indices_ = ast.literal_eval(
                packet_loss_indices_
            )  # convert string to list
            noisy_speech = packet_loss(
                noisy_speech, fs, packet_loss_indices_, int(packet_duration_ms_)
            )
        else:
            raise NotImplementedError(augmentation)

    length = int(info["length"])
    assert noisy_speech.shape[-1] == length, (info, noisy_speech.shape)

    # normalization
    scale = 0.9 / max(
        np.max(np.abs(noisy_speech)),
        np.max(np.abs(speech_sample)),
        np.max(np.abs(noise_sample)),
    )
    return speech_sample * scale, noisy_speech * scale
    
    
