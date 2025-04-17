# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

import argparse
import concurrent.futures
import glob
import os
from pathlib import Path
import sys 
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


# /public/home/qinxy/bltang/ml_framework_slurm/recipes/DNSMOS


class ComputeScore:
    def __init__(self, model_dir: str, personalized=True) -> None:
        if personalized:
            self.onnx_sess = ort.InferenceSession(
                os.path.join(os.path.abspath(model_dir), "pDNSMOS/sig_bak_ovr.onnx")
            )
        else:
            self.onnx_sess = ort.InferenceSession(
                os.path.join(os.path.abspath(model_dir), "DNSMOS/sig_bak_ovr.onnx")
            )
        self.is_personalized = personalized
        self.p808_onnx_sess = ort.InferenceSession(
            os.path.join(os.path.abspath(model_dir), "DNSMOS", "model_v8.onnx")
        )

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, input_fs, fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, self.is_personalized
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {"filename": fpath, "len_in_sec": actual_audio_len / fs, "sr": fs}
        clip_dict["num_hops"] = num_hops
        clip_dict["OVRL_raw"] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict["SIG_raw"] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict["BAK_raw"] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict["OVRL"] = np.mean(predicted_mos_ovr_seg)
        clip_dict["SIG"] = np.mean(predicted_mos_sig_seg)
        clip_dict["BAK"] = np.mean(predicted_mos_bak_seg)
        clip_dict["P808_MOS"] = np.mean(predicted_p808_mos)
        return clip_dict

    def calculate(self, file_list: list, sampling_rate, output_csv):
        """
        calcluate and write dnsmos to a file
        """
        rows = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(self.__call__, clip, sampling_rate): clip
                for clip in file_list
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_url), total=len(file_list)
            ):
                clip = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (clip, exc))
                else:
                    rows.append(data)
        df = pd.DataFrame(rows)
        
        ## Print avg SIG BAK DNSMOS info first 
        sig_avg = df['SIG'].mean()
        bak_avg = df['BAK'].mean()
        ovrl_avg= df['OVRL'].mean()
        info = f"SIG:{sig_avg}|BAK:{bak_avg}|OVRL:{ovrl_avg}"
        print(info)
        with open(str(Path(output_csv).parent / "dnsmos.log"), "w") as f:
            print(info, file= f)
        df.to_csv(output_csv)


def main(args):
    models = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []
    # p808_model_path = os.path.join("DNSMOS", "model_v8.onnx")

    # if args.personalized_MOS:
    #     primary_model_path = P_PRIMARY_MODEL_PATH
    # else:
    #     primary_model_path = NP_PRIMARY_MODEL_PATH

    compute_score = ComputeScore(model_dir = args.model_dir, personalized=args.personalized_MOS)

    # rows = []
    clips = []
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    # is_personalized_eval = args.personalized_MOS
    # desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(args.testset_dir, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)
    compute_score.calculate(clips, SAMPLING_RATE, args.csv_path)

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     future_to_url = {
    #         executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip
    #         for clip in clips
    #     }
    #     for future in tqdm(concurrent.futures.as_completed(future_to_url)):
    #         clip = future_to_url[future]
    #         try:
    #             data = future.result()
    #         except Exception as exc:
    #             print("%r generated an exception: %s" % (clip, exc))
    #         else:
    #             rows.append(data)

    # df = pd.DataFrame(rows)
    # if args.csv_path:
    #     csv_path = args.csv_path
    #     df.to_csv(csv_path)
    # else:
    #     print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/public/home/qinxy/bltang/ml_framework_slurm/recipes/DNSMOS",
        help="DNS directory containing DNSMOS and pDNSMOS ckpt",
    )
    parser.add_argument(
        "-t",
        "--testset_dir",
        default=".",
        help="Path to the dir containing audio clips in .wav to be evaluated",
    )
    parser.add_argument(
        "-o", "--csv_path", default=None, help="Dir to the csv that saves the results"
    )
    parser.add_argument(
        "-p",
        "--personalized_MOS",
        action="store_true",
        help="Flag to indicate if personalized MOS score is needed or regular",
    )

    args = parser.parse_args()

    main(args)

