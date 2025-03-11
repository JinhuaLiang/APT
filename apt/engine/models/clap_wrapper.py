import logging
import math
import torch
import librosa
import torchaudio
import numpy as np
from packaging import version
from torch import nn, Tensor
from transformers import AutoProcessor, ClapAudioModel, ClapModel
from transformers import ClapAudioConfig, ClapAudioModel

log = logging.getLogger(__name__)


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


# class CLAP(nn.Module):
#     r"""A wrapper for CLAP."""

#     def __init__(
#         self,
#         hidden_size: int = 768,
#         # padding
#         target_length: int = 1024,
#         # other
#         device: torch.device = torch.device("cpu"),
#     ):
#         super().__init__()

#         model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")

#         self.audio_cfg = model.model_cfg["audio_cfg"]
#         self.target_length = target_length
#         self.hidden_size = hidden_size
#         self.device = device
#         self.model = model.to(device)

#     def from_pretrained(self, ckpt_path: str) -> str:
#         # Download the default pretrained checkpoint.model = laion_clap.CLAP_Module(enable_fusion=False)
#         log.info(f"Load pre-trained checkpoint from: {ckpt_path}")
#         self.model.load_ckpt(ckpt_path)
#         self.model = self.model.model.audio_branch

#     def forward(self, batch: Tensor) -> Tensor:
#         return self.model.forward_features(
#             batch.to(self.device), mixup_lambda=None, device=self.device
#         )

#     def get_audio_embedding(self, batch):
#         r"""Get audio embeddings with various lengths."""
#         B, C, T, F = batch.size()
#         n_seg = math.ceil(T / self.target_length)

#         input_vit = torch.zeros(
#             B * n_seg, C, self.target_length, F, device=batch.device
#         )
#         for i in range(n_seg):
#             T_stt = self.target_length * i
#             T_end = min(T_stt + self.target_length, T)
#             input_vit[B * i : B * (i + 1), :, :, :] += batch[:, :, T_stt:T_end, :]

#         output = self.model.forward_features(input_vit.to(self.device))
#         output["batch_size"] = B

#         return output

#     def read_audio(self, audio_path):
#         waveform, sr = torchaudio.load(audio_path)
#         waveform = int16_to_float32(float32_to_int16(waveform))  # quantize wav
#         return waveform, sr

#     def get_mel(self, audio_data):
#         # mel shape: (n_mels, T)
#         mel_tf = torchaudio.transforms.MelSpectrogram(
#             sample_rate=self.audio_cfg["sample_rate"],
#             n_fft=self.audio_cfg["window_size"],
#             win_length=self.audio_cfg["window_size"],
#             hop_length=self.audio_cfg["hop_size"],
#             center=True,
#             pad_mode="reflect",
#             power=2.0,
#             norm=None,
#             onesided=True,
#             n_mels=self.audio_cfg["mel_bins"],
#             f_min=self.audio_cfg["fmin"],
#             f_max=self.audio_cfg["fmax"],
#         ).to(audio_data.device)

#         mel = mel_tf(audio_data)

#         # we use log mel spectrogram as input
#         mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
#         return mel.T  # (T, n_mels)

#     def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
#         # no mixup
#         if filename2 == None:
#             waveform, sr = self.read_audio(filename)
#         # mixup
#         else:
#             waveform1, sr = self.read_audio(filename)
#             waveform2, _ = self.read_audio(filename2)

#             if waveform1.shape[1] != waveform2.shape[1]:
#                 if waveform1.shape[1] > waveform2.shape[1]:
#                     # padding
#                     temp_wav = torch.zeros(1, waveform1.shape[1])
#                     temp_wav[0, 0 : waveform2.shape[1]] = waveform2
#                     waveform2 = temp_wav
#                 else:
#                     # cutting
#                     waveform2 = waveform2[0, 0 : waveform1.shape[1]]

#             waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2

#         fbank = self.get_mel(waveform)

#         target_length = self.target_length
#         n_frames = fbank.shape[0]

#         p = target_length - n_frames

#         # cut and pad
#         if p > 0:
#             m = torch.nn.ZeroPad2d((0, 0, 0, p))
#             fbank = m(fbank)
#         elif p < 0:
#             fbank = fbank[0:target_length, :]

#         return fbank

#     def preprocess_audio(self, audio_path):
#         try:
#             fbank = self._wav2fbank(audio_path, None, 0)
#         except:
#             fbank = torch.zeros(self.target_length, 128) + 0.01
#             print("there is an error in loading audio")

#         # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
#         print(fbank.shape)
#         return fbank.unsqueeze(dim=0)

#     @classmethod
#     def create_model(
#         cls,
#         hidden_size=768,
#         target_length=1024,
#         ckpt_path=None,
#         device=torch.device("cpu"),
#     ):
#         model = CLAP(
#             hidden_size=hidden_size,
#             target_length=target_length,
#             device=device,
#         )

#         if ckpt_path is not None:
#             model.from_pretrained(ckpt_path)

#         return model


# if __name__ == "__main__":
#     model = CLAP.create_model(
#         ckpt_path="/data/EECS-MachineListeningLab/jinhua/pretrained_model_zoo/laion_clap/music_speech_audioset_epoch_15_esc_89.98.pt",
#         device=torch.device("cuda"),
#     )
#     wav_path = "/data/EECS-MachineListeningLab/datasets/Clotho/audios/evaluation/woodsbirds.wav"
#     import pdb

#     pdb.set_trace()


class CLAPWrapper(nn.Module):
    r"""A wrapper for CLAP."""

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-fused",
        hidden_size: int = 768,
        # padding
        target_length: int = 1001,
        # other
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        # Initializing a ClapAudioConfig with laion/clap-htsat-fused style configuration
        configuration = ClapAudioConfig()

        # Initializing a ClapAudioModel (with random weights) from the laion/clap-htsat-fused style configuration
        model = ClapAudioModel(configuration)
        processor = AutoProcessor.from_pretrained(model_name)

        self.audio_cfg = configuration
        self.target_length = target_length
        self.hidden_size = hidden_size
        self.device = device
        self.model = model.to(device)
        self.processor = processor

    def from_pretrained(self, ckpt_path: str = "laion/clap-htsat-fused") -> str:
        # Download the default pretrained checkpoint.model = laion_clap.CLAP_Module(enable_fusion=False)
        log.info(f"Load pre-trained checkpoint from: {ckpt_path}")
        model = ClapModel.from_pretrained(ckpt_path)
        self.model = model.audio_model.to(self.device)

    def forward(self, batch: Tensor) -> Tensor:
        inputs = self.processor(audios=batch, return_tensors="pt")
        outputs = model(**inputs)
        return outputs

    def get_audio_embedding(self, batch):
        r"""Get audio embeddings with various lengths."""
        batch, is_longer = (
            batch["input_features"],
            batch["is_longer"],
        )
        B, C, T, F = batch.size()
        n_seg = math.ceil(T / self.target_length)

        input_vit = torch.zeros(
            B * n_seg, C, self.target_length, F, device=batch.device
        )
        for i in range(n_seg):
            T_stt = self.target_length * i
            T_end = min(T_stt + self.target_length, T)
            T_len = T_end - T_stt  # avoid edge case that T_stt + self.target_length > T
            input_vit[B * i : B * (i + 1), :, :T_len, :] += batch[:, :, T_stt:T_end, :]

        output = self.model(input_vit.to(self.device), is_longer=is_longer)

        # last_hidden_state is (N, D, F, T)
        return {
            "patch_embedding": output.last_hidden_state.permute(
                0, 1, 3, 2
            ),  # (N, D, T, F)
            "batch_size": B,
        }

    def read_audio(self, audio_path):
        waveform, sr = librosa.load(
            audio_path, sr=self.processor.feature_extractor.sampling_rate
        )
        waveform = waveform.reshape(1, -1)  # Make it (1,T) or (N,T)
        waveform = int16_to_float32(float32_to_int16(waveform))  # quantize wav
        return waveform, self.processor.feature_extractor.sampling_rate

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = self.read_audio(filename)
        # mixup
        else:
            waveform1, sr = self.read_audio(filename)
            waveform2, _ = self.read_audio(filename2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = np.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2

        ret = self.processor(audios=waveform, return_tensors="pt")

        return ret

    def preprocess_audio(self, audio_path):
        try:
            ret = self._wav2fbank(audio_path, None, 0)
        except:
            ret = {
                "input_features": torch.zeros(4, 1001, 64) + 0.01,
                "is_longer": False,
            }
            print("there is an error in loading audio")

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # fbank = fbank.unsqueeze(dim=0)
        return ret

    @classmethod
    def create_model(
        cls,
        hidden_size=768,
        target_length=1001,
        model_name="laion/clap-htsat-fused",
        device=torch.device("cpu"),
    ):
        model = CLAPWrapper(
            model_name=model_name,
            hidden_size=hidden_size,
            target_length=target_length,
            device=device,
        )

        if model_name is not None:
            model.from_pretrained(model_name)

        return model


if __name__ == "__main__":
    # load the fine-tuned checkpoints
    model = CLAPWrapper.create_model(target_length=1001, device=torch.device("cuda"))

    # predict the classification probability of each class
    wav_pth = "/data/EECS-MachineListeningLab/datasets/ESC-50/audio/3-166422-A-11.wav"
    fbank = model.preprocess_audio(wav_pth)
    _f = torch.cat([fbank["input_features"], fbank["input_features"]], dim=2)
    fbank["input_features"] = _f

    # fbank = torch.cat([fbank, fbank, fbank], dim=1)
    feat = model.get_audio_embedding(fbank)
    import pdb

    pdb.set_trace()
