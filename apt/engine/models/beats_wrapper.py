import logging
import math
import torch
import torchaudio
from packaging import version
from torch import nn, Tensor

import sys

sys.path.append("/data/home/eey340/WORKPLACE/LAM/src/beats")
from BEATs import BEATs, BEATsConfig


log = logging.getLogger(__name__)


class BEATsWrapper(nn.Module):
    r"""A wrapper for BEATs."""

    cfg = {
        "encoder_layers": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12,
        "activation_fn": "gelu",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.05,
        "dropout_input": 0.0,
        "layer_norm_first": False,
        "conv_bias": False,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "relative_position_embedding": True,
        "num_buckets": 320,
        "max_distance": 800,
        "gru_rel_pos": True,
        "deep_norm": True,
        "input_patch_size": 16,
        "layer_wise_gradient_decay_ratio": 0.6,
        "embed_dim": 512,
        "finetuned_model": True,
        "predictor_dropout": 0.0,
        "predictor_class": 527,
    }

    def __init__(
        self,
        hidden_size: int = 768,
        # padding
        target_length: int = 998,
        # other
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        cfg = BEATsConfig(self.cfg)
        model = BEATs(cfg)
        model.predictor = None  # remove classifier from the head

        self.audio_cfg = cfg
        self.target_length = target_length
        self.hidden_size = hidden_size
        self.device = device
        self.model = model.to(device)

    def from_pretrained(
        self,
        ckpt_path: str = "/path/to/ckpt.pt",
    ) -> str:
        # Download the default pretrained checkpoint.model = laion_clap.CLAP_Module(enable_fusion=False)
        log.info(f"Load pre-trained checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        del checkpoint

    def forward(self, batch: Tensor) -> Tensor:
        padding_mask = torch.zeros_like(batch).bool()

        feats, _ = self.model.extract_features(batch, padding_mask=padding_mask)
        return feats

    def get_audio_embedding(self, batch):
        r"""Get audio embeddings with various lengths."""
        batch = batch.unsqueeze(1)
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

        padding_mask = (
            torch.zeros_like(input_vit).bool().squeeze(1).all(-1).to(self.device)
        )

        feats, _ = self._get_audio_emb(
            input_vit.to(self.device), padding_mask=padding_mask
        )

        return {
            "patch_embedding": feats,  # align with audiomae
            "batch_size": B,
        }

    def _get_audio_emb(self, fbank, padding_mask):
        features = self.model.patch_embedding(fbank)

        features = features.reshape(features.shape[0], features.shape[1], -1)

        features = features.transpose(1, 2)  # (B, L, D)
        features = self.model.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.model.forward_padding_mask(features, padding_mask)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        x = self.model.dropout_input(features)

        x, layer_results = self.model.encoder(
            x,
            padding_mask=padding_mask,
        )

        return x, padding_mask

    def read_audio(self, audio_path):
        audio_input, sr = torchaudio.load(audio_path)
        resample_fn = torchaudio.transforms.Resample(sr, 16000)
        audio_input_16khz = resample_fn(audio_input)
        return audio_input_16khz, sr

    def preprocess_audio(self, audio_path):
        audio_input, sr = self.read_audio(audio_path)
        fbank = self.model.preprocess(audio_input)
        return fbank

    @classmethod
    def create_model(
        cls,
        hidden_size=768,
        target_length=998,
        ckpt_path="/data/EECS-MachineListeningLab/jinhua/pretrained_model_zoo/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        device=torch.device("cpu"),
    ):
        model = BEATsWrapper(
            hidden_size=hidden_size,
            target_length=target_length,
            device=device,
        )

        if ckpt_path is not None:
            model.from_pretrained(ckpt_path)

        return model


if __name__ == "__main__":
    # load the fine-tuned checkpoints
    model = BEATsWrapper.create_model(target_length=998, device=torch.device("cuda"))

    # predict the classification probability of each class
    wav_pth = "/data/EECS-MachineListeningLab/datasets/AudioSet/audios/eval_segments/YK1BOsSMEX-I.wav"
    fbank = model.preprocess_audio(wav_pth)
    fbank = torch.cat([fbank, fbank, fbank], dim=1)
    feat = model.get_audio_embedding(fbank)
    import pdb

    pdb.set_trace()
