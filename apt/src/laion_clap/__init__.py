# Thanks to LAION_CLAP repo.
import torch
import torchaudio
from torch import nn, Tensor
from typing import Union, List, Optional
from transformers import RobertaTokenizer
from .src.laion_clap.clap_module import create_model
from .src.laion_clap.training.data import get_audio_features


class CLAPWrapper(nn.Module):

    def __init__(
        self,
        pretrained_path: str,
        amodel: str = 'HTSAT-base',  # 'HTSAT-tiny'or 'PANN-14'
        sampling_rate: int = 48000,
        duration: int = 480000,
        train: bool = False,
        enable_fusion: bool = False,
        use_cuda: bool = False,
    ) -> None:
        super().__init__()
        precision = 'fp32'
        tmodel = 'roberta'  # the best text encoder in our training
        enable_fusion = False  # False if you do not want to use the fusion model
        fusion_type = 'aff_2d' if enable_fusion else 'None'
        pretrained = pretrained_path
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.device = torch.device(
            'cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        self.model, self.model_cfg = create_model(amodel,
                                                  tmodel,
                                                  pretrained,
                                                  precision=precision,
                                                  device=self.device,
                                                  enable_fusion=enable_fusion,
                                                  fusion_type=fusion_type)

        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))).exp().to(
                self.device)

        if not train:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def get_text_embeddings(self, texts):
        r"""Get a single/list of sentences and return text embeddings.
        Args:
            text: str or list of str. e.g., 
            emb = get_text_embeddings("Dog"), or
            emb = get_text_embeddings(["Dog", "Cat"])
        """
        text_tokens = self.tokenizer(texts).to(self.device)
        embed = self.model.get_text_embedding(text_tokens)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        return embed

    @torch.no_grad()
    def get_audio_embeddings(self,
                             audio_files: list,
                             resample: bool = True) -> Tensor:
        r"""Get list of audio files and returns audio embeddings."""
        sr = self.sampling_rate if resample else None
        audios = self._preprocess_audio(audio_files, sample_rate=sr)

        return self._batch_extract_audio_features(audios)

    def _batch_extract_audio_features(self, wavs: List[Tensor]) -> Tensor:
        audio_dict_list = list()
        for w in wavs:
            audio_dict = get_audio_features(
                dict(),
                w,
                self.duration,
                data_truncating='fusion',
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'])
            audio_dict_list.append(audio_dict)
        embed = self.model.get_audio_embedding(audio_dict_list)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        return embed

    def _preprocess_audio(self,
                          audio_files: list,
                          sample_rate: Optional[int] = None) -> list:
        r"""Return a list of torch.Tensor, shape = (, n_time_samples)."""
        audios = list()
        # Load and reample audio
        for file in audio_files:
            wav, sr = torchaudio.load(file)
            if sample_rate and (sr != sample_rate):
                wav = torchaudio.functional.resample(wav,
                                                     orig_freq=sr,
                                                     new_freq=sample_rate)
            wav = wav.squeeze().to(self.device)
            audios.append(wav)
        return audios

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def forward(self, batch, modality: str):
        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode

        if (modality == "audio"):
            with torch.no_grad():
                audio_dict_list = []
                # assert self.sampling_rate == 32000, "We only support 32000 sampling rate"
                # batch: [bs, 1, t-samples]
                # adapt to the pre-trained checkpoint by resampling
                batch = torchaudio.functional.resample(
                    batch, orig_freq=self.sampling_rate, new_freq=48000)  #
                for waveform in self.batch_to_list(batch):
                    audio_dict = {}
                    audio_dict = get_audio_features(
                        audio_dict,
                        waveform,
                        480000,
                        data_truncating='fusion',
                        data_filling='repeatpad',
                        audio_cfg=self.model_cfg['audio_cfg'])
                    audio_dict_list.append(audio_dict)
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict_list)
        elif (modality == "text"):
            with torch.no_grad():
                # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                text_data = self.tokenizer(batch)
                embed = self.model.get_text_embedding(text_data)
        return embed

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return result

    def compute_embedding_similarity(self, audio_embeddings, text_embeddings):
        r"""Compute similarity between text and audio embeddings."""

        similarity = self.logit_scale * text_embeddings @ audio_embeddings.T
        return similarity.T

    def extract_feature_and_calculate_similarity(
            self,
            audios: List[Union[str, Tensor]],
            texts: List[str],
            enable_softmax: bool = False,
            resample: bool = False) -> list:
        if isinstance(audios[0], str):
            audio_emb = self.get_audio_embeddings(audios, resample=resample)
        else:
            audio_emb = self._batch_extract_audio_features(texts)

        text_emb = self.get_text_embeddings(texts)

        if not enable_softmax:
            return self.compute_embedding_similarity(audio_emb, text_emb)
        else:
            return self.compute_embedding_similarity(audio_emb,
                                                     text_emb).softmax(dim=-1)


# import os
# import wget
# import torch
# import librosa
# from transformers import RobertaTokenizer

# from src.laion_clap.clap_module.factory import load_state_dict
# from src.laion_clap.clap_module import create_model
# from src.laion_clap.training.data import get_audio_features
# from src.laion_clap.training.data import int16_to_float32, float32_to_int16

# class CLAP_Module(torch.nn.Module):

#     def __init__(self,
#                  enable_fusion=False,
#                  device=None,
#                  amodel='HTSAT-tiny',
#                  tmodel='roberta') -> None:
#         """Initialize CLAP Model

#         Parameters
#         ----------
#         enable_fusion: bool
#             if true, it will create the fusion clap model, otherwise non-fusion clap model (default: false)
#         device: str
#             if None, it will automatically detect the device (gpu or cpu)
#         amodel: str
#             audio encoder architecture, default: HTSAT-tiny
#         tmodel: str
#             text encoder architecture, default: roberta
#         """
#         super(CLAP_Module, self).__init__()
#         if device is None:
#             device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#         precision = 'fp32'

#         if enable_fusion:
#             fusion_type = 'aff_2d'
#             model, model_cfg = create_model(amodel,
#                                             tmodel,
#                                             precision=precision,
#                                             device=device,
#                                             enable_fusion=enable_fusion,
#                                             fusion_type=fusion_type)
#         else:
#             model, model_cfg = create_model(amodel,
#                                             tmodel,
#                                             precision=precision,
#                                             device=device,
#                                             enable_fusion=enable_fusion)
#         self.enable_fusion = enable_fusion
#         self.model = model
#         self.model_cfg = model_cfg
#         self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')

#     def tokenizer(self, text):
#         result = self.tokenize(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=77,
#             return_tensors="pt",
#         )
#         return {k: v.squeeze(0) for k, v in result.items()}

#     def load_ckpt(self, ckpt=None, model_id=-1):
#         """Load the pretrained checkpoint of CLAP model

#         Parameters
#         ----------
#         ckpt: str
#             if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. \n
#             For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
#         model_id:
#             if model_id is specified, you can download our best ckpt, as:
#                 id = 0 --> 630k non-fusion ckpt \n
#                 id = 1 --> 630k+audioset non-fusion ckpt \n
#                 id = 2 --> 630k fusion ckpt \n
#                 id = 3 --> 630k+audioset fusion ckpt \n
#             Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
#         """
#         download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
#         download_names = [
#             '630k-best.pt', '630k-audioset-best.pt', '630k-fusion-best.pt',
#             '630k-audioset-fusion-best.pt'
#         ]
#         if ckpt is not None:
#             print(f'Load the specified checkpoint {ckpt} from users.')
#         else:
#             print(f'Load our best checkpoint in the paper.')
#             if model_id == -1:
#                 model_id = 3 if self.enable_fusion else 1
#             package_dir = os.path.dirname(os.path.realpath(__file__))
#             weight_file_name = download_names[model_id]
#             ckpt = os.path.join(package_dir, weight_file_name)
#             if os.path.exists(ckpt):
#                 print(f'The checkpoint is already downloaded')
#             else:
#                 print('Downloading laion_clap weight files...')
#                 ckpt = wget.download(download_link + weight_file_name,
#                                      os.path.dirname(ckpt))
#                 print('Download completed!')
#         print('Load Checkpoint...')
#         ckpt = load_state_dict(ckpt, skip_params=True)
#         self.model.load_state_dict(ckpt)
#         param_names = [n for n, p in self.model.named_parameters()]
#         for n in param_names:
#             print(n, "\t", "Loaded" if n in ckpt else "Unloaded")

#     def get_audio_embedding_from_filelist(self, x, use_tensor=False):
#         """get audio embeddings from the audio file list

#         Parameters
#         ----------
#         x: List[str] (N,):
#             an audio file list to extract features, audio files can have different lengths (as we have the feature fusion machanism)
#         use_tensor: boolean:
#             if True, it will return the torch tensor, preserving the gradient (default: False).
#         Returns
#         ----------
#         audio_embed : numpy.darray | torch.Tensor (N,D):
#             audio embeddings that extracted from audio files
#         """
#         self.model.eval()
#         audio_input = []
#         for f in x:
#             # load the waveform of the shape (T,), should resample to 48000
#             audio_waveform, _ = librosa.load(f, sr=48000)
#             # quantize
#             audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
#             audio_waveform = torch.from_numpy(audio_waveform).float()
#             temp_dict = {}
#             temp_dict = get_audio_features(
#                 temp_dict,
#                 audio_waveform,
#                 480000,
#                 data_truncating='fusion'
#                 if self.enable_fusion else 'rand_trunc',
#                 data_filling='repeatpad',
#                 audio_cfg=self.model_cfg['audio_cfg'],
#                 require_grad=audio_waveform.requires_grad)
#             audio_input.append(temp_dict)
#         audio_embed = self.model.get_audio_embedding(audio_input)
#         if not use_tensor:
#             audio_embed = audio_embed.detach().cpu().numpy()
#         return audio_embed

#     def get_audio_embedding_from_data(self, x, use_tensor=False):
#         """get audio embeddings from the audio data

#         Parameters
#         ----------
#         x: np.darray | torch.Tensor (N,T):
#             audio data, must be mono audio tracks.
#         use_tensor: boolean:
#             if True, x should be the tensor input and the output will be the tesnor, preserving the gradient (default: False).
#             Note that if 'use tensor' is set to True, it will not do the quantize of the audio waveform (otherwise the gradient will not be preserved).
#         Returns
#         ----------
#         audio embed: numpy.darray | torch.Tensor (N,D):
#             audio embeddings that extracted from audio files
#         """
#         self.model.eval()
#         audio_input = []
#         for audio_waveform in x:
#             # quantize
#             if not use_tensor:
#                 audio_waveform = int16_to_float32(
#                     float32_to_int16(audio_waveform))
#                 audio_waveform = torch.from_numpy(audio_waveform).float()
#             temp_dict = {}
#             temp_dict = get_audio_features(
#                 temp_dict,
#                 audio_waveform,
#                 480000,
#                 data_truncating='fusion'
#                 if self.enable_fusion else 'rand_trunc',
#                 data_filling='repeatpad',
#                 audio_cfg=self.model_cfg['audio_cfg'],
#                 require_grad=audio_waveform.requires_grad)
#             audio_input.append(temp_dict)
#         audio_embed = self.model.get_audio_embedding(audio_input)
#         if not use_tensor:
#             audio_embed = audio_embed.detach().cpu().numpy()
#         return audio_embed

#     def get_text_embedding(self, x, tokenizer=None, use_tensor=False):
#         """get text embeddings from texts

#         Parameters
#         ----------
#         x: List[str] (N,):
#             text list
#         tokenizer: func:
#             the tokenizer function, if not provided (None), will use the default Roberta tokenizer.
#         use_tensor: boolean:
#             if True, the output will be the tesnor, preserving the gradient (default: False).
#         Returns
#         ----------
#         text_embed : numpy.darray | torch.Tensor (N,D):
#             text embeddings that extracted from texts
#         """
#         self.model.eval()
#         if tokenizer is not None:
#             text_input = tokenizer(x)
#         else:
#             text_input = self.tokenizer(x)
#         text_embed = self.model.get_text_embedding(text_input)
#         if not use_tensor:
#             text_embed = text_embed.detach().cpu().numpy()
#         return text_embed
