import torch
import torchaudio
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.models.wav2vec2 import components
from torchaudio.pipelines import HUBERT_BASE

from .data import Tokenizer

FINETUNING_HUBERT_CONFIG = {
    "encoder_projection_dropout": 0,
    "encoder_attention_dropout": 0,
    "encoder_ff_interm_dropout": 0.1,
    "encoder_dropout": 0,
    "encoder_layer_drop": 0.1,  # In torchaudio: 0.05
    "mask_prob": 0.75,  # In torchaudio: 0.65
    "mask_channel_prob": 0.5,
    "mask_channel_length": 10,  # In torchaudio and fairseq: 64. This is the value for pretraining.
    "num_classes": 500,  # Number of classes during HuBERT pretraining.
}


class HuBERTPhoneme(nn.Module, PyTorchModelHubMixin):
    def __init__(self, freeze_encoder: bool = True, ctc_training: bool = False) -> None:
        """Initialize the model.

        Parameters
        ----------
        freeze_encoder : bool, optional
            Whether to freeze the Transformer encoder of HuBERT, by default True.
            The convolutional layers are always frozen.
        """
        super().__init__()
        self.model = torchaudio.models.hubert_pretrain_base(**FINETUNING_HUBERT_CONFIG)
        self.model.wav2vec2.load_state_dict(HUBERT_BASE.get_model().state_dict())
        self.aux = nn.Linear(HUBERT_BASE._params["encoder_embed_dim"], Tokenizer(with_blank=ctc_training).vocab_size)
        self.freeze_encoder = freeze_encoder
        self.ctc_training = ctc_training

    def forward(self, waveforms: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Extract logits during training, with masking."""
        if self.freeze_encoder:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, lengths)
                padding_mask = components._get_padding_mask(x, out_len)
                x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
                x, _ = self.model.mask_generator(x, padding_mask)
                x = self.model.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, lengths)
                padding_mask = components._get_padding_mask(x, out_len)
            x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
            x, _ = self.model.mask_generator(x, padding_mask)
            x = self.model.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        logits = self.aux(x)
        return logits, out_len

    def inference(self, waveforms: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Extract logits during inference. No masking is applied."""
        x, out_len = self.model.wav2vec2(waveforms, lengths)
        logits = self.aux(x)
        return logits, out_len

    @torch.jit.export
    def extract_features(self, waveforms: Tensor, lengths: Tensor | None = None) -> tuple[list[Tensor], Tensor | None]:
        """Extract features from intermediate layers. No masking is applied."""
        x, out_len = self.model.wav2vec2.extract_features(waveforms, lengths)
        x.append(self.aux(x[-1]))
        return x, out_len

    def train(self, mode: bool = True) -> "HuBERTPhoneme":
        """Override the train method to set the encoder in eval mode if it is frozen."""
        if self.freeze_encoder:
            self.model.wav2vec2.eval()
        else:
            self.model.wav2vec2.train(mode)
        self.aux.train(mode)
        return self


class FrameLevelCrossEntropyLoss(nn.Module):
    """Frame-level cross-entropy loss. Operates on logits and targets with the same length."""

    def __init__(self, ignore_index: int) -> None:
        """Initialize the loss function.

        Parameters
        ----------
        ignore_index : int
            The index to ignore when computing the loss.
            Set this as the padding index in the target sequence.
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, logit_lengths: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        """Compute the frame-level cross-entropy loss.

        Parameters
        ----------
        logits : Tensor
            The model's output logits. Do not apply softmax. (N, T, C)
        logit_lengths : Tensor
            The lengths of the logits. (N,)
        targets : Tensor
            The target sequence. (N, T)
        target_lengths : Tensor
            The lengths of the target sequence. (N,)

        Returns
        -------
        Tensor
            Computed loss.
        """
        assert logits.ndim == 3 and targets.ndim == 2
        assert all(logit_lengths == target_lengths)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.flatten(), ignore_index=self.ignore_index)


class CTCLossOnLogits(nn.Module):
    def __init__(self, blank: int) -> None:
        super().__init__()
        self.blank = blank

    def forward(self, logits: Tensor, logit_lengths: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        assert logits.ndim == 3 and targets.ndim == 2
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)
        return F.ctc_loss(log_probs, targets, logit_lengths, target_lengths, blank=self.blank, zero_infinity=True)
