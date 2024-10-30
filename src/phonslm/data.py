import functools
import math
from os import PathLike
from typing import Iterable

import torch
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.pipelines import HUBERT_BASE

from .reader import read_alignments, read_manifest

SAMPLE_RATE = 16_000
ALIGNMENT_FREQ = 100  # in Hz

CONV_LAYER_CONFIG = HUBERT_BASE._params["extractor_conv_layer_config"]
MODEL_DOWNSAMPLING = functools.reduce(lambda x, y: x * y, [layer[2] for layer in CONV_LAYER_CONFIG])
MODEL_FREQ = SAMPLE_RATE // MODEL_DOWNSAMPLING  # in Hz
FEATURE_SIZE = 1 / MODEL_FREQ

SUBSAMPLE = ALIGNMENT_FREQ // MODEL_FREQ


class Tokenizer:
    # fmt:off
    PHONEMES = {
        "SIL": 0, "AA": 1, "AE": 2, "AH": 3, "AO": 4, "AW": 5, "AY": 6, "B": 7,
        "CH": 8, "D": 9, "DH": 10, "EH": 11, "ER": 12, "EY": 13, "F": 14, "G": 15,
        "HH": 16, "IH": 17, "IY": 18, "JH": 19, "K": 20, "L": 21, "M": 22, "N": 23,
        "NG": 24, "OW": 25, "OY": 26, "P": 27, "R": 28, "S": 29, "SH": 30, "T": 31,
        "TH": 32, "UH": 33, "UW": 34, "V": 35, "W": 36, "Y": 37, "Z": 38, "ZH": 39,
    }
    # fmt:on

    def __init__(self, with_blank: bool = False) -> None:
        self.token_to_id = self.PHONEMES | {"<pad>": self.pad_id}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.with_blank = with_blank

    @property
    def vocab_size(self) -> int:
        if self.with_blank:
            return len(self.PHONEMES) + 1
        return len(self.PHONEMES)

    @property
    def silence_id(self) -> int:
        return self.PHONEMES["SIL"]

    @property
    def pad_id(self) -> int:
        return len(self.PHONEMES)

    def encode(self, phones: list[str] | str) -> torch.LongTensor:
        if isinstance(phones, str):
            phones = phones.split(" ")
        return torch.LongTensor([self.token_to_id[phone] for phone in phones])

    def decode(self, tokens: Iterable[int]) -> str:
        return " ".join(self.id_to_token[int(token)] for token in tokens if token < self.pad_id)


def conv_output_length(input_length: int):
    """Compute the output length of a sequence after the series of 1D
    convolutions in HuBERT's feature extractor."""
    for _, kernel_size, stride in CONV_LAYER_CONFIG:
        input_length = math.floor((input_length - kernel_size) / stride + 1)
    return input_length


def subsample_tokens(tokens: Tensor, num_samples: int) -> Tensor:
    """Subsample the phone sequence to match the output length of HuBERT."""
    subsampled = tokens[::SUBSAMPLE]
    output_length = conv_output_length(num_samples)
    if len(subsampled) == output_length:
        return subsampled
    if len(subsampled) == output_length + 1:
        return subsampled[:-1]
    raise ValueError(f"Length mismatch: {len(tokens)} vs {output_length}")


class PhonemeDataset(Dataset):
    def __init__(
        self,
        manifest_path: PathLike,
        alignments_path: PathLike,
        tokenizer: Tokenizer,
        deduplicate: bool = False,
    ) -> None:
        """Phone dataset for finetuning HuBERT.

        Parameters
        ----------
        manifest_path : PathLike
            Path to the manifest file. First row is the root of the dataset.
            Subsequent rows are tab-separated with the fileid and the number of samples.
        alignments_path : PathLike
            Path to the alignments file.
            Each row is tab-separated with the fileid and the phone sequence (itself separated by spaces).
        tokenizer : Tokenizer
            The phone tokenizer.
        deduplicate : bool, optional
            Whether to deduplicate phones in the sequence, by default False.
        """
        super().__init__()
        self.manifest = read_manifest(manifest_path)
        self.indices = dict(enumerate(self.manifest))
        alignments = read_alignments(alignments_path)
        self.tokens = {fileid: tokenizer.encode(phones) for fileid, phones in alignments.items()}
        self.deduplicate = deduplicate

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx not in self.indices:
            raise IndexError(f"Index {idx} out of range")
        fileid = self.indices[idx]
        path, num_samples = self.manifest[fileid]
        audio, sr = torchaudio.load(str(path))
        assert audio.size(0) == 1
        audio = audio.squeeze()
        assert sr == SAMPLE_RATE and len(audio) == num_samples
        tokens = subsample_tokens(self.tokens[fileid], num_samples)
        if self.deduplicate:
            return audio, tokens.unique_consecutive()
        return audio, tokens

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Collate function to pad audio and token sequences to the same length."""
        audios, tokens = zip(*batch)
        padded_audios = pad_sequence(audios, batch_first=True)
        padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=Tokenizer().pad_id)
        length_audios = torch.LongTensor([len(audio) for audio in audios])
        length_tokens = torch.LongTensor([len(token) for token in tokens])
        return padded_audios, padded_tokens, length_audios, length_tokens
