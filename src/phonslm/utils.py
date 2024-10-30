import random
from typing import Literal

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from torch import distributed
from torchaudio.functional import edit_distance

_KMEANS_CHECKPOINTS = {
    "standard": ("coml/hubert-phoneme-classification", "km500-base-l11.joblib", "kmeans"),
    "finetuned": ("coml/hubert-phoneme-classification", "km500-ft100h-l12.joblib", "kmeans"),
}


def load_kmeans(name: Literal["standard", "finetuned"]) -> KMeans:
    if name not in _KMEANS_CHECKPOINTS:
        raise ValueError(f"Unknown kmeans checkpoint: {name}. Should be either 'standard' or 'finetuned'.")
    repo_id, filename, revision = _KMEANS_CHECKPOINTS[name]
    path = hf_hub_download(repo_id, filename, revision=revision)
    return joblib.load(path)


def create_mask_from_lengths(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Create a mask from a tensor of lengths."""
    max_length = max_length or lengths.max().item()
    return torch.arange(max_length, device=lengths.device).expand(len(lengths), max_length) < lengths.unsqueeze(1)


def word_error_rate(hypotheses: list[str], targets: list[str], sep: str = " ") -> float:
    """Compute the WER or PER between the predictions and the targets.
    Predictions and targets are list of strings where words / phonemes are separated by 'sep'.
    Normalize at the end."""
    assert isinstance(hypotheses, list) and isinstance(targets, list)
    total_edit_distance, total_length = 0, 0
    for hypothesis, target in zip(hypotheses, targets):
        assert isinstance(hypothesis, str) and isinstance(target, str)
        total_edit_distance += edit_distance(hypothesis.split(sep), target.split(sep))
        total_length += len(target.split(sep))
    return total_edit_distance / total_length


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        distributed.all_reduce(total, distributed.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def log(self):
        avg = self.avg
        self.reset()
        return avg


class Records:
    """Record average meters for multiple values."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.meters = {name: AverageMeter() for name in names}

    def __getitem__(self, name: str) -> AverageMeter:
        return self.meters[name]

    def log(self) -> dict[str, float]:
        return {name: meter.log() for name, meter in self.meters.items()}
