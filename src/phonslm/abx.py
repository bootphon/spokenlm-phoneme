import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

from .data import FEATURE_SIZE

COMMAND = (
    "zrc-abx2 {features} {item} --out {out} --file_extension {file_extension} "
    + " --speaker_mode {speaker_mode} --context_mode {context_mode} "
    + f"--feature_size {FEATURE_SIZE}"
)


@dataclass
class ABXDataset:
    name: str
    features_name: str
    item: Path
    audio: Path
    context_mode: Literal["within", "any", "all"]

    def __post_init__(self):
        self.item = self.item.resolve()
        self.audio = self.audio.resolve()
        assert self.item.is_file(), self.item
        assert self.audio.is_dir(), self.audio
        assert self.context_mode in ["within", "any", "all"], self.context_mode


def zerospeech_2017_datasets() -> list[ABXDataset]:
    try:
        root = Path(os.environ["APP_DIR"]) / "datasets" / "zrc2017-test-dataset"
    except KeyError as error:
        raise KeyError("Please set APP_DIR as used by zerospeech-benchmarks") from error
    return [
        ABXDataset(
            name=f"zr2017-{lang}-{duration}",
            features_name=f"zr2017-{lang}-{duration}",
            item=root / lang / duration / f"{duration}.item",
            audio=root / lang / duration,
            context_mode="within",
        )
        for lang, duration in product(["english", "french", "german", "mandarin", "wolof"], ["1s", "10s", "120s"])
    ]


def librispeech_abx_datasets() -> list[ABXDataset]:
    try:
        root = Path(os.environ["APP_DIR"]) / "datasets" / "abxLS-dataset"
    except KeyError as error:
        raise KeyError("Please set APP_DIR as used by zerospeech-benchmarks") from error
    return [
        ABXDataset(
            name=f"zr2021-{subset}-{mode}",
            features_name=f"zr2021-{subset}",
            item=root / subset / f"{mode}-{subset}.item",
            audio=root / subset,
            context_mode="all" if mode == "phoneme" else "within",
        )
        for subset, mode in product(["dev-clean", "dev-other"], ["phoneme", "triphone"])
    ]


def abx_cmds(root: Path, output: Path, subset: Literal["zrc2017", "abxLS"] = "abxLS", extension: str = ".pt") -> None:
    for features_dir in root.resolve().glob("*"):
        for dataset in librispeech_abx_datasets() if subset == "abxLS" else zerospeech_2017_datasets():
            out = output / features_dir.name / dataset.name
            out.mkdir(exist_ok=True, parents=True)
            for speaker_mode in ["within", "across"]:
                if not (features_dir / dataset.features_name).is_dir():
                    continue
                if (out / speaker_mode / "ABX_scores.csv").is_file():
                    if "triphone" in dataset.name:
                        continue
                    with open(out / speaker_mode / "ABX_scores.csv", "r") as f:
                        lines = f.readlines()
                        if len(lines) == 3:
                            continue
                cmd = COMMAND.format(
                    features=features_dir / dataset.features_name,
                    item=dataset.item,
                    file_extension=extension,
                    context_mode=dataset.context_mode,
                    speaker_mode=speaker_mode,
                    out=out / speaker_mode,
                )
                print(cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print ABX commands.")
    parser.add_argument("root", type=Path, help="Root directory containing features.")
    parser.add_argument("output", type=Path, help="Output directory.")
    parser.add_argument("--subset", type=str, default="abxLS", choices=["abxLS", "zrc2017"])
    parser.add_argument("--extension", type=str, default=".pt")
    args = parser.parse_args()
    abx_cmds(args.root, args.output, args.subset, args.extension)
