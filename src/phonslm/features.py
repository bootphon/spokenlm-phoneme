import csv
from os import PathLike
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import SAMPLE_RATE, Tokenizer
from .model import HuBERTPhoneme
from .reader import read_manifest


class AudioDataset(Dataset):
    def __init__(self, manifest_path: PathLike) -> None:
        super().__init__()
        self.manifest = read_manifest(manifest_path)
        self.indices = dict(enumerate(self.manifest))

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Path]:
        if index not in self.indices:
            raise IndexError
        path, num_samples = self.manifest[self.indices[index]]
        waveform, sample_rate = torchaudio.load(str(path))
        assert sample_rate == SAMPLE_RATE, sample_rate
        assert waveform.ndim == 2, waveform.shape
        assert waveform.size(1) == num_samples, len(waveform)
        return waveform, path


def extract_features(checkpoint: Path, output: Path, manifests: list[Path], layers: list[int]) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = HuBERTPhoneme()
    model.load_state_dict(load_file(checkpoint))
    model.eval().to(device)
    datasets = {manifest.stem: AudioDataset(manifest) for manifest in set(manifests)}
    if len(layers) == 0:
        layers = list(range(1, 14))
    for layer in layers:
        for name in datasets.keys():
            (output / f"{layer}/{name}").mkdir(exist_ok=True, parents=True)
    with torch.inference_mode(), tqdm(total=sum(len(dataset) for dataset in datasets.values())) as pbar:
        for name, dataset in datasets.items():
            pbar.set_description(name)
            for waveform, path in dataset:
                layer_features, _ = model.extract_features(waveform.to(device))
                filename = path.with_suffix(".pt").name
                for layer in layers:
                    torch.save(layer_features[layer - 1].squeeze().cpu(), output / f"{layer}/{name}/{filename}")
                pbar.update()


def concatenate_features(features: Path, output: Path, manifests: list[Path]) -> None:
    for manifest in manifests:
        output.mkdir(exist_ok=True, parents=True)
        if (output / f"{manifest.stem}_0_1.len").is_file() and (output / f"{manifest.stem}_0_1.npy").is_file():
            print(f"Already concatenated {manifest.stem}")
            return
        with open(manifest, newline="") as tsvfile:
            lines = list(csv.reader(tsvfile, delimiter="\t"))[1:]
        paths = [features / manifest.stem / Path(wav).with_suffix(".pt") for wav, _ in lines]
        shapes = [torch.load(str(path), mmap=True).shape for path in paths]
        assert len(set(shape[1] for shape in shapes)) == 1
        assert all(len(shape) == 2 for shape in shapes)
        with open(output / f"{manifest.stem}_0_1.len", "w") as f:
            for shape in shapes:
                f.write(f"{shape[0]}\n")

        dtype = torch.load(paths[0]).numpy().dtype
        length = sum(shape[0] for shape in shapes)
        memory_map = np.memmap(output / f"{manifest.stem}.dat", dtype=dtype, mode="w+", shape=(length, shapes[0][1]))
        index = 0
        for path, shape in tqdm(list(zip(paths, shapes)), desc=f"Concatenating {manifest.stem}"):
            memory_map[index : index + shape[0]] = torch.load(path).numpy()
            index += shape[0]
        np.save(output / f"{manifest.stem}_0_1.npy", memory_map)
        (output / f"{manifest.stem}.dat").unlink()


def split_features(features: Path, output: Path, manifests: list[Path]) -> None:
    for manifest in manifests:
        out = output / manifest.stem
        out.mkdir(exist_ok=True, parents=True)
        with open(features / f"{manifest.stem}_0_1.len", "r") as file:
            lengths = [int(length) for length in file.read().splitlines()]
        with open(manifest, newline="") as f:
            names = [Path(wav).with_suffix(".pt") for wav, _ in list(csv.reader(f, delimiter="\t"))[1:]]
        assert len(lengths) == len(names), f"Invalid for {manifest.stem}"
        if all((out / name).is_file() for name in names):
            print("Already splitted")
            return

        concatenated = np.load(features / f"{manifest.stem}_0_1.npy")
        index = 0
        for length, name in tqdm(list(zip(lengths, names)), desc=f"Splitting {manifest.stem}"):
            tensor = torch.from_numpy(concatenated[index : index + length]).clone()
            assert tensor.ndim == 2, tensor.shape
            (out / name).parent.mkdir(exist_ok=True)
            torch.save(tensor, out / name)
            index += length


def onehot_features(labels: Path, output: Path, manifest: Path, num_classes: int = 500, sep: str = " ") -> None:
    with open(manifest, newline="") as f:
        names = [Path(wav).with_suffix(".pt") for wav, _ in list(csv.reader(f, delimiter="\t"))[1:]]
    with open(labels, "r") as f:
        lines = f.read().splitlines()
    assert len(names) == len(lines)
    output.mkdir(parents=True, exist_ok=True)
    for name, line in tqdm(list(zip(names, lines))):
        tensor = torch.LongTensor([int(x) for x in line.split(sep)])
        onehot_tensor = F.one_hot(tensor, num_classes=num_classes)
        torch.save(onehot_tensor, output / name)


def centroid_features(labels: Path, kmeans: Path, output: Path, manifest: Path, sep: str = " ") -> None:
    with open(manifest, newline="") as f:
        names = [Path(wav).with_suffix(".pt") for wav, _ in list(csv.reader(f, delimiter="\t"))[1:]]
    with open(labels, "r") as f:
        lines = f.read().splitlines()
    assert len(names) == len(lines)
    output.mkdir(parents=True, exist_ok=True)
    kmeans_model = joblib.load(kmeans)
    for name, line in tqdm(list(zip(names, lines))):
        centroid = torch.from_numpy(kmeans_model.cluster_centers_[[int(x) for x in line.split(sep)]])
        torch.save(centroid, output / name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature utilities.")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    pextract = subparsers.add_parser("extract", help="Extract features to OUTPUT with a trained model.")
    pconcat = subparsers.add_parser("concatenate", help="Concatenate features to OUTPUT.")
    psplit = subparsers.add_parser("split", help="Split features to OUTPUT.")
    ponehot = subparsers.add_parser("onehot", help="One-hot encode labels to OUTPUT.")
    pcentroid = subparsers.add_parser("centroid", help="Centroid encode labels to OUTPUT.")

    pextract.add_argument("checkpoint", type=Path, help="Path to the checkpoint.")
    pextract.add_argument("output", type=Path, help="Output directory.")
    pextract.add_argument("manifests", nargs="+", help="Manifest file(s) containing audio paths and samples.")
    pextract.add_argument("--layers", type=int, nargs="*", default=[], help="Layers to extract.")

    pconcat.add_argument("features", type=Path, help="Features directory.")
    pconcat.add_argument("output", type=Path, help="Output directory.")
    pconcat.add_argument("manifests", nargs="+", help="Manifest file(s) containing audio paths and samples.")

    psplit.add_argument("features", type=Path, help="Features directory.")
    psplit.add_argument("output", type=Path, help="Output directory.")
    psplit.add_argument("manifests", nargs="+", help="Manifest file(s) containing audio paths and samples.")

    ponehot.add_argument("labels", type=Path, help="Labels file.")
    ponehot.add_argument("output", type=Path, help="Output directory.")
    ponehot.add_argument("manifest", type=Path, help="Manifest file.")
    ponehot.add_argument("-k", "--num_classes", type=int, default=500, help="Number of classes.")
    ponehot.add_argument("--sep", type=str, default=" ", help="Separator.")

    pcentroid.add_argument("labels", type=Path, help="Labels file.")
    pcentroid.add_argument("kmeans", type=Path, help="KMeans model file.")
    pcentroid.add_argument("output", type=Path, help="Output directory.")
    pcentroid.add_argument("manifest", type=Path, help="Manifest file.")
    pcentroid.add_argument("--sep", type=str, default=" ", help="Separator.")

    args = parser.parse_args()
    if args.subcommand == "extract":
        extract_features(args.checkpoint, args.output, args.manifests, args.layers)
    elif args.subcommand == "concatenate":
        concatenate_features(args.features, args.output, args.manifests)
    elif args.subcommand == "split":
        split_features(args.features, args.output, args.manifests)
    elif args.subcommand == "onehot":
        onehot_features(args.labels, args.output, args.manifest, args.num_classes, args.sep)
    elif args.subcommand == "centroid":
        centroid_features(args.labels, args.kmeans, args.output, args.manifest, args.sep)
    else:
        parser.print_help()
