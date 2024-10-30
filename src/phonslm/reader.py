import csv
from os import PathLike
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def write_manifest(dataset: PathLike, output: PathLike, file_extension: str = ".wav") -> None:
    """Write a manifest file containing the file paths and their number of samples.
    First line is the root directory of the dataset.
    Each line contains the relative path of the file and its number of samples."""
    lines = [str(Path(dataset).resolve())]
    paths = list(Path(dataset).rglob(f"*{file_extension}"))
    for name in tqdm(paths):
        lines.append(f"{name.relative_to(dataset)}\t{sf.info(name).frames}")
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")


def read_manifest(file_path: PathLike) -> dict[str, tuple[Path, int]]:
    """Read a manifest file containing the file paths and their number of samples.
    Returns a dictionary with the file id as the key and a tuple of the file path
    and its number of samples as the value."""
    manifest = {}
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        root = Path(next(reader)[0])
        for row in reader:
            assert len(row) == 2, f"Invalid tsv file: {file_path}"
            file, num_samples = root / row[0], int(row[1])
            assert file.stem not in manifest, f"Duplicate file id: {file.stem}"
            manifest[file.stem] = (file, num_samples)
    return manifest


def read_units(units_path: PathLike, manifest_path: PathLike, sep: str = " ") -> dict[str, list[int]]:
    """Read discrete units and return a dictionary with the file id as the key and
    a list of units as the value.

    Parameters
    ----------
    units_path : PathLike
        Path to the file containing the discrete units.
        Each row contains the units for a file, separated by `sep`.

    manifest_path : PathLike
        Path to the manifest file containing the file paths and their number of samples.

    sep : str, optional
        Separator between units, by default " "

    Returns
    -------
    dict[str, list[int]]
        Dictionary with the file id as the key and a list of units as the value.
    """
    with open(units_path, "r") as f:
        lines = f.read().splitlines()
    manifest = read_manifest(manifest_path)
    assert len(lines) == len(manifest)
    units = {fileid: [int(unit) for unit in lines[i].split(sep)] for i, fileid in enumerate(manifest)}
    assert len(units) == len(lines)
    return units


def read_alignments(alignments_path: PathLike, sep: str = " ") -> dict[str, list[str]]:
    """Read alignments and return a dictionary with the file id as the key and
    a list of phones as the value."""
    phones = {}
    with open(alignments_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            assert len(row) == 2
            phones[row[0]] = row[1].split(sep)
    return phones
