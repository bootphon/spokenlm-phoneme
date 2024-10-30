import argparse
import os
from pathlib import Path

try:
    zerospeech = Path(os.environ["APP_DIR"]).resolve() / "datasets"
    librispeech = Path(os.environ["DSDIR"]).resolve() / "LibriSpeech"
except KeyError:
    print("Please set the environment variables APP_DIR and DSDIR")
    exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change the headers of the manifest files.")
    parser.add_argument("manifest_dir")
    args = parser.parse_args()

    expresso = None if "EXPRESSO" not in os.environ else Path(os.environ["EXPRESSO"]).resolve()

    for manifest in Path(args.manifest_dir).glob("*.tsv"):
        with manifest.open() as f:
            lines = f.readlines()
        assert len(lines[0].split(" ")) == 1, "The first line should contain only the root path."
        root = Path(lines[0].strip())

        if root.is_dir():  # Skip if the root is a directory
            continue
        if manifest.stem.startswith("zr2021-"):
            root = zerospeech / "abxLS-dataset" / manifest.stem.removeprefix("zr2021-")
        elif manifest.stem.startswith("swuggy-"):
            root = zerospeech / "sLM21-dataset/lexical" / manifest.stem.removeprefix("swuggy-")
        elif manifest.stem.startswith("sblimp-"):
            root = zerospeech / "sLM21-dataset/syntactic" / manifest.stem.removeprefix("sblimp-")
        elif manifest.stem in [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]:
            root = librispeech / manifest.stem
        elif manifest.stem.startswith("expresso-"):
            if expresso is not None:
                root = expresso / manifest.stem.removeprefix("expresso-").removeprefix("read-")
            else:
                print("EXPRESSO environment variable not set. Skipping.")
                continue
        else:
            print(f"Unknown manifest {manifest.stem}. Skipping.")
            continue

        assert root.is_dir(), f"Directory {root} does not exist."
        with manifest.open("w") as f:
            f.write(f"{root}\n")
            f.writelines(lines[1:])
