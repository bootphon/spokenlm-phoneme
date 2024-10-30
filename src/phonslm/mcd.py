import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def compute_mcd(manifest_original: Path, manifest_generated: Path, output: Path) -> None:
    cmds = []
    with open(manifest_original, "r") as f_orig, open(manifest_generated, "r") as f_gen:
        original = f_orig.read().splitlines()
        generated = f_gen.read().splitlines()
        orig_root, gen_root = Path(original[0]), Path(generated[0])
        assert len(original) == len(generated)
        for orig, gen in zip(original[1:], generated[1:]):
            orig_path = orig_root / orig.split("\t")[0]
            gen_path = gen_root / gen.split("\t")[0]
            cmds.append((f"mcd-cli from-wav {orig_path} {gen_path}", orig_path, gen_path))

    scores = []
    for cmd, orig_path, gen_path in tqdm(cmds):
        out = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        score = float(out.stdout.splitlines()[0].removeprefix("Mel-Cepstral Distance: "))
        scores.append((score, orig_path, gen_path))
    pd.DataFrame(scores, columns=["score", "orig_path", "gen_path"]).to_csv(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Mel-Cepstral Distances.")
    parser.add_argument("manifest_original", type=Path, help="Original manifest file.")
    parser.add_argument("manifest_generated", type=Path, help="Generated manifest file.")
    parser.add_argument("output", type=Path, help="Output CSV file.")
    args = parser.parse_args()
    compute_mcd(args.manifest_original, args.manifest_generated, args.output)
