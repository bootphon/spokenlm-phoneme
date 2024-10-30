import argparse
from pathlib import Path

import joblib
import torch
import torchaudio
from safetensors.torch import load_file
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .data import Tokenizer
from .model import HuBERTPhoneme

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Expresso code dataset")
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--kmeans", help="Path to kmeans. Required if layer != 13")
    parser.add_argument("--layer", required=True, type=int, help="Layer between 1 and 13")
    parser.add_argument("--checkpoint", type=str, help="If not specified, use default pretrained HuBERT")
    parser.add_argument("--file-extension", default=".wav", help="Audio file extension")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer()
    if args.kmeans is not None:
        kmeans: MiniBatchKMeans = joblib.load(args.kmeans)
    elif args.layer != 13:
        raise ValueError("Must set kmeans if layer != 13")
    else:
        kmeans = None

    model = HuBERTPhoneme(tokenizer.vocab_size)
    if args.checkpoint is not None:
        model.load_state_dict(load_file(args.checkpoint))
    model.eval().to(device)

    output = []
    with torch.inference_mode():
        for path in tqdm(list(Path(args.dataset).resolve().rglob(f"*{args.file_extension}"))):
            spk = path.parent.parent.name
            wav = torchaudio.load(path)[0].to(device)
            features = model.extract_features(wav)[0][args.layer - 1]
            if kmeans is not None:
                code = kmeans.predict(features.cpu().squeeze().numpy())
            else:
                code = features.cpu().squeeze().numpy().argmax(-1)
            output.append({"audio": str(path), "hubert": " ".join(map(str, code)), "spk": spk})

    with open(args.output, "w") as f:
        f.write("\n".join(map(str, output)))
