import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from .data import PhonemeDataset, Tokenizer
from .model import HuBERTPhoneme
from .utils import word_error_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate finetuned Hubert model: PER and frame-level accuracy.")
    parser.add_argument("manifest", type=str, help="Path to the manifest file.")
    parser.add_argument("alignments", type=str, help="Path to the alignments file.")
    parser.add_argument("model", type=str, help="Path to the model file.")
    parser.add_argument("--revision", type=str, help="Revision passed to huggingface hub")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer(with_blank=True)
    model = HuBERTPhoneme.from_pretrained(args.model, revision=args.revision).to(device).eval()
    dataset = PhonemeDataset(args.manifest, args.alignments, tokenizer, deduplicate=model.ctc_training)

    hypotheses, references = [], []
    y_true, y_pred = [], []
    with torch.inference_mode():
        for audio, target in tqdm(dataset):
            output, _ = model.inference(audio.to(device).unsqueeze(0))
            predictions = output.argmax(dim=-1).squeeze().cpu()
            references.append(tokenizer.decode(target.unique_consecutive()))
            hypotheses.append(tokenizer.decode(predictions.unique_consecutive()))
            if not model.ctc_training:
                y_true += target.flatten().tolist()
                y_pred += predictions.flatten().tolist()
    print(f"PER: {word_error_rate(hypotheses, references):.2%}")
    if not model.ctc_training:
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
        print(classification_report(y_true, y_pred, target_names=tokenizer.PHONEMES))
