# Improving Spoken Language Modeling with Phoneme Classification: A Simple Fine-tuning Approach

Companion repository to the paper ["Improving Spoken Language Modeling with Phoneme Classification: A Simple Fine-tuning Approach"](https://arxiv.org/abs/2410.00025) accepted to EMNLP 2024.

## Installation

Clone this repository:

```bash
git clone https://github.com/bootphon/spokenlm-phoneme.git
cd spokenlm-phoneme
```

You can then install this package and work from here.

If you want to be sure that the dependencies are well resolved, you can use `uv` and just run `uv sync`.
This will create a virtual environment in `.venv` and install this package.

Install the following tool to be able to compute the Mel-Cepstral distortion:

```bash
uv tool install mel-cepstral-distance -p 3.9
```

## Artifacts

The fine-tuned HuBERT are here: https://huggingface.co/coml/hubert-phoneme-classification:
- To use the model fine-tuned on LibriSpeech train-clean-100:
  ```python
  from phonslm import HuBERTPhoneme

  model = HuBERTPhoneme.from_pretrained("coml/hubert-phoneme-classification")
  ```
- If you want to use the models fine-tuned on LibriLight limited, specify the duration to the `revision` identifier (10min, 1h or 10h):
  ```python
  from phonslm import HuBERTPhoneme

  model = HuBERTPhoneme.from_pretrained("coml/hubert-phoneme-classification", revision="10min")
  ```
- Similarly for the models fine-tuned with the CTC loss:
  ```python
  from phonslm import HuBERTPhoneme

  model = HuBERTPhoneme.from_pretrained("coml/hubert-phoneme-classification", revision="ctc-100h")
  ```

The kmeans trained from L11 of standard HuBERT or L12 of finetuned HuBERT are easily accessible like this:
```python
from phonslm import load_kmeans

kmeans_std = load_kmeans("standard")
kmeans_ft = load_kmeans("finetuned")
```

The language models trained from the discrete units are here https://huggingface.co/coml/hubert-phoneme-classification/tree/language-models

If you want to download the ABX datasets for evaluation, set `APP_DIR` to where you want to download the datasets and run:

```bash
zrc datasets:pull abxLS-dataset
```

The manifest files and alignments are in `assets/data.tar.gz`. Change the headers of the manifest files to the root of your LibriSpeech and ABX datasets.

## Usage

### Finetuning HuBERT for phone classification

- Create a `config.yaml` containing at least:
  - `workdir`: where the checkpoints of the runs will be saved
  - `project_name`: wandb project name
  - `train_manifest`, `val_manifest`, `train_alignment`, `val_alignment`: paths to the manifest and alignment files for training and validation.
- Launch training with:
  ```bash
  accelerate launch -m phonslm.train config.yaml
  ```

For CTC training, same thing with `accelerate launch -m phonslm.train_ctc config.yaml`.

### Evaluation

#### Accuracy and PER

To get frame-level accuracy and PER with greedy decoding, run:
```bash
python -m phonslm.evaluate $MANIFEST_PATH $ALIGNMENTS_PATH $CHECKPOINT
```
where `CHECKPOINT` is the path / the name of the model, and `MANIFEST_PATH` and `ALIGNMENTS_PATH` are the paths to the manifest and alignment files.

#### ABX

First, extract features with:
```bash
python -m phonslm.features extract $CHECKPOINT $FEATURES -m $MANIFEST_1 -m $MANIFEST_2 -m $MANIFEST_3
```
where `CHECKPOINT` is the path to the safetensors checkpoint of the model, `FEATURES` is the path to the output features directory, and `MANIFEST_1`, `MANIFEST_2`, `MANIFEST_3` are the paths to the manifest files.
You can also specify the intermediate layers with the `-l` option.

Then, print the ABX evaluation commands with:
```bash
python -m phonslm.abx $FEATURES $OUTPUT > abx_commands.sh
```

Finally, run the ABX evaluation in parallel on a SLURM cluster with:
```bash
./scripts/launch_parallel.sh abx_commands.sh
```

### Clustering

Currently, to perform clustering, first extract features similary to the ABX evaluation, then concatenate them into a single file with:
```bash
python -m phonslm.features concatenate $FEATURES/$LAYER $CONCATENATED/$LAYER -m $MANIFEST_1 -m $MANIFEST_2 -m $MANIFEST_3
```
where `LAYER` is the layer from where features were extracted, and `CONCATENATED` is the output directory.

Then, run the clustering using [fairseq kmeans script in the HuBERT example directory](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans#k-means-clustering).
Dump the labels with [their script](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans#k-means-application).

### Language modeling from discrete units

Preprocess the data for fairseq:
```bash
fairseq-preprocess --only-source \
    --trainpref $1/train.km \
    --validpref $1/valid.km \
    --testpref $1/test.km \
    --destdir $1/fairseq_bin_data \
    --workers 10

cp $1/fairseq_bin_data/dict.txt $1
```

The LSTM is trained with fairseq, with the same config as the Zerospeech baseline system:
```bash
fairseq-train --fp16 $1/fairseq_bin \
    --task language_modeling \
    --save-dir $1 \
    --keep-last-epochs 2 \
    --tensorboard-logdir $1/tensorboard \
    --arch lstm_lm \
    --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
    --decoder-out-embed-dim 200 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
    --dropout 0.1 --weight-decay 0.01 \
    --sample-break-mode none --tokens-per-sample 2048\
    --max-tokens 163840 --update-freq 1 --max-update 100000
```

To perform one-hot encoding, or to compute probabilities from the LSTM, use the scripts in the [Zerospeech baselines repository](https://github.com/zerospeech/zerospeech2021_baseline).

## Citation

```bibtex
@inproceedings{poli-2024-improving,
    title = "Improving Spoken Language Modeling with Phoneme Classification: A Simple Fine-tuning Approach",
    author = "Poli, Maxime and Chemla, Emmanuel and Dupoux, Emmanuel",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
}
```
