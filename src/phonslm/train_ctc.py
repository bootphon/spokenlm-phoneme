import argparse
import dataclasses
import datetime
import time
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PhonemeDataset, Tokenizer
from .external.sampler import LengthGroupedSampler
from .external.scheduler import TriStageLRScheduler
from .model import CTCLossOnLogits, HuBERTPhoneme
from .utils import Records, set_seed


@dataclasses.dataclass
class Configuration:
    workdir: str
    project_name: str
    train_manifest: str
    val_manifest: str
    train_alignment: str
    val_alignment: str
    resume_from_checkpoint: str | None = None
    wandb_mode: Literal["offline", "online", "disabled"] = "offline"
    logging_steps: int = 100

    seed: int = 0
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] | None = "fp16"
    num_workers: int = 4
    pin_memory: bool = True
    group_by_length: bool = True

    max_steps: int = 20_000
    warmup_ratio: float = 0.01
    decay_ratio: float = 0.5
    freeze_encoder_ratio: float = 0.1

    train_batch_size: int = 32
    adam_eps: float = 1e-8
    adam_betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    max_lr: float = 5e-5
    max_grad_norm: float = 5.0

    eval_batch_size: int = 32
    eval_epochs: int = 1

    def __post_init__(self):
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
        assert self.mixed_precision in [None, "no", "fp16", "bf16", "fp8"]
        assert self.wandb_mode in ["offline", "online", "disabled"]

    @property
    def logging_dir(self) -> Path:
        return Path(self.workdir).resolve()  # "wandb" is added by wandb.init

    @property
    def project_dir(self) -> Path:
        return Path(self.workdir).resolve() / self.project_name / self.start_time

    def to_wandb_config(self) -> dict[str, Any]:
        exclude_keys = [
            "wandb_mode",
            "logging_steps",
            "workdir",
            "project_name",
            "train_manifest",
            "val_manifest",
            "train_alignment",
            "val_alignment",
            "resume_from_checkpoint",
        ]
        return {k: v for k, v in dataclasses.asdict(self).items() if k not in exclude_keys}


@torch.inference_mode()
def evaluation(
    model: HuBERTPhoneme,
    criterion: nn.Module,
    dataloader: DataLoader,
    is_main_process: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Validation", leave=False, disable=not is_main_process):
        inputs, targets, input_lengths, target_lengths = batch
        logits, logit_lengths = model.inference(inputs, input_lengths)
        loss = criterion(logits, logit_lengths, targets, target_lengths)
        total_loss += loss.float()
    model.train()
    return total_loss.float().item() / len(dataloader)


def training(cfg: Configuration) -> None:
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.project_dir / "config.yaml", "w") as f:
        yaml.dump(dataclasses.asdict(cfg), f)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, project_dir=cfg.project_dir, log_with="wandb")
    accelerator.print(dataclasses.asdict(cfg))
    accelerator.init_trackers(
        cfg.project_name,
        config=cfg.to_wandb_config() | {"num_processes": accelerator.num_processes},
        init_kwargs={"wandb": {"mode": cfg.wandb_mode, "name": cfg.start_time, "dir": cfg.logging_dir}},
    )
    set_seed(cfg.seed)

    tokenizer = Tokenizer(with_blank=True)
    model = HuBERTPhoneme(tokenizer.vocab_size)
    train_dataset = PhonemeDataset(cfg.train_manifest, cfg.train_alignment, tokenizer, deduplicate=True)
    if cfg.group_by_length:
        num_samples = [train_dataset.manifest[train_dataset.indices[idx]][1] for idx in range(len(train_dataset))]
        sampler = LengthGroupedSampler(cfg.train_batch_size, num_samples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        collate_fn=PhonemeDataset.collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        shuffle=not cfg.group_by_length,
        sampler=sampler if cfg.group_by_length else None,
    )
    eval_dataloader = DataLoader(
        PhonemeDataset(cfg.val_manifest, cfg.val_alignment, tokenizer, deduplicate=True),
        batch_size=cfg.eval_batch_size,
        collate_fn=PhonemeDataset.collate_fn,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    warmup_steps, decay_steps = int(cfg.warmup_ratio * cfg.max_steps), int(cfg.decay_ratio * cfg.max_steps)
    freeze_encoder_steps = int(cfg.freeze_encoder_ratio * cfg.max_steps)
    hold_steps = cfg.max_steps - warmup_steps - decay_steps
    lr_scheduler = TriStageLRScheduler(optimizer, warmup_steps, hold_steps, decay_steps)
    criterion = CTCLossOnLogits(tokenizer.pad_id)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if cfg.resume_from_checkpoint is not None:
        ckpt = Path(cfg.resume_from_checkpoint).resolve()
        if not ckpt.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")
        if not ckpt.name.startswith("step_"):
            raise ValueError(f"Invalid checkpoint file: {ckpt.name}. Must start with 'step_'")
        accelerator.print(f"Resuming from {cfg.resume_from_checkpoint}")
        accelerator.load_state(ckpt)
        step = int(ckpt.stem.removeprefix("step_"))
        starting_epoch = step // len(train_dataloader)
        to_skip = step - starting_epoch * len(train_dataloader)
        active_dataloader = accelerator.skip_first_batches(train_dataloader, to_skip)
        if step >= freeze_encoder_steps:
            accelerator.print("Unfreezing the encoder.")
            model.freeze_encoder = False
    else:
        step, starting_epoch = 0, 0

    records = Records(["train/loss", "train/batch_time", "train/data_time", "train/grad_norm"])
    pbar = tqdm(total=cfg.max_steps, desc="Train", initial=step, disable=not accelerator.is_main_process)
    active_dataloader = train_dataloader
    for epoch in range(starting_epoch, cfg.max_steps // len(train_dataloader) + 1):
        model.train()
        tick = time.perf_counter()
        for inputs, targets, input_lengths, target_lengths in active_dataloader:
            records["train/data_time"].update(time.perf_counter() - tick)
            logits, logit_lengths = model(inputs, input_lengths)
            loss = criterion(logits, logit_lengths, targets, target_lengths)
            records["train/loss"].update(loss.detach().float().item(), inputs.size(0))
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                total_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                records["train/grad_norm"].update(total_norm, inputs.size(0))
            lr = lr_scheduler.get_lr()[0]
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            records["train/batch_time"].update(time.perf_counter() - tick)
            step += 1
            pbar.update()
            pbar.set_postfix(loss=records["train/loss"].avg)
            if step % cfg.logging_steps == 0:
                accelerator.log(records.log() | {"train/epoch": epoch, "train/lr": lr}, step)
            if step == freeze_encoder_steps:
                accelerator.print(f"Unfreezing the encoder at step {step}.")
                accelerator.log({"event": "Unfreeze encoder"}, step)
                model.freeze_encoder = False
                model.train()
            if step >= cfg.max_steps:
                accelerator.save_state(cfg.project_dir / f"step_{step}")
                val_loss = evaluation(model, criterion, eval_dataloader, accelerator.is_main_process)
                accelerator.log({"val/loss": val_loss}, step)
                accelerator.end_training()
                return
            tick = time.perf_counter()
        if epoch % cfg.eval_epochs == 0:
            accelerator.save_state(cfg.project_dir / f"step_{step}")
            val_loss = evaluation(model, criterion, eval_dataloader, accelerator.is_main_process)
            accelerator.log({"val/loss": val_loss}, step)
        active_dataloader = train_dataloader
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuBERT finetuning with CTC loss.")
    parser.add_argument("config", type=str, help="Path to the full YAML configuration file.")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    training(Configuration(**config))
