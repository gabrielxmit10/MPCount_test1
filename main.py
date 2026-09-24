"""Config-driven MPCount training, validation, testing, and visualization."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.den_cls_dataset import DenClsDataset
from datasets.den_dataset import DensityMapDataset
from datasets.mdc_dataset import MDCDenClsDataset
from models.models import (
    DGModel_base,
    DGModel_cls,
    DGModel_final,
    DGModel_mem,
    DGModel_memadd,
    DGModel_memcls,
)
from trainers.dgtrainer import DGTrainer
from utils.misc import get_seeded_generator, seed_everything, seed_worker


def get_model(name, params):
    models = {
        "base": DGModel_base,
        "mem": DGModel_mem,
        "memadd": DGModel_memadd,
        "cls": DGModel_cls,
        "memcls": DGModel_memcls,
        "final": DGModel_final,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name](**params)


def get_loss():
    return nn.MSELoss()


def get_dataset(name, params, method):
    datasets = {
        "den": DensityMapDataset,
        "den_cls": DenClsDataset,
        "mdc_den_cls": MDCDenClsDataset,
    }
    if name == "jhu_domain":
        from datasets.jhu_domain_dataset import JHUDomainDataset

        datasets[name] = JHUDomainDataset
    elif name == "jhu_domain_cls":
        from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset

        datasets[name] = JHUDomainClsDataset
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")
    dataset_type = datasets[name]
    dataset = dataset_type(method=method, **params)
    return dataset, dataset_type.collate


def get_optimizer(name, params, model):
    optimizers = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizers[name](model.parameters(), **params)


def get_scheduler(name, params, optimizer):
    schedulers = {
        "step": torch.optim.lr_scheduler.StepLR,
        "multistep": torch.optim.lr_scheduler.MultiStepLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    }
    if name in {None, "none"}:
        return None
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")
    return schedulers[name](optimizer, **params)


def _expand_environment(value):
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _apply_overrides(cfg, args):
    cfg = copy.deepcopy(cfg)
    if args.data_root:
        for section in ("train_dataset", "val_dataset", "test_dataset"):
            if section in cfg:
                cfg[section]["params"]["root"] = args.data_root
    if args.checkpoint is not None:
        cfg["checkpoint"] = None if args.checkpoint.lower() == "none" else args.checkpoint
    if args.run_name:
        cfg["version"] = args.run_name
    if args.device:
        cfg["device"] = args.device
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.persistent_dir:
        cfg["persistent_dir"] = args.persistent_dir
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs
    if args.batch_size is not None and "train_loader" in cfg:
        cfg["train_loader"]["batch_size"] = args.batch_size
    if args.crop_size is not None:
        for section in ("train_dataset", "val_dataset", "test_dataset"):
            if section in cfg:
                cfg[section]["params"]["crop_size"] = args.crop_size
    if args.learning_rate is not None and "optimizer" in cfg:
        cfg["optimizer"]["params"]["lr"] = args.learning_rate
        if cfg.get("scheduler", {}).get("name") == "onecycle":
            cfg["scheduler"]["params"]["max_lr"] = args.learning_rate
    if args.num_workers is not None:
        for section in ("train_loader", "val_loader", "test_loader"):
            if section in cfg:
                cfg[section]["num_workers"] = args.num_workers
                if args.num_workers == 0:
                    cfg[section]["persistent_workers"] = False
    if args.patch_size is not None:
        cfg["patch_size"] = args.patch_size
    if getattr(args, "pretrained", None) is not None:
        cfg["model"]["params"]["pretrained"] = args.pretrained
    if getattr(args, "deterministic", None) is not None:
        cfg["model"]["params"]["deterministic"] = args.deterministic
    if args.eval_split and "test_dataset" in cfg:
        cfg["test_dataset"]["params"]["split_file"] = args.eval_split
    return _expand_environment(cfg)


def load_config(config_path, task, args):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = _apply_overrides(cfg, args)

    init_params = {
        "seed": cfg["seed"],
        "version": cfg["version"],
        "device": cfg["device"],
        "log_para": cfg["log_para"],
        "patch_size": cfg["patch_size"],
        "mode": cfg["mode"],
        "output_dir": cfg.get("output_dir", "logs"),
        "persistent_dir": cfg.get("persistent_dir"),
    }
    task_params = {
        "model": get_model(cfg["model"]["name"], cfg["model"]["params"]),
        "checkpoint": cfg.get("checkpoint"),
    }

    seed_everything(cfg["seed"])
    generator = get_seeded_generator(cfg["seed"])

    if task in {"train", "train_test"}:
        task_params["loss"] = get_loss()
        train_dataset, collate = get_dataset(
            cfg["train_dataset"]["name"], cfg["train_dataset"]["params"], method="train"
        )
        task_params["train_dataloader"] = DataLoader(
            train_dataset,
            collate_fn=collate,
            worker_init_fn=seed_worker,
            generator=generator,
            **cfg["train_loader"],
        )
        val_dataset, _ = get_dataset(
            cfg["val_dataset"]["name"], cfg["val_dataset"]["params"], method="val"
        )
        task_params["val_dataloader"] = DataLoader(val_dataset, **cfg["val_loader"])
        task_params["optimizer"] = get_optimizer(
            cfg["optimizer"]["name"], cfg["optimizer"]["params"], task_params["model"]
        )
        scheduler_params = copy.deepcopy(cfg["scheduler"].get("params", {}))
        if cfg["scheduler"]["name"] == "onecycle":
            scheduler_params["epochs"] = cfg["num_epochs"]
            scheduler_params["steps_per_epoch"] = len(task_params["train_dataloader"])
        task_params["scheduler"] = get_scheduler(
            cfg["scheduler"]["name"], scheduler_params, task_params["optimizer"]
        )
        task_params["num_epochs"] = cfg["num_epochs"]
        task_params["resume_checkpoint"] = args.resume_checkpoint or cfg.get("resume_checkpoint")

    if task != "train":
        test_dataset, _ = get_dataset(
            cfg["test_dataset"]["name"], cfg["test_dataset"]["params"], method="test"
        )
        task_params["test_dataloader"] = DataLoader(test_dataset, **cfg["test_loader"])

    return cfg, init_params, task_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdc_train.yml", help="YAML configuration file")
    parser.add_argument("--task", default="train", choices=["train", "test", "vis", "train_test"])
    parser.add_argument("--data-root", help="Override every dataset root")
    parser.add_argument("--checkpoint", help="Model weights; pass 'none' for random initialization")
    parser.add_argument("--resume-checkpoint", help="Full last_resume.pth state for interrupted training")
    parser.add_argument("--output-dir", help="Fast/local parent directory for run outputs")
    parser.add_argument("--persistent-dir", help="Optional Drive parent directory synced each epoch")
    parser.add_argument("--run-name", help="Override the run/version name")
    parser.add_argument("--device", help="For example cuda:0 or cpu")
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--crop-size", type=int, help="Override the training crop size")
    parser.add_argument("--learning-rate", type=float, help="Override optimizer LR and OneCycle max LR")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--patch-size", type=int, help="Evaluation patch size")
    parser.add_argument("--eval-split", help="Split file used by the test dataset, e.g. val.txt or test.txt")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, init_params, task_params = load_config(args.config, args.task, args)
    trainer = DGTrainer(**init_params)
    copied_config = Path(trainer.log_dir) / Path(args.config).name
    shutil.copy2(args.config, copied_config)
    with (Path(trainer.log_dir) / "resolved_config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    if args.task == "train":
        trainer.train(**task_params)
    elif args.task == "test":
        trainer.test(**task_params)
    elif args.task == "vis":
        trainer.vis(**task_params)
    elif args.task == "train_test":
        test_loader = task_params.pop("test_dataloader")
        trainer.train(**task_params)
        trainer.test(task_params["model"], test_loader, str(Path(trainer.log_dir) / "best.pth"))


if __name__ == "__main__":
    main()
