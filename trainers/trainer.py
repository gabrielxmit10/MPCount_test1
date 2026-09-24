"""Reusable training loop with resumable and Drive-safe checkpoints."""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils.misc import AverageMeter, DictAvgMeter, easy_track, get_current_datetime


class Trainer:
    def __init__(self, seed, version, device, output_dir="logs", persistent_dir=None):
        self.seed = seed
        self.version = version
        self.device = torch.device(device)
        self.log_dir = os.path.join(output_dir, self.version)
        os.makedirs(self.log_dir, exist_ok=True)
        self.persistent_run_dir = None
        if persistent_dir:
            self.persistent_run_dir = os.path.join(persistent_dir, self.version)
            os.makedirs(self.persistent_run_dir, exist_ok=True)

    def log(self, msg, verbose=True, **kwargs):
        if verbose:
            print(msg, **kwargs)
        with open(os.path.join(self.log_dir, "log.txt"), "a", encoding="utf-8") as handle:
            handle.write(msg + kwargs.get("end", "\n"))

    @staticmethod
    def _torch_load(path, device):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch 2.0 has no weights_only argument.
            return torch.load(path, map_location=device)

    @staticmethod
    def _model_state(model):
        if isinstance(model, nn.Module):
            return model.state_dict()
        return [module.state_dict() for module in model]

    def _load_model_state(self, model, state, strict=True):
        if isinstance(model, nn.Module):
            incompatible = model.load_state_dict(state, strict=strict)
            if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
                self.log(
                    f"Partial checkpoint: {len(incompatible.missing_keys)} missing and "
                    f"{len(incompatible.unexpected_keys)} unexpected keys"
                )
        else:
            for module, module_state in zip(model, state):
                module.load_state_dict(module_state, strict=strict)

    @staticmethod
    def _extract_model_state(payload):
        if not isinstance(payload, dict):
            return payload
        for key in ("model_state_dict", "model", "state_dict"):
            if key in payload:
                return payload[key]
        return payload

    def load_ckpt(self, model, path, strict=True):
        if path:
            self.log(f"Loading model weights from {path}")
            payload = self._torch_load(path, self.device)
            self._load_model_state(model, self._extract_model_state(payload), strict=strict)

    @staticmethod
    def _atomic_torch_save(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def save_ckpt(self, model, path):
        self._atomic_torch_save(self._model_state(model), path)

    def _save_resume(self, model, optimizer, scheduler, next_epoch, best_criterion, best_epoch):
        payload = {
            "format_version": 1,
            "model_state_dict": self._model_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "next_epoch": next_epoch,
            "best_criterion": best_criterion,
            "best_epoch": best_epoch,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        self._atomic_torch_save(payload, os.path.join(self.log_dir, "last_resume.pth"))

    def _load_resume(self, model, optimizer, scheduler, path):
        self.log(f"Resuming complete training state from {path}")
        payload = self._torch_load(path, self.device)
        required = {"model_state_dict", "optimizer_state_dict", "next_epoch"}
        missing = required - set(payload) if isinstance(payload, dict) else required
        if missing:
            raise ValueError(f"Resume checkpoint is not a full training state; missing {sorted(missing)}")
        self._load_model_state(model, payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload.get("python_random_state") is not None:
            random.setstate(payload["python_random_state"])
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.set_rng_state(payload["torch_random_state"])
        if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_random_state"])
        return (
            int(payload["next_epoch"]),
            float(payload.get("best_criterion", float("inf"))),
            int(payload.get("best_epoch", -1)),
        )

    def _sync_persistent(self, names=None):
        if not self.persistent_run_dir:
            return
        names = set(names) if names is not None else None
        for source in Path(self.log_dir).iterdir():
            if (
                source.is_file()
                and source.suffix in {".pth", ".txt", ".csv", ".yml", ".json"}
                and (names is None or source.name in names)
            ):
                try:
                    destination = Path(self.persistent_run_dir) / source.name
                    temporary = destination.with_name(destination.name + ".tmp")
                    shutil.copy2(source, temporary)
                    os.replace(temporary, destination)
                except OSError as exc:
                    self.log(f"WARNING: could not sync {source.name} to persistent storage: {exc}")

    def set_model_train(self, model):
        if isinstance(model, nn.Module):
            model.train()
        else:
            for module in model:
                module.train()

    def set_model_eval(self, model):
        if isinstance(model, nn.Module):
            model.eval()
        else:
            for module in model:
                module.eval()

    def train_step(self, model, loss, optimizer, batch, epoch):
        raise NotImplementedError

    def val_step(self, model, batch):
        raise NotImplementedError

    def test_step(self, model, batch):
        raise NotImplementedError

    def vis_step(self, model, batch):
        raise NotImplementedError

    def train_epoch(
        self,
        model,
        loss,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        epoch,
        best_criterion,
        best_epoch,
    ):
        start_time = time.time()
        self.set_model_train(model)
        loss_meter = AverageMeter()
        per_batch_scheduler = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
        for batch in easy_track(train_dataloader, description=f"Epoch {epoch}: Training..."):
            train_loss = self.train_step(model, loss, optimizer, batch, epoch)
            loss_meter.update(train_loss)
            if per_batch_scheduler:
                scheduler.step()
        self.log(f"Epoch {epoch}: Training loss: {loss_meter.avg:.4f} Version: {self.version}")

        self.set_model_eval(model)
        criterion_meter = AverageMeter()
        additional_meter = DictAvgMeter()
        for batch in easy_track(val_dataloader, description=f"Epoch {epoch}: Validating..."):
            with torch.no_grad():
                criterion, additional = self.val_step(model, batch)
            n = additional.pop("n", 1)
            criterion_meter.update(criterion, n)
            additional_meter.update(additional, n)
        current_criterion = criterion_meter.avg

        if scheduler is not None and not per_batch_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_criterion)
            else:
                scheduler.step()

        metrics = " ".join(f"{key}: {value:.4f}" for key, value in additional_meter.avg.items())
        self.log(
            f"Epoch {epoch}: Val MAE: {current_criterion:.4f} {metrics} "
            f"best: {best_criterion:.4f}, time: {time.time() - start_time:.1f}s"
        )

        self.save_ckpt(model, os.path.join(self.log_dir, "last.pth"))
        sync_names = {"last_resume.pth", "log.txt", "resolved_config.yml"}
        if current_criterion < best_criterion:
            best_criterion = current_criterion
            best_epoch = epoch
            self.log(f"Epoch {epoch}: saving best model")
            self.save_ckpt(model, os.path.join(self.log_dir, "best.pth"))
            sync_names.add("best.pth")
        self._save_resume(model, optimizer, scheduler, epoch + 1, best_criterion, best_epoch)
        self._sync_persistent(sync_names)
        return best_criterion, best_epoch

    def train(
        self,
        model,
        loss,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        checkpoint=None,
        resume_checkpoint=None,
        num_epochs=100,
    ):
        self.log(f"Start training at {get_current_datetime()}")
        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]
        loss = loss.to(self.device)
        start_epoch, best_criterion, best_epoch = 0, float("inf"), -1
        if resume_checkpoint:
            start_epoch, best_criterion, best_epoch = self._load_resume(
                model, optimizer, scheduler, resume_checkpoint
            )
        else:
            self.load_ckpt(model, checkpoint)

        if start_epoch >= num_epochs:
            self.log(f"Resume checkpoint already reached epoch {start_epoch}; requested total is {num_epochs}")
        for epoch in range(start_epoch, num_epochs):
            best_criterion, best_epoch = self.train_epoch(
                model,
                loss,
                train_dataloader,
                val_dataloader,
                optimizer,
                scheduler,
                epoch,
                best_criterion,
                best_epoch,
            )

        self.log(f"Best epoch: {best_epoch}, best MAE: {best_criterion}")
        self.log(f"Training results saved to {self.log_dir}")
        self.log(f"End training at {get_current_datetime()}")
        self._sync_persistent()

    def test(self, model, test_dataloader, checkpoint=None):
        self.log(f"Start testing at {get_current_datetime()}")
        self.load_ckpt(model, checkpoint)
        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]
        self.set_model_eval(model)
        result_meter = DictAvgMeter()
        details = []
        for batch in easy_track(test_dataloader, description="Testing..."):
            with torch.no_grad():
                result = self.test_step(model, batch)
            detail = result.pop("_detail", None)
            if detail:
                details.append(detail)
            result_meter.update(result)

        mae = result_meter.avg.get("mae", float("nan"))
        rmse = math.sqrt(result_meter.avg.get("squared_error", float("nan")))
        self.log(f"Testing results: MAE: {mae:.4f} RMSE: {rmse:.4f}")
        results_path = Path(self.log_dir) / "test_predictions.csv"
        if details:
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "predicted_count", "ground_truth_count", "absolute_error"])
                writer.writeheader()
                writer.writerows(details)
        self.log(f"Testing results saved to {self.log_dir}")
        self.log(f"End testing at {get_current_datetime()}")
        self._sync_persistent()
        return {"mae": mae, "rmse": rmse}

    def vis(self, model, test_dataloader, checkpoint=None):
        self.log(f"Start visualization at {get_current_datetime()}")
        self.load_ckpt(model, checkpoint)
        os.makedirs(os.path.join(self.log_dir, "vis"), exist_ok=True)
        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]
        self.set_model_eval(model)
        for batch in easy_track(test_dataloader, description="Visualizing..."):
            with torch.no_grad():
                self.vis_step(model, batch)
        self.log(f"Visualization results saved to {self.log_dir}")
        self.log(f"End visualization at {get_current_datetime()}")
