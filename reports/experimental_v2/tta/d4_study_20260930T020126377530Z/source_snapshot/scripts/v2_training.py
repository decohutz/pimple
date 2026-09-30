"""Explicit CNN configuration, validation-only training and predeclared seed selection."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import time
from urllib.parse import urlparse
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T

from scripts.experimental_protocol import promotion_gate
from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import (content_id, environment, identifier, provenance, read_json, relative_path,
                                 run_id, seal_bundle, utc_now, verify_bundle, write_json)
from scripts.v2_data import dataset_context, experiment_identity, load_development
from scripts.v2_metrics import group_bootstrap, metrics_from_logits, read_predictions, write_predictions


def default_config(architecture: str = "resnet50", seed: int = 42) -> dict:
    weights = {"resnet50": "IMAGENET1K_V2", "mobilenet_v3_large": "IMAGENET1K_V2", "efficientnet_b0": "IMAGENET1K_V1"}
    if architecture not in weights:
        raise ValueError("Architecture is outside the existing project families")
    return {"architecture": architecture, "pretrained_weights": weights[architecture],
            "allow_pretrained_download": False, "image_size": 224,
            "batch_size": 192 if architecture == "mobilenet_v3_large" else 128,
            "epochs": 18, "optimizer": "AdamW", "learning_rate": 3e-4,
            "weight_decay": 5e-5 if architecture == "mobilenet_v3_large" else 1e-4,
            "scheduler": "CosineAnnealingLR", "loss": "cross_entropy", "loss_reduction": "weighted_mean_true_class", "label_smoothing": .05,
            "class_weighting": "inverse_frequency_mean_one", "sampling": "shuffle",
            "augmentations": {"horizontal_flip_p": .5, "vertical_flip_p": .5, "rotation_degrees": 180,
                              "brightness": .1, "contrast": .1, "saturation": .1, "hue": .02},
            "seed": seed, "early_stopping_patience": 3, "checkpoint_metric": "validation_macro_f1",
            "checkpoint_tie": "earliest_epoch", "nan_policy": "fail_run", "freeze_backbone_epochs": 0,
            "num_workers": 0, "device": "auto", "amp": True, "deterministic": True, "tf32": False}


def validate_config(config: dict) -> None:
    if set(config) != set(default_config(config["architecture"])):
        raise ValueError("Training configuration has missing/unknown fields")
    fixed = {"optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "loss": "cross_entropy",
             "loss_reduction": "weighted_mean_true_class", "checkpoint_metric": "validation_macro_f1", "checkpoint_tie": "earliest_epoch",
             "nan_policy": "fail_run", "freeze_backbone_epochs": 0}
    if any(config[k] != v for k, v in fixed.items()):
        raise ValueError("Unsupported training behavior; version the implementation explicitly")
    for k in ["image_size", "batch_size", "epochs", "early_stopping_patience"]:
        if type(config[k]) is not int or config[k] < 1: raise ValueError(f"Invalid {k}")
    if type(config["seed"]) is not int or config["seed"] < 0 or type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("Invalid seed/workers")
    if not 0 <= config["label_smoothing"] < 1 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimization parameters")
    if not all(np.isfinite(config[k]) for k in ["label_smoothing", "learning_rate", "weight_decay"]):
        raise ValueError("Optimization parameters must be finite")
    allowed_weights = {"IMAGENET1K_V1"} if config["architecture"] == "efficientnet_b0" else {"IMAGENET1K_V1", "IMAGENET1K_V2"}
    if config["pretrained_weights"] is not None and config["pretrained_weights"] not in allowed_weights:
        raise ValueError("Use an explicit pretrained weights version, never DEFAULT")
    if config["class_weighting"] not in {"inverse_frequency_mean_one", "none"} or config["sampling"] not in {"shuffle", "weighted"}:
        raise ValueError("Unknown imbalance strategy")
    if config["sampling"] == "weighted" and config["class_weighting"] != "none":
        raise ValueError("Do not combine class weighting and weighted sampling")
    if set(config["augmentations"]) != set(default_config()["augmentations"]):
        raise ValueError("Augmentation configuration differs from implementation")
    if not all(np.isfinite(v) and v >= 0 for v in config["augmentations"].values()):
        raise ValueError("Invalid augmentation parameter")
    if any(config["augmentations"][k] > 1 for k in ["horizontal_flip_p", "vertical_flip_p"]) or config["augmentations"]["hue"] > .5:
        raise ValueError("Invalid flip probability/hue")
    for k in ["amp", "tf32", "deterministic", "allow_pretrained_download"]:
        if type(config[k]) is not bool: raise ValueError(f"{k} must be boolean")


def cuda_build_support() -> dict:
    if not torch.cuda.is_available():
        return {"available": False, "supported": False, "reason": "CUDA unavailable"}
    major, minor = torch.cuda.get_device_capability(0)
    target = f"sm_{major}{minor}"
    architectures = torch.cuda.get_arch_list()
    # Conservative preflight: no assumption of PTX forward compatibility.
    return {"available": True, "supported": target in architectures,
            "device_architecture": target, "compiled_architectures": architectures,
            "reason": "Device architecture explicitly present in this PyTorch build" if target in architectures
                      else "Device architecture absent from this PyTorch build; CUDA must be validated in a compatible environment"}


def resolve_config(config: dict) -> dict:
    config = copy.deepcopy(config); validate_config(config)
    support = cuda_build_support() if config["device"] in {"auto", "cuda"} else {"supported": False}
    if config["device"] == "auto":
        config["device"] = "cuda" if support["supported"] else "cpu"
        if support["available"] and not support["supported"]:
            warnings.warn(support["reason"] + "; auto resolved to CPU", RuntimeWarning)
    if config["device"] not in {"cpu", "cuda"} or (config["device"] == "cuda" and not support["supported"]):
        raise ValueError("Requested device is unavailable/unsupported")
    config["amp"] = bool(config["amp"] and config["device"] == "cuda")
    return config


def recipe_identity(config: dict) -> str:
    return "recipe_" + content_id({k: v for k, v in config.items() if k not in {"seed", "allow_pretrained_download"}})[:16]


def preprocessing(config: dict) -> dict:
    return {"type": "torchvision_imagenet_classification", "image_size": [config["image_size"]] * 2,
            "normalize_mean": [.485, .456, .406], "normalize_std": [.229, .224, .225],
            "color_mode": "RGB", "exif_transpose": True, "interpolation": "bilinear", "antialias": True}


def evaluation_transform(preprocess: dict):
    expected = preprocessing({"image_size": preprocess["image_size"][0]})
    if preprocess != expected:
        raise ValueError("Unsupported/inconsistent preprocessing contract")
    return T.Compose([T.Resize(preprocess["image_size"], interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                      T.ToTensor(), T.Normalize(preprocess["normalize_mean"], preprocess["normalize_std"])])


def training_transform(config: dict):
    a = config["augmentations"]; p = preprocessing(config)
    return T.Compose([T.Resize(p["image_size"], interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                      T.RandomHorizontalFlip(a["horizontal_flip_p"]), T.RandomVerticalFlip(a["vertical_flip_p"]),
                      T.RandomRotation(a["rotation_degrees"], interpolation=T.InterpolationMode.NEAREST, fill=0),
                      T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
                      T.ToTensor(), T.Normalize(p["normalize_mean"], p["normalize_std"])])


class LesionDataset(Dataset):
    def __init__(self, root: Path, frame: pd.DataFrame, transform):
        self.root, self.frame, self.transform = root, frame.reset_index(drop=True), transform

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(relative_path(self.root, row.image_path)) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, int(row.target_index), row.image_stem


def seed_everything(config: dict):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config["deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def build_model(config: dict, initialize_pretrained: bool = True):
    name = config["architecture"]
    constructors = {"resnet50": (models.resnet50, models.ResNet50_Weights),
                    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights),
                    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights)}
    constructor, enum = constructors[name]
    weights = enum[config["pretrained_weights"]] if initialize_pretrained and config["pretrained_weights"] is not None else None
    origin = {"weights": str(weights), "sha256": None}
    if weights is not None:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        if not cached.is_file() and not config["allow_pretrained_download"]:
            raise FileNotFoundError("Pretrained weights not cached; explicitly configure their acquisition before training")
    model = constructor(weights=weights)
    if weights is not None: origin["sha256"] = sha256_file(cached)
    if name == "resnet50": model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    else:
        idx = 3 if name == "mobilenet_v3_large" else 1
        model.classifier[idx] = nn.Linear(model.classifier[idx].in_features, len(CLASSES))
    model.pretrained_origin = origin
    return model


def make_loaders(root: Path, frames: dict, config: dict):
    if set(frames) != {"train", "validation"} or any(set(df.split) != {role} for role, df in frames.items()):
        raise ValueError("Training loaders accept exactly train and validation")
    counts = np.bincount(frames["train"].target_index, minlength=7)
    if (counts == 0).any(): raise ValueError("Training class missing")
    weights = counts.sum() / (7 * counts); weights /= weights.mean()
    generator = torch.Generator().manual_seed(config["seed"])
    sampler = None
    if config["sampling"] == "weighted":
        sampler = WeightedRandomSampler(weights[frames["train"].target_index], len(frames["train"]), replacement=True, generator=generator)
    common = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
              "pin_memory": config["device"] == "cuda", "worker_init_fn": seed_worker,
              "persistent_workers": False, "drop_last": False}
    train = DataLoader(LesionDataset(root, frames["train"], training_transform(config)), shuffle=sampler is None,
                       sampler=sampler, generator=generator, **common)
    validation = DataLoader(LesionDataset(root, frames["validation"], evaluation_transform(preprocessing(config))),
                            shuffle=False, generator=torch.Generator().manual_seed(config["seed"] + 1), **common)
    return train, validation, weights


@torch.inference_mode()
def predict_loader(model, loader, device: str, criterion=None):
    model.eval(); logits, targets, ids = [], [], []; loss_sum, loss_denominator = 0.0, 0.0
    for x, y, batch_ids in loader:
        out = model(x.to(device)).float()
        if not torch.isfinite(out).all(): raise ValueError("Nonfinite inference logits")
        if criterion is not None:
            truth = y.to(device)
            loss_sum += float(criterion(out, truth).sum().item())
            loss_denominator += float(criterion.weight[truth].sum().item()) if criterion.weight is not None else len(y)
        logits.append(out.cpu().numpy()); targets.append(y.numpy()); ids.extend(batch_ids)
    if not targets: raise ValueError("Empty evaluation loader")
    targets = np.concatenate(targets)
    return np.concatenate(logits), targets, ids, loss_sum / loss_denominator if criterion is not None else None


def checkpoint_improved(score: float, best: float) -> bool:
    if not np.isfinite(score): raise ValueError("Nonfinite validation score; run cannot be selected")
    return score > best  # Exact ties retain the earliest epoch.


def fit_epochs(model, train_loader, validation_loader, config: dict, class_weights, on_best, on_epoch=None):
    """Small testable engine. Caller owns checkpoint persistence and run identity."""
    device = config["device"]; model.to(device)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if config["class_weighting"] != "none" else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config["label_smoothing"], reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    best, best_epoch, stale = -float("inf"), None, 0
    history, stop = [], "epochs_completed"
    for epoch in range(1, config["epochs"] + 1):
        model.train(); total, n, loss_weight = 0.0, 0, 0.0; lr = optimizer.param_groups[0]["lr"]
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=config["amp"]):
                losses = criterion(model(x), y)
                denominator = weights[y].sum() if weights is not None else len(y)
                loss = losses.sum() / denominator
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(losses.sum().item()); loss_weight += float(denominator); n += len(y)
        if n == 0: raise ValueError("Empty training loader")
        logits, y, ids, validation_loss = predict_loader(model, validation_loader, device, criterion)
        metrics = metrics_from_logits(y, logits)
        if not np.isfinite(validation_loss): raise ValueError("Nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": total / loss_weight, "validation_loss": validation_loss,
                        "validation_nll": metrics["nll"], "validation_accuracy": metrics["accuracy"],
                        "validation_macro_f1": metrics["macro_f1"], "learning_rate": lr})
        if on_epoch is not None: on_epoch(history)
        if checkpoint_improved(metrics["macro_f1"], best):
            best, best_epoch, stale = metrics["macro_f1"], epoch, 0
            on_best(epoch, model, metrics)
        else: stale += 1
        print(f"epoch={epoch} train_loss={total/loss_weight:.4f} validation_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        scheduler.step()
        if stale >= config["early_stopping_patience"]:
            stop = "early_stopping_validation_macro_f1"; break
    return pd.DataFrame(history), {"best_epoch": best_epoch, "stop_reason": stop, "epochs_ran": len(history)}


def register_baseline_criteria(root: Path, baseline_dir: Path, output: Path) -> dict:
    verify_bundle(baseline_dir)
    baseline = read_json(baseline_dir / "baseline_results.json")
    data = experiment_identity(dataset_context(root))
    if baseline["kind"] != "baseline_reference" or baseline["split_role"] != "validation" or baseline["dataset"] != data:
        raise ValueError("Criteria require baseline validation results on this v2 split")
    if not all(v.get("converged", True) for v in baseline["results"].values()):
        raise ValueError("An unconverged baseline cannot establish the reference")
    scores = {k: v["validation"]["macro_f1"] for k, v in baseline["results"].items()}
    criteria = {"minimum_validation_macro_f1_mean": max(scores.values()), "minimum_validation_recall": {}, "require_calibration": False}
    record = {"registered_at_utc": utc_now(), "dataset": data, "criteria": criteria,
              "rationale": "At least match the strongest prespecified v2 baseline on validation; no clinical recall floors.",
              "baseline_sha256": sha256_file(baseline_dir / "baseline_results.json"), "baseline_validation_macro_f1": scores}
    record["criteria_id"] = content_id(record)
    write_json(output, record)
    return record


def register_plan(root: Path, experiment_id: str, configs: list[dict], seeds: list[int], criteria_path: Path) -> Path:
    identifier(experiment_id)
    criteria = read_json(criteria_path)
    if content_id({k: v for k, v in criteria.items() if k != "criteria_id"}) != criteria["criteria_id"]:
        raise ValueError("Criteria identity changed")
    data = experiment_identity(dataset_context(root))
    if criteria["dataset"] != data: raise ValueError("Criteria are from another dataset/split")
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Declare unique training seeds before training")
    recipes = {}
    for config in configs:
        config = resolve_config(config)
        recipe = recipe_identity(config)
        if recipe in recipes: raise ValueError("Duplicate recipe")
        recipes[recipe] = {k: v for k, v in config.items() if k != "seed"}
    if not recipes: raise ValueError("At least one recipe is required")
    plan = {"experiment_id": experiment_id, "registered_at_utc": utc_now(), "dataset": data,
            "criteria": criteria, "seeds": sorted(seeds), "recipes": recipes,
            "selection": "mean_validation_macro_f1_then_mean_accuracy_then_recipe_id; candidate_f1_then_accuracy_then_smallest_seed"}
    plan["plan_id"] = content_id(plan)
    path = root / "reports/experimental_v2/plans" / experiment_id / "plan.json"
    write_json(path, plan)
    return path


def read_plan(path: Path) -> dict:
    plan = read_json(path)
    if content_id({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Predeclared plan changed")
    return plan


def train_run(root: Path, plan_path: Path, recipe_id: str, seed: int) -> Path:
    plan = read_plan(plan_path)
    frames, certificate = load_development(root)
    data = experiment_identity(certificate)
    if plan["dataset"] != data or seed not in plan["seeds"] or recipe_id not in plan["recipes"]:
        raise ValueError("Run differs from predeclared plan/dataset")
    config = resolve_config({**plan["recipes"][recipe_id], "seed": seed})
    if recipe_identity(config) != recipe_id: raise ValueError("Effective device/recipe changed after registration")
    rid = run_id(f"{config['architecture']}_seed{seed}")
    # One attempt per plan/recipe/seed. Failed attempts remain visible, not silently replaced.
    write_json(plan_path.parent / "attempts" / f"{recipe_id}_seed{seed}.json", {"run_id": rid, "plan_id": plan["plan_id"]})
    output = root / "models/experimental_v2/runs" / rid
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    write_json(output / "preprocess_config.json", preprocessing(config))
    write_json(output / "label_map.json", {"index_to_label": CLASSES, "label_to_index": {v: i for i, v in enumerate(CLASSES)}})
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    write_json(output / "environment.json", environment(include_torch=True))
    prov = provenance(root, output, data)
    prov.update(experiment_id=plan["experiment_id"], run_id=rid, recipe_id=recipe_id, seed=seed,
                architecture=config["architecture"], plan_id=plan["plan_id"])
    write_json(output / "provenance.json", prov)
    checkpoint = output / "checkpoint.pt"
    try:
        seed_everything(config)
        model = build_model(config)
        write_json(output / "initialization.json", model.pretrained_origin)
        train, validation, weights = make_loaders(root, frames, config)
        write_json(output / "class_weights.json", {"source": "train", "class_order": CLASSES, "values": weights.tolist(), "applied_to_loss": config["class_weighting"] != "none"})
        config_hash, prov_hash = sha256_file(output / "config.json"), sha256_file(output / "provenance.json")
        def save_best(epoch, model, metrics):
            torch.save({"state_dict": model.state_dict(), "model_name": config["architecture"], "img_size": config["image_size"],
                        "num_classes": 7, "index_to_label": CLASSES, "epoch": epoch, "seed": seed,
                        "recipe_id": recipe_id, "config_sha256": config_hash, "provenance_sha256": prov_hash,
                        "preprocess_sha256": sha256_file(output / "preprocess_config.json"),
                        "split_version": data["split_version"], "val_f1_macro": metrics["macro_f1"]}, checkpoint)
        if config["device"] == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        history, stopping = fit_epochs(model, train, validation, config, weights, save_best,
                                      on_epoch=lambda rows: pd.DataFrame(rows).to_csv(output / "training_history.csv", index=False))
        if config["device"] == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        history.to_csv(output / "training_history.csv", index=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        logits, y, ids, _ = predict_loader(model, validation, config["device"])
        digest = sha256_file(checkpoint)
        metrics = write_predictions(output, frames["validation"], logits, ids, y, digest, "validation")
        write_json(output / "validation_interval.json", group_bootstrap(y, logits.argmax(1), frames["validation"].group_id))
        (output / "checkpoint.sha256").write_text(digest + "\n", encoding="ascii")
        write_json(output / "run_summary.json", {"status": "completed", "run_id": rid, "experiment_id": plan["experiment_id"],
                   "recipe_id": recipe_id, "architecture": config["architecture"], "seed": seed, "dataset": data,
                   "plan_id": plan["plan_id"], **stopping, "validation_macro_f1": metrics["macro_f1"],
                   "validation_accuracy": metrics["accuracy"], "training_seconds": elapsed,
                   "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_bytes": checkpoint.stat().st_size,
                   "peak_gpu_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None,
                   "inference_latency_ms": None, "latency_note": "Not measured; no cross-device latency claim"})
        seal_bundle(output)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc), "created_at_utc": utc_now()})
        raise
    return output


def validate_run(directory: Path) -> dict:
    verify_bundle(directory)
    config, prov = read_json(directory / "config.json"), read_json(directory / "provenance.json")
    summary, plan = read_json(directory / "run_summary.json"), read_plan(directory / "plan.json")
    validate_config(config)
    ckpt = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    required = {"config_sha256": sha256_file(directory / "config.json"), "provenance_sha256": sha256_file(directory / "provenance.json"),
                "preprocess_sha256": sha256_file(directory / "preprocess_config.json"), "index_to_label": CLASSES,
                "model_name": config["architecture"], "seed": config["seed"], "num_classes": 7,
                "img_size": config["image_size"], "split_version": prov["dataset"]["split_version"], "recipe_id": recipe_identity(config)}
    if any(ckpt.get(k) != v for k, v in required.items()): raise ValueError("Checkpoint/config/provenance mismatch")
    if read_json(directory / "preprocess_config.json") != preprocessing(config) or read_json(directory / "label_map.json")["index_to_label"] != CLASSES:
        raise ValueError("Preprocessing/class mapping mismatch")
    if summary["status"] != "completed" or summary["dataset"] != plan["dataset"] or prov["dataset"] != plan["dataset"]:
        raise ValueError("Incomplete run or dataset provenance mismatch")
    if summary["plan_id"] != plan["plan_id"] or summary["recipe_id"] != recipe_identity(config) or summary["seed"] != config["seed"]:
        raise ValueError("Run selection identity mismatch")
    expected = {**plan["recipes"][summary["recipe_id"]], "seed": config["seed"]}
    if config != expected or config["seed"] not in plan["seeds"]: raise ValueError("Config differs from plan")
    history = pd.read_csv(directory / "training_history.csv")
    if history.empty or not np.isfinite(history.validation_macro_f1).all():
        raise ValueError("Invalid validation history")
    best = history.sort_values(["validation_macro_f1", "epoch"], ascending=[False, True]).iloc[0]
    if ckpt["epoch"] != int(best.epoch) or summary["best_epoch"] != int(best.epoch) or summary["epochs_ran"] != len(history):
        raise ValueError("Checkpoint epoch does not match validation-only selection")
    if (directory / "checkpoint.sha256").read_text().strip() != sha256_file(directory / "checkpoint.pt"):
        raise ValueError("Checkpoint digest differs from its sidecar")
    table, logits = read_predictions(directory, "validation", sha256_file(directory / "checkpoint.pt"))
    measured = metrics_from_logits(table.target_index, logits, table.group_id)
    saved = read_json(directory / "validation_metrics.json")
    if saved["split_role"] != "validation" or not np.isclose(summary["validation_macro_f1"], measured["macro_f1"], rtol=0, atol=1e-12):
        raise ValueError("Validation score differs from persisted predictions")
    for k in ["macro_f1", "accuracy"]:
        if not np.isclose(saved[k], measured[k], rtol=0, atol=1e-12): raise ValueError("Validation metrics changed")
    if not np.isclose(summary["validation_accuracy"], measured["accuracy"], rtol=0, atol=1e-12):
        raise ValueError("Validation accuracy differs from predictions")
    if not np.isclose(ckpt["val_f1_macro"], best.validation_macro_f1, rtol=0, atol=1e-12) or not np.isclose(ckpt["val_f1_macro"], measured["macro_f1"], rtol=0, atol=1e-12):
        raise ValueError("Checkpoint score differs from history/predictions")
    return summary


def select_candidate(root: Path, plan_path: Path, output: Path) -> dict:
    plan = read_plan(plan_path)
    rows, paths = [], {}
    expected = {(recipe, seed) for recipe in plan["recipes"] for seed in plan["seeds"]}
    attempts = list((plan_path.parent / "attempts").glob("*.json"))
    for path in attempts:
        attempt = read_json(path)
        folder = root / "models/experimental_v2/runs" / identifier(attempt["run_id"])
        summary = validate_run(folder)  # Failed/partial attempts stop selection explicitly.
        if summary["plan_id"] != plan["plan_id"]: raise ValueError("Run is outside plan")
        rows.append(summary); paths[summary["run_id"]] = folder
    seen = [(r["recipe_id"], r["seed"]) for r in rows]
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError("Selection requires exactly one complete attempt per planned recipe/seed")
    df = pd.DataFrame(rows)
    stats = df.groupby(["recipe_id", "architecture"], as_index=False).agg(
        runs=("seed", "size"), unique_seeds=("seed", "nunique"), mean=("validation_macro_f1", "mean"),
        std=("validation_macro_f1", "std"), min=("validation_macro_f1", "min"), max=("validation_macro_f1", "max"),
        accuracy_mean=("validation_accuracy", "mean"), training_seconds_mean=("training_seconds", "mean"),
        parameters=("parameters", "first"), checkpoint_bytes_mean=("checkpoint_bytes", "mean"))
    stats = stats.sort_values(["mean", "accuracy_mean", "recipe_id"], ascending=[False, False, True])
    winner = stats.iloc[0]
    selected = df.loc[df.recipe_id == winner.recipe_id].sort_values(
        ["validation_macro_f1", "validation_accuracy", "seed"], ascending=[False, False, True]).iloc[0]
    candidate_dir = paths[selected.run_id]
    output.mkdir(parents=True, exist_ok=False)
    df.drop(columns="dataset").to_csv(output / "runs.csv", index=False)
    stats.to_csv(output / "recipes.csv", index=False)
    selection = {"plan_id": plan["plan_id"], "experiment_id": plan["experiment_id"], "dataset": plan["dataset"],
                 "split_role": "validation", "selection_policy": plan["selection"], "unique_recipes": len(stats),
                 "total_runs": len(rows), "unique_training_seeds": len(plan["seeds"]),
                 "recipe_id": selected.recipe_id, "candidate_run_id": selected.run_id,
                 "candidate_directory": candidate_dir.relative_to(root).as_posix(),
                 "candidate_artifacts_sha256": sha256_file(candidate_dir / "artifacts.json"),
                 "validation_macro_f1_mean": float(winner["mean"]),
                 "validation_macro_f1_std": None if pd.isna(winner["std"]) else float(winner["std"]),
                 "run_artifact_hashes": {rid: sha256_file(folder / "artifacts.json") for rid, folder in paths.items()}}
    write_json(output / "selection.json", selection)
    write_json(output / "plan.json", plan)
    seal_bundle(output)
    return selection
