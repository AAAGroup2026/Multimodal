"""Train and evaluate RIMoE on MOSI or MOSEI."""
import argparse
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import get_config_regression
from .data import MMDataset
from .model import RIMoE
from .bayesian_structure import laplace_mixture_statistics
from .losses import entropy_balance, eva_imp
from .metrics import MetricsTop


RELATION_COLUMNS = ("relation_w_ta", "relation_w_tv", "relation_w_av", "relation_w_tav")
MODALITY_COLUMNS = ("modality_w_t", "modality_w_v", "modality_w_a")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(model="rimoe")
    parser.add_argument("--dataset", choices=("mosi", "mosei"), default="mosi")
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--pretrained", default="bert-base-uncased")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--resume-existing", default=None,
                        help="Resume an existing run directory; reuse saved seed checkpoints")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 1112, 1113, 1114, 1115])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--bert-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--aism-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient clipping max norm; 0 disables clipping (default)")
    parser.add_argument("--fusion-method", choices=("sum", "concat"), default=None)
    parser.add_argument("--alignment", choices=("aligned", "unaligned"), default="aligned")
    parser.add_argument("--lambda-uni", type=float, default=1.0)
    parser.add_argument("--lambda-ent", type=float, default=0.1)
    parser.add_argument("--lambda-router", type=float, default=0.01)
    parser.add_argument("--aism-dim", type=int, default=256)
    parser.add_argument("--aism-num-heads", type=int, default=4)
    parser.add_argument("--aism-num-layers", type=int, default=1)
    parser.add_argument("--aism-dropout", type=float, default=0.1)
    parser.add_argument("--aism-temperature", type=float, default=1.0)
    parser.add_argument("--aism-router-hidden", type=int, default=128)
    parser.add_argument("--bayes-scale", type=float, default=1.0)
    parser.add_argument("--bayes-objective", choices=("nll", "mean_mae"), default="nll")
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def build_model(args):
    return RIMoE(args)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loaders(args, seed):
    datasets = {split: MMDataset(args, mode=split) for split in ("train", "valid", "test")}
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Train, valid and test splits must all contain at least one sample")
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders


def make_config(cli_args, seed):
    args = get_config_regression("rimoe", cli_args.dataset)
    args.model_name = cli_args.model
    args.dataset_name = cli_args.dataset
    args.featurePath = str(Path(cli_args.feature_path).expanduser().resolve())
    args.pretrained = cli_args.pretrained
    args.use_bert = True
    args.use_finetune = True
    args.need_data_aligned = cli_args.alignment == "aligned"
    args.need_model_aligned = True
    args.train_mode = "regression"
    args.cur_seed = seed
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.learning_rate = cli_args.backbone_lr
    args.early_stop = cli_args.early_stop
    args.update_epochs = 1
    args.grad_clip = cli_args.grad_clip
    args.device = None
    args.fusion_method = cli_args.fusion_method or args.fusion_method
    args.temperature = 0.1  # Modality-router temperature.
    args.aism_num_heads = cli_args.aism_num_heads
    args.aism_dim = cli_args.aism_dim
    args.aism_num_layers = cli_args.aism_num_layers
    args.aism_dropout = cli_args.aism_dropout
    args.aism_temperature = cli_args.aism_temperature
    args.aism_router_hidden = cli_args.aism_router_hidden
    args.bayes_scale = cli_args.bayes_scale
    args.bayes_objective = cli_args.bayes_objective
    return args


def make_optimizer(model, args, cli_args):
    groups = {"bert": [], "backbone": [], "aism": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("aism.", "aism_input_projections.", "aism_adapter.", "aism_scale_logit")):
            groups["aism"].append(parameter)
        elif name.startswith("text_model."):
            groups["bert"].append(parameter)
        else:
            groups["backbone"].append(parameter)
    param_groups = []
    if groups["bert"]:
        param_groups.append({"params": groups["bert"], "lr": cli_args.bert_lr, "group_name": "bert"})
    if groups["backbone"]:
        param_groups.append({"params": groups["backbone"], "lr": cli_args.backbone_lr, "group_name": "backbone"})
    if groups["aism"]:
        param_groups.append({"params": groups["aism"], "lr": cli_args.aism_lr, "group_name": "aism"})
    return torch.optim.Adam(param_groups)


def count_trainable(model):
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    aism = sum(parameter.numel() for name, parameter in model.named_parameters()
               if parameter.requires_grad and name.startswith(("aism.", "aism_input_projections.", "aism_adapter.", "aism_scale_logit")))
    return total, aism


def compute_loss(output, labels, args, cli_args):
    criterion = F.l1_loss
    bayes_stats = laplace_mixture_statistics(
        output["component_means"], output["structure_log_prior"],
        labels, getattr(args, "bayes_scale", 1.0))
    pred_mae = criterion(output["logits_c"], labels)
    main = bayes_stats["nll"] if cli_args.bayes_objective == "nll" else pred_mae
    uni = (criterion(output["logits_l"], labels) +
           criterion(output["logits_v"], labels) +
           criterion(output["logits_a"], labels)) / 3.0

    t_error = eva_imp(output["logits_l"], labels) + 0.1
    v_error = eva_imp(output["logits_v"], labels) + 0.1
    a_error = eva_imp(output["logits_a"], labels) + 0.1
    inverse_errors = torch.cat((t_error.reciprocal(), v_error.reciprocal(),
                                a_error.reciprocal()), dim=1)
    target_modal_weights = inverse_errors / inverse_errors.sum(dim=1, keepdim=True)
    router = F.mse_loss(output["channel_weight"], target_modal_weights.detach())
    entropy = entropy_balance(output["channel_weight"])

    total = (main + cli_args.lambda_uni * uni + cli_args.lambda_ent * entropy +
             cli_args.lambda_router * router)
    parts = {"main": main, "uni": uni, "entropy": entropy,
             "router": router, "pred_mae": pred_mae, "nll": bayes_stats["nll"]}
    return total, parts


def ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values detected in {name}")


def gradient_diagnostics(model):
    probes = {
        "aism_encoder": [p for n, p in model.named_parameters()
                         if n.startswith("aism.encoder.")],
        "aism_router": [p for n, p in model.named_parameters()
                        if n.startswith("aism.structure_router.")],
        "aism_adapter": [p for n, p in model.named_parameters()
                         if n.startswith("aism_adapter.")],
        "aism_input_projection": [p for n, p in model.named_parameters()
                                  if n.startswith("aism_input_projections.")],
        "aism_scale": [p for n, p in model.named_parameters()
                       if n == "aism_scale_logit"],
        "modality_encoder": [p for n, p in model.named_parameters()
                                  if n.startswith(("self_attentions_l.",
                                                   "self_attentions_a.",
                                                   "self_attentions_v."))],
    }
    result = {}
    for key, parameters in probes.items():
        if not parameters:
            result[key] = "n/a"
            continue
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            result[key] = "none"
        else:
            for grad in grads:
                ensure_finite(f"{key} gradient", grad)
            result[key] = float(torch.sqrt(sum(g.detach().float().pow(2).sum() for g in grads)))
    return result


def _batch_to_device(batch, device):
    text = batch["text"].to(device, non_blocking=True)
    audio = batch["audio"].to(device, non_blocking=True)
    vision = batch["vision"].to(device, non_blocking=True)
    labels = batch["labels"]["M"].to(device, non_blocking=True).view(-1, 1)
    return text, audio, vision, labels


def _numpy_route(output, key, width):
    value = output.get(key)
    if value is None:
        return np.full((output["logits_c"].shape[0], width), np.nan, dtype=np.float32)
    return value.detach().float().cpu().numpy()


def column_nan_stats(values):
    values = np.asarray(values, dtype=np.float64)
    means, stds = [], []
    for column in values.T:
        finite = column[np.isfinite(column)]
        means.append(float(finite.mean()) if finite.size else float("nan"))
        stds.append(float(finite.std()) if finite.size else float("nan"))
    return means, stds


def nullable_values(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _normalize_id(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def evaluate(model, loader, device, metrics_fn, args, collect_rows=False):
    model.eval()
    predictions, labels, ids = [], [], []
    modality_routes, relation_routes, betas = [], [], []
    posterior_routes, component_means = [], []
    nll_values, prior_entropies, posterior_entropies = [], [], []
    prior_argmaxes, posterior_argmaxes = [], []
    batch_loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, target = _batch_to_device(batch, device)
            ensure_finite("evaluation labels", target)
            output = model(text, audio, vision)
            for key in ("logits_c", "logits_l", "logits_v", "logits_a", "channel_weight"):
                ensure_finite(key, output[key])
            pred_cpu, label_cpu = output["logits_c"].detach().cpu(), target.detach().cpu()
            batch_loss_sum += F.l1_loss(output["logits_c"], target).item()
            predictions.append(pred_cpu)
            labels.append(label_cpu)
            modality_routes.append(_numpy_route(output, "channel_weight", 3))
            relation_routes.append(_numpy_route(output, "relation_weight", 4))
            if "component_means" in output and "structure_log_prior" in output:
                stats = laplace_mixture_statistics(
                    output["component_means"], output["structure_log_prior"],
                    target, getattr(args, "bayes_scale", 1.0))
                prior = output["structure_prior"].detach().float()
                posterior = stats["posterior"].detach().float()
                posterior_routes.append(posterior.cpu().numpy())
                component_means.append(output["component_means"].detach().float().cpu().numpy())
                nll_values.append(stats["nll_per_sample"].detach().float().cpu())
                prior_entropies.append((-prior * prior.clamp_min(1e-12).log()).sum(1).cpu())
                posterior_entropies.append((-posterior * posterior.clamp_min(1e-12).log()).sum(1).cpu())
                prior_argmaxes.append(prior.argmax(1).cpu())
                posterior_argmaxes.append(posterior.argmax(1).cpu())
            beta = output.get("aism_beta")
            beta_value = float(beta.detach().cpu()) if beta is not None else float("nan")
            # beta is a single learned scalar for the whole model, while the
            # sample export has one row per test example.
            betas.extend([beta_value] * target.shape[0])
            if collect_rows:
                ids.extend(_normalize_id(item) for item in batch["id"])

    pred_tensor = torch.cat(predictions, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    result = metrics_fn(pred_tensor, label_tensor)
    result = {key: float(value) for key, value in result.items()}
    result["main_mae"] = float(F.l1_loss(pred_tensor, label_tensor))
    result["main_loss"] = result["main_mae"]
    # Mean of batch MAEs, retained as a diagnostic only.
    result["Loss"] = round(batch_loss_sum / len(loader), 4)
    if nll_values:
        nll_tensor = torch.cat(nll_values)
        result["nll"] = float(nll_tensor.mean())
        components = torch.from_numpy(np.concatenate(component_means, axis=0))
        result["component_spread"] = float(components.std(dim=1, unbiased=False).mean())
        result["component_mae_ta"] = float((components[:, 0] - label_tensor.reshape(-1)).abs().mean())
        result["component_mae_tv"] = float((components[:, 1] - label_tensor.reshape(-1)).abs().mean())
        result["component_mae_av"] = float((components[:, 2] - label_tensor.reshape(-1)).abs().mean())
        result["component_mae_tav"] = float((components[:, 3] - label_tensor.reshape(-1)).abs().mean())
        prior_all = torch.from_numpy(np.concatenate(relation_routes, axis=0))
        posterior_all = torch.from_numpy(np.concatenate(posterior_routes, axis=0))
        result["prior_entropy"] = float((-prior_all * prior_all.clamp_min(1e-12).log()).sum(1).mean())
        result["posterior_entropy"] = float((-posterior_all * posterior_all.clamp_min(1e-12).log()).sum(1).mean())
    pred_np = pred_tensor.numpy().reshape(-1)
    label_np = label_tensor.numpy().reshape(-1)
    modality_routes = np.concatenate(modality_routes, axis=0)
    relation_routes = np.concatenate(relation_routes, axis=0)
    modality_mean, modality_std = column_nan_stats(modality_routes)
    relation_mean, relation_std = column_nan_stats(relation_routes)
    finite_betas = np.asarray(betas, dtype=np.float64)
    finite_betas = finite_betas[np.isfinite(finite_betas)]
    route_stats = {
        "modality_mean": modality_mean,
        "modality_std": modality_std,
        "relation_mean": nullable_values(relation_mean),
        "relation_std": nullable_values(relation_std),
        "beta": float(finite_betas.mean()) if finite_betas.size else None,
        "bayes_scale": float(getattr(args, "bayes_scale", float("nan"))) if posterior_routes else None,
        "temperature": float(getattr(args, "aism_temperature", float("nan"))) if posterior_routes else None,
    }
    if posterior_routes:
        posterior_all = np.concatenate(posterior_routes, axis=0)
        prior_all = relation_routes
        prior_entropy_all = torch.cat(prior_entropies).numpy()
        posterior_entropy_all = torch.cat(posterior_entropies).numpy()
        prior_argmax_all = torch.cat(prior_argmaxes).numpy()
        posterior_argmax_all = torch.cat(posterior_argmaxes).numpy()
        route_stats.update({
            "prior_mean": prior_all.mean(axis=0).tolist(),
            "prior_std": prior_all.std(axis=0).tolist(),
            "posterior_mean": posterior_all.mean(axis=0).tolist(),
            "posterior_std": posterior_all.std(axis=0).tolist(),
            "prior_entropy": float(prior_entropy_all.mean()),
            "posterior_entropy": float(posterior_entropy_all.mean()),
            "prior_argmax_frequency": (
                np.bincount(prior_argmax_all, minlength=4).astype(float) / len(prior_all)
            ).tolist(),
            "posterior_argmax_frequency": (
                np.bincount(posterior_argmax_all, minlength=4).astype(float) / len(prior_all)
            ).tolist(),
        })
    sample_rows = None
    if collect_rows:
        sample_rows = {
            "ids": ids,
            "labels": label_np,
            "predictions": pred_np,
            "modality_routes": modality_routes,
            "relation_routes": relation_routes,
            "betas": betas,
        }
        if posterior_routes:
            sample_rows.update({
                "posterior_routes": np.concatenate(posterior_routes, axis=0),
                "component_means": np.concatenate(component_means, axis=0),
                "nll": torch.cat(nll_values).numpy(),
                "prior_entropies": torch.cat(prior_entropies).numpy(),
                "posterior_entropies": torch.cat(posterior_entropies).numpy(),
                "prior_argmaxes": torch.cat(prior_argmaxes).numpy(),
                "posterior_argmaxes": torch.cat(posterior_argmaxes).numpy(),
                "bayes_scale": float(getattr(args, "bayes_scale", 1.0)),
            })
    return result, route_stats, sample_rows


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_exports(seed_dir, rows):
    seed_dir = Path(seed_dir)
    prediction_rows = []
    routing_rows = []
    for index, sample_id in enumerate(rows["ids"]):
        prediction_rows.append({"id": sample_id,
                                "label": float(rows["labels"][index]),
                                "prediction": float(rows["predictions"][index])})
        route = rows["modality_routes"][index]
        relation = rows["relation_routes"][index]
        record = {
            "id": sample_id,
            "label": float(rows["labels"][index]),
            "prediction": float(rows["predictions"][index]),
        }
        record.update({name: float(route[i]) for i, name in enumerate(MODALITY_COLUMNS)})
        record.update({name: (float(relation[i]) if np.isfinite(relation[i]) else "")
                       for i, name in enumerate(RELATION_COLUMNS)})
        record["beta"] = rows["betas"][index] if np.isfinite(rows["betas"][index]) else ""
        bayes_available = "posterior_routes" in rows
        prior = relation if bayes_available else [float("nan")] * 4
        posterior = rows["posterior_routes"][index] if bayes_available else [float("nan")] * 4
        mu = rows["component_means"][index] if bayes_available else [float("nan")] * 4
        for i, name in enumerate(("ta", "tv", "av", "tav")):
            record[f"prior_{name}"] = float(prior[i]) if np.isfinite(prior[i]) else ""
            record[f"posterior_{name}"] = float(posterior[i]) if np.isfinite(posterior[i]) else ""
            record[f"mu_{name}"] = float(mu[i]) if np.isfinite(mu[i]) else ""
        record["prior_entropy"] = float(rows["prior_entropies"][index]) if bayes_available else ""
        record["posterior_entropy"] = float(rows["posterior_entropies"][index]) if bayes_available else ""
        record["prior_argmax"] = int(rows["prior_argmaxes"][index]) if bayes_available else ""
        record["posterior_argmax"] = int(rows["posterior_argmaxes"][index]) if bayes_available else ""
        record["nll"] = float(rows["nll"][index]) if bayes_available else ""
        record["bayes_scale"] = float(rows.get("bayes_scale", float("nan"))) if bayes_available else ""
        routing_rows.append(record)
    write_csv(seed_dir / "test_predictions.csv", prediction_rows)
    write_csv(seed_dir / "test_routing.csv", routing_rows)


def _route_csv_values(stats, prefix, names):
    values = {}
    means = stats["modality_mean"] if prefix == "modality" else stats["relation_mean"]
    stds = stats["modality_std"] if prefix == "modality" else stats["relation_std"]
    for index, name in enumerate(names):
        values[f"{prefix}_{name}_mean"] = float(means[index])
        values[f"{prefix}_{name}_std"] = float(stds[index])
    return values


def train_seed(cli_args, seed, run_dir, device, logger):
    seed_everything(seed)
    args = make_config(cli_args, seed)
    args.device = device
    loaders = make_loaders(args, seed)
    model = build_model(args).to(device)
    optimizer = make_optimizer(model, args, cli_args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cli_args.lr_patience
    )
    metrics_fn = MetricsTop("regression").getMetics(cli_args.dataset)
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    total_params, aism_params = count_trainable(model)
    logger.info("seed=%d trainable_params=%d aism_added_params=%d", seed, total_params, aism_params)

    best_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, cli_args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sums = {key: 0.0 for key in ("total", "main", "nll", "pred_mae", "uni", "entropy", "router")}
        seen = 0
        train_predictions, train_labels = [], []
        modality_routes, relation_routes = [], []
        gradient_info = {"aism_encoder": "n/a", "aism_router": "n/a",
                         "aism_adapter": "n/a", "aism_scale": "n/a"}
        num_batches = len(loaders["train"])
        progress = tqdm(enumerate(loaders["train"]), total=num_batches,
                        desc=f"{cli_args.model} seed={seed} epoch={epoch}", leave=False)
        for batch_index, batch in progress:
            group_start = (batch_index // cli_args.grad_accum_steps) * cli_args.grad_accum_steps
            group_steps = min(cli_args.grad_accum_steps, num_batches - group_start)
            if batch_index == group_start:
                optimizer.zero_grad(set_to_none=True)
            text, audio, vision, labels = _batch_to_device(batch, device)
            ensure_finite("training labels", labels)
            output = model(text, audio, vision)
            loss, parts = compute_loss(output, labels, args, cli_args)
            ensure_finite("total loss", loss)
            (loss / group_steps).backward()

            batch_count = labels.shape[0]
            seen += batch_count
            loss_sums["total"] += float(loss.detach()) * batch_count
            for key, value in parts.items():
                loss_sums[key] += float(value.detach()) * batch_count
            train_predictions.append(output["logits_c"].detach().cpu())
            train_labels.append(labels.detach().cpu())
            modality_routes.append(_numpy_route(output, "channel_weight", 3))
            relation_routes.append(_numpy_route(output, "relation_weight", 4))

            is_last = batch_index + 1 == num_batches
            if (batch_index + 1) % cli_args.grad_accum_steps == 0 or is_last:
                gradient_info = gradient_diagnostics(model)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        ensure_finite(f"gradient {name}", parameter.grad)
                if cli_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cli_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_pred = torch.cat(train_predictions)
        train_true = torch.cat(train_labels)
        train_main_mae = float(F.l1_loss(train_pred, train_true))
        modality_train = np.concatenate(modality_routes)
        relation_train = np.concatenate(relation_routes)
        modality_mean, modality_std = column_nan_stats(modality_train)
        relation_mean, relation_std = column_nan_stats(relation_train)
        val, val_routes, _ = evaluate(model, loaders["valid"], device, metrics_fn, args)
        test, test_routes, _ = evaluate(model, loaders["test"], device, metrics_fn, args)
        scheduler.step(val["main_mae"])
        beta_value = float(torch.sigmoid(model.aism_scale_logit).detach().cpu())
        for group_name, grad_norm in gradient_info.items():
            if grad_norm == "none" or (isinstance(grad_norm, (int, float)) and grad_norm == 0.0):
                logger.warning("seed=%d epoch=%d monitored gradient is missing/zero: %s=%s",
                               seed, epoch, group_name, grad_norm)

        row = {
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": loss_sums["total"] / seen,
            "train_main_mae": train_main_mae,
            "train_main_loss": loss_sums["main"] / seen,
            "train_nll": loss_sums["nll"] / seen,
            "train_pred_mae": loss_sums["pred_mae"] / seen,
            "train_uni_loss": loss_sums["uni"] / seen,
            "train_entropy_loss": loss_sums["entropy"] / seen,
            "train_router_loss": loss_sums["router"] / seen,
            "val_main_mae": val["main_mae"],
            "val_nll": val.get("nll", float("nan")),
            "val_loss": val["Loss"],
            "val_acc2": val["Acc_2"],
            "val_f1": val["F1_score"],
            "val_acc7": val["Acc_7"],
            "test_main_mae": test["main_mae"],
            "test_nll": test.get("nll", float("nan")),
            "component_spread": val.get("component_spread", float("nan")),
            "component_mae_ta": val.get("component_mae_ta", float("nan")),
            "component_mae_tv": val.get("component_mae_tv", float("nan")),
            "component_mae_av": val.get("component_mae_av", float("nan")),
            "component_mae_tav": val.get("component_mae_tav", float("nan")),
            "prior_entropy": val.get("prior_entropy", float("nan")),
            "posterior_entropy": val.get("posterior_entropy", float("nan")),
            "bayes_scale": val_routes.get("bayes_scale"),
            "aism_temperature": val_routes.get("temperature"),
            "test_loss": test["Loss"],
            "test_acc2": test["Acc_2"],
            "test_f1": test["F1_score"],
            "test_acc7": test["Acc_7"],
            "beta": beta_value,
            "grad_aism_encoder": gradient_info["aism_encoder"],
            "grad_aism_router": gradient_info["aism_router"],
            "grad_aism_adapter": gradient_info["aism_adapter"],
            "grad_aism_scale": gradient_info["aism_scale"],
            "lr_bert": next((group["lr"] for group in optimizer.param_groups if group.get("group_name") == "bert"), 0.0),
            "lr_backbone": next((group["lr"] for group in optimizer.param_groups if group.get("group_name") == "backbone"), 0.0),
            "lr_aism": next((group["lr"] for group in optimizer.param_groups if group.get("group_name") == "aism"), 0.0),
        }
        row.update(_route_csv_values({
            "modality_mean": modality_mean,
            "modality_std": modality_std,
            "relation_mean": relation_mean,
            "relation_std": relation_std,
        }, "modality", ("t", "v", "a")))
        row.update(_route_csv_values({
            "modality_mean": modality_mean,
            "modality_std": modality_std,
            "relation_mean": relation_mean,
            "relation_std": relation_std,
        }, "relation", ("ta", "tv", "av", "tav")))
        row["val_relation_mean"] = json.dumps(val_routes["relation_mean"])
        row["val_relation_std"] = json.dumps(val_routes["relation_std"])
        row["val_prior_mean"] = json.dumps(val_routes.get("prior_mean"))
        row["val_prior_std"] = json.dumps(val_routes.get("prior_std"))
        row["val_posterior_mean"] = json.dumps(val_routes.get("posterior_mean"))
        row["val_posterior_std"] = json.dumps(val_routes.get("posterior_std"))
        row["val_prior_argmax_frequency"] = json.dumps(val_routes.get("prior_argmax_frequency"))
        row["val_posterior_argmax_frequency"] = json.dumps(val_routes.get("posterior_argmax_frequency"))
        row["test_relation_mean"] = json.dumps(test_routes["relation_mean"])
        row["test_relation_std"] = json.dumps(test_routes["relation_std"])
        history.append(row)
        write_csv(seed_dir / "train_history.csv", history)
        logger.info(
            "seed=%d epoch=%d TEST MAE=%.5f ACC2=%.4f F1=%.4f ACC7=%.4f",
            seed, epoch, test["main_mae"], test["Acc_2"], test["F1_score"], test["Acc_7"],
        )
        logger.info(
            "seed=%d epoch=%d train_total_loss=%.5f train_main_mae=%.5f "
            "val_loss=%.4f val_mae=%.5f ACC2=%.4f F1=%.4f ACC7=%.4f beta=%s "
            "route_modality_mean=%s route_modality_std=%s "
            "route_relation_mean=%s route_relation_std=%s grad=%s",
            seed, epoch, row["train_total_loss"], train_main_mae, val["Loss"], val["main_mae"],
            val["Acc_2"], val["F1_score"], val["Acc_7"],
            f"{beta_value:.4f}" if np.isfinite(beta_value) else "n/a",
            np.round(modality_mean, 4).tolist(), np.round(modality_std, 4).tolist(),
            np.round(relation_mean, 4).tolist(), np.round(relation_std, 4).tolist(), gradient_info,
        )

        if val["main_mae"] < best_mae - 1e-8:
            best_mae = val["main_mae"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch,
                        "val_metrics": val, "config": vars(cli_args)}, seed_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= cli_args.early_stop:
            logger.info("seed=%d early stopping at epoch=%d (best epoch=%d)", seed, epoch, best_epoch)
            break

    del model, optimizer, scheduler, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return evaluate_best_seed(cli_args, seed, seed_dir, device, logger)


def evaluate_best_seed(cli_args, seed, seed_dir, device, logger):
    """Evaluate/export an already saved best checkpoint without retraining."""
    seed_dir = Path(seed_dir)
    checkpoint_path = seed_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint_path}")
    args = make_config(cli_args, seed)
    args.device = device
    loaders = make_loaders(args, seed)
    model = build_model(args).to(device)
    metrics_fn = MetricsTop("regression").getMetics(cli_args.dataset)
    total_params, aism_params = count_trainable(model)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    test, test_routes, test_samples = evaluate(
        model, loaders["test"], device, metrics_fn, args, collect_rows=True
    )
    test_metrics = {
        "seed": seed,
        "best_epoch": checkpoint["epoch"],
        "best_val_mae": checkpoint["val_metrics"]["main_mae"],
        "best_val_loss": checkpoint["val_metrics"].get("Loss"),
        "test_metrics": test,
        "test_routing": test_routes,
        "trainable_params": total_params,
        "aism_added_params": aism_params,
        "best_checkpoint": str(checkpoint_path),
    }
    with (seed_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(test_metrics, stream, ensure_ascii=False, indent=2, allow_nan=True)
    write_sample_exports(seed_dir, test_samples)
    with (seed_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump({**vars(cli_args), "seed": seed, "resolved_feature_path": args.featurePath,
                   "resolved_pretrained": args.pretrained,
                   "fusion_method": args.fusion_method,
                   "need_data_aligned": args.need_data_aligned,
                   "modality_router_order": ["T", "V", "A"],
                   "relation_order": ["TA", "TV", "AV", "TAV"],
                   },
                  stream, ensure_ascii=False, indent=2)
    logger.info("seed=%d best_epoch=%d TEST metrics=%s routes=%s",
                seed, checkpoint["epoch"], test, test_routes)
    del model, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_metrics


def _scalar_test_metrics(result):
    metrics = result.get("test_metrics", {})
    # Accept the legacy accidentally nested shape while keeping the current
    # flat evaluate_best_seed output canonical.
    if isinstance(metrics, dict) and isinstance(metrics.get("test_metrics"), dict):
        metrics = metrics["test_metrics"]
    return {key: value for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))}


def summarize_seed_results(all_results):
    """Aggregate only scalar test metrics across independent seeds."""
    if not all_results:
        raise ValueError("Cannot summarize an empty seed result list")
    metric_names = sorted(_scalar_test_metrics(all_results[0]))
    summary = {
        key: {
            "mean": float(np.mean([
                _scalar_test_metrics(result)[key] for result in all_results
            ])),
            "std": float(np.std([
                _scalar_test_metrics(result)[key] for result in all_results
            ], ddof=1)) if len(all_results) > 1 else 0.0,
        }
        for key in metric_names
    }
    validation_values = [result["best_val_mae"] for result in all_results]
    summary_rows = []
    for result in all_results:
        row = {
            "seed": result["seed"],
            "best_epoch": result["best_epoch"],
            "best_val_mae": result["best_val_mae"],
            "trainable_params": result["trainable_params"],
            "aism_added_params": result["aism_added_params"],
        }
        row.update(_scalar_test_metrics(result))
        summary_rows.append(row)
    summary_rows.append({
        "seed": "mean±std",
        "best_epoch": "",
        "best_val_mae": f"{np.mean(validation_values):.6f}±{np.std(validation_values, ddof=1) if len(validation_values) > 1 else 0.0:.6f}",
        "trainable_params": all_results[0]["trainable_params"],
        "aism_added_params": all_results[0]["aism_added_params"],
        **{key: f"{values['mean']:.6f}±{values['std']:.6f}"
           for key, values in summary.items()},
    })
    return summary, summary_rows


def configure_logging(run_dir):
    logger = logging.getLogger("rimoe")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _resume_config_mismatches(saved, cli_args, feature_path):
    """Return result-affecting differences before reusing a run directory."""
    expected = {
        "model": cli_args.model,
        "dataset": cli_args.dataset,
        "feature_path": str(feature_path),
        "pretrained": cli_args.pretrained,
        "alignment": cli_args.alignment,
        "fusion_method": cli_args.fusion_method,
        "aism_dim": cli_args.aism_dim,
        "aism_num_heads": cli_args.aism_num_heads,
        "aism_num_layers": cli_args.aism_num_layers,
        "aism_dropout": cli_args.aism_dropout,
        "aism_temperature": cli_args.aism_temperature,
        "aism_router_hidden": cli_args.aism_router_hidden,
        "bayes_scale": cli_args.bayes_scale,
        "bayes_objective": cli_args.bayes_objective,
        "lambda_uni": cli_args.lambda_uni,
        "lambda_ent": cli_args.lambda_ent,
        "lambda_router": cli_args.lambda_router,
    }
    mismatches = []
    saved_feature = saved.get("resolved_feature_path", saved.get("feature_path"))
    if saved_feature is not None and str(Path(saved_feature).expanduser()) != expected["feature_path"]:
        mismatches.append(f"feature_path: saved={saved_feature!r}, requested={expected['feature_path']!r}")
    for key, value in expected.items():
        if key == "feature_path":
            continue
        if key in saved and saved[key] != value:
            mismatches.append(f"{key}: saved={saved[key]!r}, requested={value!r}")
    return mismatches


def main(argv=None):
    cli_args = parse_args(argv)
    if not cli_args.seeds or len(cli_args.seeds) != len(set(cli_args.seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if cli_args.epochs <= 0 or cli_args.early_stop <= 0 or cli_args.grad_accum_steps <= 0:
        raise ValueError("epochs, early-stop and grad-accum-steps must be positive")
    if cli_args.batch_size <= 0 or cli_args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    if cli_args.aism_dim <= 0 or cli_args.aism_num_heads <= 0 or cli_args.aism_dim % cli_args.aism_num_heads:
        raise ValueError("aism-dim must be positive and divisible by aism-num-heads")
    if cli_args.bayes_scale <= 0 or not np.isfinite(cli_args.bayes_scale):
        raise ValueError("bayes-scale must be a finite positive scalar")
    feature_path = Path(cli_args.feature_path).expanduser().resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {feature_path}")
    pretrained_path = Path(cli_args.pretrained).expanduser()
    if pretrained_path.is_dir():
        cli_args.pretrained = str(pretrained_path.resolve())
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    elif (pretrained_path.is_absolute() or
          cli_args.pretrained.startswith(("./", "../", "~/", ".\\", "..\\"))):
        raise FileNotFoundError(f"Pretrained model directory does not exist: {pretrained_path}")

    if cli_args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif cli_args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if cli_args.resume_existing:
        run_dir = Path(cli_args.resume_existing).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Existing run directory does not exist: {run_dir}")
        existing_config = run_dir / "config.json"
        if existing_config.is_file():
            with existing_config.open("r", encoding="utf-8") as stream:
                saved_config = json.load(stream)
            mismatches = _resume_config_mismatches(saved_config, cli_args, feature_path)
            if mismatches:
                raise ValueError("Cannot resume with incompatible configuration: " + "; ".join(mismatches))
    else:
        output_root = Path(cli_args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        experiment_name = cli_args.experiment_name or (
            f"{cli_args.model}_{cli_args.dataset}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        if Path(experiment_name).name != experiment_name or experiment_name in (".", ".."):
            raise ValueError("experiment-name must be one directory name")
        run_dir = output_root / experiment_name
        run_dir.mkdir(parents=True, exist_ok=False)
    if not (run_dir / "config.json").is_file():
        with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump({**vars(cli_args), "resolved_feature_path": str(feature_path),
                       "resolved_pretrained": cli_args.pretrained,
                       "device": str(device), "seeds": cli_args.seeds},
                      stream, ensure_ascii=False, indent=2)
    logger = configure_logging(run_dir)
    logger.info("device=%s model=%s dataset=%s data=%s", device, cli_args.model,
                cli_args.dataset, feature_path)

    all_results = []
    for seed in cli_args.seeds:
        seed_dir = run_dir / f"seed_{seed}"
        best_checkpoint = seed_dir / "best.pt"
        metrics_path = seed_dir / "test_metrics.json"
        exports_exist = (seed_dir / "test_predictions.csv").is_file() and \
            (seed_dir / "test_routing.csv").is_file() and \
            (seed_dir / "config.json").is_file()
        if best_checkpoint.is_file() and metrics_path.is_file() and exports_exist:
            with metrics_path.open("r", encoding="utf-8") as stream:
                result = json.load(stream)
            logger.info("seed=%d already complete; reusing its saved test results", seed)
            all_results.append(result)
        elif best_checkpoint.is_file():
            logger.info("seed=%d checkpoint found; evaluating/exporting it without retraining", seed)
            all_results.append(evaluate_best_seed(cli_args, seed, seed_dir, device, logger))
        else:
            all_results.append(train_seed(cli_args, seed, run_dir, device, logger))

    summary, summary_rows = summarize_seed_results(all_results)
    write_csv(run_dir / "summary.csv", summary_rows)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"seeds": cli_args.seeds, "summary": summary,
                   "results": all_results}, stream, ensure_ascii=False, indent=2, allow_nan=True)
    logger.info("Finished %d independent seeds. Summary=%s output=%s",
                len(all_results), summary, run_dir)


if __name__ == "__main__":
    main()
