"""CPU regression tests using synthetic features and no pretrained text model."""
import csv
import json
import logging
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rimoe.config import get_config_regression
from rimoe.data import MMDataset
from rimoe.interaction_structure import AdaptiveInteractionStructure
from rimoe.model import RIMoE
from rimoe.training import (
    _resume_config_mismatches, compute_loss, evaluate_best_seed,
    make_config, make_optimizer, parse_args, summarize_seed_results, train_seed,
)


torch.set_num_threads(1)


def model_args(fusion="sum", aligned=True):
    args = get_config_regression("rimoe", "mosi")
    args.use_bert = False
    args.feature_dims = [16, 4, 8]
    args.dst_feature_dim_nheads = [16, 4]
    args.nlevels = 1
    args.need_data_aligned = aligned
    args.fusion_method = fusion
    for name in ("attn_dropout", "attn_dropout_a", "attn_dropout_v", "relu_dropout",
                 "embed_dropout", "res_dropout", "output_dropout", "text_dropout"):
        setattr(args, name, 0.0)
    args.aism_dim = 16
    args.aism_num_heads = 4
    args.aism_num_layers = 1
    args.aism_dropout = 0.0
    args.aism_router_hidden = 16
    args.aism_temperature = 0.7
    args.bayes_scale = 1.0
    return args


def inputs(batch=2, aligned=True):
    return (torch.randn(batch, 50, 16),
            torch.randn(batch, 50 if aligned else 375, 4),
            torch.randn(batch, 50 if aligned else 500, 8))


class TestRIMoE(unittest.TestCase):
    def test_shapes_mixture_and_checkpoint_roundtrip(self):
        for fusion in ("sum", "concat"):
            for batch in (1, 3):
                with self.subTest(fusion=fusion, batch=batch):
                    args = model_args(fusion)
                    model = RIMoE(args).eval()
                    values = inputs(batch)
                    with torch.no_grad():
                        output = model(*values)
                    self.assertEqual(output["logits_c"].shape, (batch, 1))
                    self.assertEqual(output["component_means"].shape, (batch, 4))
                    self.assertEqual(output["relation_features"].shape, (batch, 4, 16))
                    self.assertTrue(torch.allclose(output["structure_prior"].sum(1), torch.ones(batch)))
                    expected = (output["structure_prior"] * output["component_means"]).sum(1, keepdim=True)
                    self.assertTrue(torch.allclose(output["logits_c"], expected))
                    clone = RIMoE(args).eval()
                    clone.load_state_dict(model.state_dict(), strict=True)
                    with torch.no_grad():
                        self.assertTrue(torch.equal(clone(*values)["logits_c"], output["logits_c"]))

    def test_unaligned_forward(self):
        model = RIMoE(model_args(aligned=False)).eval()
        with torch.no_grad():
            output = model(*inputs(aligned=False))
        self.assertEqual(output["component_means"].shape, (2, 4))

    def test_training_reaches_interaction_and_backbone_parameters(self):
        for objective in ("nll", "mean_mae"):
            with self.subTest(objective=objective):
                args = model_args()
                model = RIMoE(args).train()
                cli = SimpleNamespace(bayes_objective=objective, lambda_uni=1.0,
                                      lambda_ent=0.1, lambda_router=0.01)
                output = model(*inputs())
                labels = torch.tensor([[-0.7], [0.8]])
                loss, parts = compute_loss(output, labels, args, cli)
                expected = (parts["main"] + parts["uni"] + 0.1 * parts["entropy"] +
                            0.01 * parts["router"])
                self.assertTrue(torch.allclose(loss, expected))
                self.assertTrue(torch.allclose(parts["main"], parts[objective if objective == "nll" else "pred_mae"]))
                loss.backward()
                parameters = dict(model.named_parameters())
                for name in ("aism.encoder.layers.0.self_attn.in_proj_weight",
                             "aism.structure_router.0.weight", "aism_adapter.1.weight",
                             "aism_scale_logit", "self_attentions_l.layers.0.self_attn.in_proj_weight"):
                    gradient = parameters[name].grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_pair_structures_exclude_absent_modality(self):
        module = AdaptiveInteractionStructure(dim=16, num_heads=4, dropout=0.0).eval()
        values = [torch.randn(2, 16) for _ in range(3)]
        with torch.no_grad():
            base = module(*values)["relation_features"]
            for structure, absent in ((0, 2), (1, 1), (2, 0)):
                changed = list(values)
                changed[absent] = changed[absent] + torch.randn_like(changed[absent])
                output = module(*changed)["relation_features"]
                self.assertTrue(torch.equal(base[:, structure], output[:, structure]))

    def test_invalid_scale(self):
        for scale in (0, -1, float("nan"), float("inf")):
            args = model_args()
            args.bayes_scale = scale
            with self.assertRaises(ValueError):
                RIMoE(args)

    def test_cli_and_dataset_defaults(self):
        cli = parse_args(["--feature-path", "dummy.pkl"])
        self.assertEqual(cli.model, "rimoe")
        self.assertEqual(cli.pretrained, "bert-base-uncased")
        self.assertEqual(cli.seeds, [1111, 1112, 1113, 1114, 1115])
        for dataset, dims in (("mosi", [768, 5, 20]), ("mosei", [768, 74, 35])):
            cli.dataset = dataset
            args = make_config(cli, 1111)
            self.assertEqual(args.feature_dims, dims)
            self.assertEqual(args.num_workers, cli.num_workers)

    def test_optimizer_groups(self):
        model = RIMoE(model_args())
        cli = SimpleNamespace(bert_lr=2e-5, backbone_lr=1e-4, aism_lr=3e-4)
        optimizer = make_optimizer(model, model.args, cli)
        self.assertEqual({g["group_name"]: g["lr"] for g in optimizer.param_groups},
                         {"backbone": 1e-4, "aism": 3e-4})
        parameter_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(set(parameter_ids), {id(p) for p in model.parameters() if p.requires_grad})

    def test_resume_rejects_changed_model_settings(self):
        cli = parse_args(["--feature-path", "dummy.pkl"])
        saved = vars(cli).copy()
        saved["bayes_scale"] = 2.0
        errors = _resume_config_mismatches(saved, cli, Path(cli.feature_path).resolve())
        self.assertTrue(any("bayes_scale" in error for error in errors))

    def test_seed_summary(self):
        results = [{"seed": seed, "best_epoch": 1, "best_val_mae": mae,
                    "trainable_params": 100, "aism_added_params": 25,
                    "test_metrics": {"main_mae": mae}} for seed, mae in ((1, 0.5), (2, 0.7))]
        summary, rows = summarize_seed_results(results)
        self.assertAlmostEqual(summary["main_mae"]["mean"], 0.6)
        self.assertAlmostEqual(summary["main_mae"]["std"], 0.141421356, places=6)
        self.assertEqual(len(rows), 3)

    def test_train_evaluate_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "synthetic.pkl"
            arrays = [value.numpy() for value in inputs(3)]
            split = {"text": arrays[0], "audio": arrays[1], "vision": arrays[2],
                     "raw_text": ["a", "b", "c"], "id": ["id-a", "id-b", "id-c"],
                     "regression_labels": np.array([-1.0, 0.5, 1.0], dtype=np.float32)}
            with feature_path.open("wb") as stream:
                pickle.dump({key: split for key in ("train", "valid", "test")}, stream)
            cli = parse_args(["--feature-path", str(feature_path), "--seeds", "1111",
                              "--epochs", "1", "--batch-size", "2"])
            args = model_args()
            args.featurePath = str(feature_path)
            args.batch_size = 2
            args.num_workers = 0
            dataset = MMDataset(args)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0]["id"], "id-a")
            with patch("rimoe.training.make_config", return_value=args):
                logger = logging.getLogger("test-rimoe")
                result = train_seed(cli, 1111, root, torch.device("cpu"), logger)
                seed_dir = root / "seed_1111"
                again = evaluate_best_seed(cli, 1111, seed_dir, torch.device("cpu"), logger)
            self.assertAlmostEqual(result["test_metrics"]["main_mae"], again["test_metrics"]["main_mae"], places=6)
            self.assertTrue((seed_dir / "best.pt").is_file())
            with (seed_dir / "test_routing.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["id"] for row in rows}, set(split["id"]))
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertAlmostEqual(sum(float(row[f"prior_{key}"]) for key in ("ta", "tv", "av", "tav")), 1.0, places=5)
                self.assertAlmostEqual(float(row["beta"]), result["test_routing"]["beta"], places=6)
            with (seed_dir / "config.json").open(encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["relation_order"], ["TA", "TV", "AV", "TAV"])


if __name__ == "__main__":
    unittest.main()
