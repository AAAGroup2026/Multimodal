# RIMoE

Bayesian inference over multimodal interaction structures for sentiment
regression on CMU-MOSI and CMU-MOSEI.

This release contains the Bayesian model only. A modality backbone encodes text
(T), audio (A), and vision (V). A shared interaction encoder builds TA, TV, AV,
and TAV experts. The model predicts a weighted mean of four expert outputs using
a feature-conditioned prior. Labels are used for the Laplace mixture training
objective and posterior diagnostics; inference uses the prior alone.

## Setup

Use Python 3.9 or newer. Install PyTorch for your CUDA environment, then install
the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Dependencies are bounded for compatibility; this file is not an environment lock.

## Data

Prepare a trusted preprocessed MOSI or MOSEI pickle file with `train`, `valid`,
and `test` dictionaries. Each split must contain:

| Field | Shape / contents |
| --- | --- |
| `text_bert` | `[N, 3, 50]`: token IDs, attention mask, segment IDs |
| `audio` | `[N, L_a, D_a]` |
| `vision` | `[N, L_v, D_v]` |
| `regression_labels` | `[N]` or `[N, 1]` sentiment scores |
| `raw_text` | `N` strings |
| `id` | `N` sample identifiers |
| `audio_lengths`, `vision_lengths` | `N` lengths, required for unaligned data |

MOSI uses audio/vision dimensions 5/20; MOSEI uses 74/35. Aligned sequences have
50 steps. Unaligned audio/vision lengths are 375/500 for MOSI and 500/500 for
MOSEI. Text is encoded with a 768-dimensional BERT model.

Datasets and pretrained weights are not included. `--pretrained` accepts a
Hugging Face model ID (default: `bert-base-uncased`) or an existing local model
directory. A local directory enables offline loading.

## Train

Run from the repository directory:

```bash
python train.py --dataset mosi --feature-path /path/to/MOSI/aligned_50.pkl --output-root outputs --experiment-name mosi_bayesian
python train.py --dataset mosei --feature-path /path/to/MOSEI/aligned_50.pkl --output-root outputs --experiment-name mosei_bayesian
```

The defaults use five seeds: 1111, 1112, 1113, 1114, 1115. For one seed, add
`--seeds 1111`. Use `--alignment unaligned` for unaligned features and
`--device cpu` for CPU execution. Run `python train.py --help` for all options.

The default objective is Laplace mixture NLL plus unimodal MAE, modality entropy,
and modality router supervision. Unimodal distillation is disabled in this
Bayesian model. `--bayes-objective mean_mae` retains the mixture architecture
while replacing its primary loss with prediction MAE.

Model selection, early stopping, and learning-rate scheduling use validation
MAE. Test metrics recorded during training are diagnostic and do not select
checkpoints. The `Loss` metric is a mean of batch MAEs; `main_mae` is the MAE over
all samples. Existing split shuffling and optimization defaults are preserved.

## Outputs

Each experiment writes `config.json`, `training.log`, `summary.csv`, and
`summary.json`. Each `seed_<seed>/` directory contains:

- `best.pt`: model state, epoch, validation metrics, and CLI configuration.
- `train_history.csv`: losses, metrics, routing statistics, and gradient norms.
- `test_metrics.json`: test metrics from the best validation checkpoint.
- `test_predictions.csv`: sample IDs, targets, and predictions.
- `test_routing.csv`: modality weights, structure priors/posteriors, and expert means.
- `config.json`: resolved per-seed configuration.

Modality weights use **T, V, A** order. Structure weights use **TA, TV, AV, TAV**
order. Posterior columns depend on ground truth and are for analysis only.

To reuse completed seeds or evaluate an existing best checkpoint, run the same
command and settings with `--resume-existing outputs/mosi_bayesian`. A seed
without a best checkpoint starts training again; optimizer state is not resumed.
The `aism_*` parameter keys are retained for compatibility with earlier Bayesian
state dictionaries. Old command-line model names and run configurations must be
migrated to the current RIMoE entry point.

## Layout

```text
train.py                       Training / saved-checkpoint entry point
rimoe/
  model.py                     Bayesian mixture model
  backbone.py                  Shared modality backbone
  interaction_structure.py     TA / TV / AV / TAV encoder and router
  bayesian_structure.py        Laplace likelihood and posterior
  training.py                  Training, evaluation, and exports
  data.py                      Dataset loading
  config.py                    Configuration loading
  configs/default.json         MOSI / MOSEI defaults
  encoders/                    BERT and Transformer components
  losses.py                    Auxiliary routing losses
  metrics.py                   Sentiment metrics
  router.py                    Modality router
tests/                         CPU regression and integration tests
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic tensors and data, with downloading BERT or datasets.

## Acknowledgments

The modality backbone and supporting encoder/data/metric code were adapted from
**EMOE: Modality-Specific Enhanced Dynamic Emotion Experts**, by Yiyang Fang,
Wenke Huang, Guancheng Wan, Kehua Su, and Mang Ye (CVPR 2025). The Transformer
components also follow the Multimodal Transformer implementation. See
[THIRD_PARTY.md](THIRD_PARTY.md) for provenance and licensing status.

The publication title, author list, citation, and project license for RIMoE
should be added by the maintainers when finalized.
