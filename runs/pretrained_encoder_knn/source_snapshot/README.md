# Residual View for DCASE2026 Task 2

## ASD3DP Run 21

- [Run 21 protocol and results](docs/RUN21_REPORT.md)
- [Run 21 configuration](config/run21_asd3dp.yaml)

Minimal reproduction code for the DCASE2026 Task 2 submitted systems.

Paper link will be added later. This repository keeps only the runnable code
and the main settings used for the reported development-set scores.

## Setup

Install the minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

If you use conda follow:

```bash
conda create -n residual_view python=3.11 -y
conda activate residual_view

"$CONDA_PREFIX/bin/python" -m pip install -r requirements.txt
```

Prepare the following external files locally:

- DCASE2026 Task 2 dataset
- pretrained audio encoder checkpoints
- upstream encoder code/checkpoints required by BEATs, SSLAM, and DaSheng
  - SSLAM: https://github.com/ta012/SSLAM/tree/main
  - BEATs: https://github.com/microsoft/unilm/tree/master/beats
  - DaSheng: https://github.com/XiaoMi/dasheng

Set dataset paths in:

```text
config/data_config_2026.yaml
```

The example scripts assume these local checkpoint roots:

```text
./transformer-ssl-asd/sslam
./transformer-ssl-asd/beats
```

DaSheng uses the package default checkpoint.

## Run

Generate embedding caches and residual-view scores for the final encoder
set:

```bash
bash scripts/run_residual_view_dev.sh
```

or

```bash
bash scripts/run_residual_view_eval.sh
```

The script first generates near/far pair embedding caches using
the residual-view configuration. PRPS is not applied during encoder forward
passes. Instead, `run_residual_view.py --config ...` loads the cached pair
embeddings, constructs the residual view and the projection key, performs
prototype selection inside the cached embedding space, and writes submission
folders for eval configs.

Run a specific system config directly:

```bash
python run_residual_view.py --config config/system1_eval.yaml
python run_residual_view.py --config config/system2_eval.yaml
```

## Main Settings

| setting | value |
|:--|:--|
| audio crop | fixed 10 s |
| residual coefficient | `alpha = 0.5` |
| MemMix coefficient | `lambda = 0.9` |
| MemMix support | `990` |
| kNN | `K = 1` |
| PRPS prototypes | `128` |
| decision threshold | train-normal 95 percentile |
| domain normalization | off |

## License

The code in this repository is released under the MIT License. External
pretrained encoders, datasets, and upstream repositories are not vendored here
and remain under their own licenses.

## Acknowledgements

This repository builds on the GenRep-style frozen embedding and memory-bank
pipeline. We thank the authors of
[GenRepASD](https://github.com/Phuriches/GenRepASD) for the framework and
implementation direction.

We also thank the authors of the upstream audio encoders used by these systems:

- [SSLAM](https://github.com/ta012/SSLAM/tree/main)
- [BEATs](https://github.com/microsoft/unilm/tree/master/beats)
- [DaSheng](https://github.com/XiaoMi/dasheng)

## Citation
