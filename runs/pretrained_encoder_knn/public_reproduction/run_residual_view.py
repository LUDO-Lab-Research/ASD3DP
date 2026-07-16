import pickle
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from scipy.stats import hmean
from src.datasets.prepare_dcase2026 import get_dcase2026
from src.datasets.audio_dataset import AudioDataset
from src.encoders import build_feature_extractor
from src.residual_view.input_type import INPUT_TYPE_NAMES, expand_input_type
from src.residual_view.differet_view import (
    DIFFERENT_VIEW_NAMES,
    PAIR_INPUT_TYPES,
    SCALED_DIFFERENT_VIEW_NAMES,
    collapse_paired_sequence,
    make_different_view_features,
)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import argparse
import yaml


ROOT = Path(__file__).resolve().parent
PIPELINE_CHILD_ENV = "RESIDUAL_VIEW_PIPELINE_CHILD"


def load_yaml_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    residual_config = config.get("residual_view", {})
    if residual_config is not None and not isinstance(residual_config, dict):
        raise ValueError("residual_view must be a YAML mapping.")
    merged = dict(config)
    merged.update(residual_config or {})
    if "fixed_residual_alpha" not in merged and "residual_alpha" in merged:
        merged["fixed_residual_alpha"] = merged["residual_alpha"]
    if "alpha" not in merged and "memmix_alpha" in merged:
        merged["alpha"] = merged["memmix_alpha"]
    return merged


def add_config_defaults(parser, config):
    valid_keys = {
        action.dest
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    defaults = {
        key: value
        for key, value in config.items()
        if key in valid_keys
    }
    parser.set_defaults(**defaults)


def cache_models_from_config(config):
    cache_models = config.get("cache_models")
    if cache_models is None:
        return []
    if not isinstance(cache_models, list) or not cache_models:
        raise ValueError("cache_models must be a non-empty list.")

    parsed = []
    for index, item in enumerate(cache_models):
        if not isinstance(item, dict):
            raise ValueError(f"cache_models[{index}] must be a mapping.")
        parsed.append(
            {
                "model_name": str(item["model_name"]),
                "pretrained_model_dir": str(item["pretrained_model_dir"]),
            }
        )
    return parsed


def run_pipeline_from_config(config_path, config):
    config_path = str(Path(config_path).resolve())
    env = dict(os.environ)
    env[PIPELINE_CHILD_ENV] = "1"

    for model in cache_models_from_config(config):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                config_path,
                "--model_name",
                model["model_name"],
                "--pretrained_model_dir",
                model["pretrained_model_dir"],
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "run_projection_select.py",
            "--config",
            config_path,
        ],
        cwd=ROOT,
        check=True,
    )


config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument("--config", type=str, default=None)
config_args, remaining_argv = config_parser.parse_known_args()
config_defaults = load_yaml_config(config_args.config)

if (
    config_args.config is not None
    and os.environ.get(PIPELINE_CHILD_ENV) != "1"
    and "--model_name" not in remaining_argv
    and "-h" not in remaining_argv
    and "--help" not in remaining_argv
    and cache_models_from_config(config_defaults)
):
    run_pipeline_from_config(config_args.config, config_defaults)
    sys.exit(0)

parser = argparse.ArgumentParser(parents=[config_parser])
parser.add_argument('--model_name', type=str, default="beats", help='ast or beats')
parser.add_argument('--train_pct', type=float, default=1, help='path to save results')
parser.add_argument('--pretrained_model_dir', type=str, default="./transformer-ssl-asd/beats", help='path to saved models')
parser.add_argument('--dataset_name', type=str, default='dcase2026')
parser.add_argument('--eval_split', type=str, default="valid", help='Include valid set')
parser.add_argument('--train_split', type=str, default="train", help='train')
parser.add_argument('--top_k', type=int, default=1, help='Top k')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--temporal_pooling', action="store_true", default=False, help='Temporal pooling or not')
parser.add_argument('--spectral_pooling', action="store_true", default=False, help='Spectral pooling or not')
parser.add_argument('--audio_length', type=int, default=160000, help='Audio length')
parser.add_argument('--save_path', type=str, default='train_features', help='path to save results')
parser.add_argument('--n_mix_support', type=int, default=None, help='Number of support samples to mix')
parser.add_argument('--alpha', type=float, default=0.90, help='Alpha value for mixup')
parser.add_argument('--save_official', action="store_true", default=False, help='Save official submission files')
parser.add_argument('--channel_index', type=int, default=0, help='0=near, 1=far')
parser.add_argument(
    '--machine_names',
    type=str,
    default='all',
    help='Comma-separated machine names to run, or all.',
)
parser.add_argument(
    '--input_type',
    type=str,
    default=None,
    choices=sorted(INPUT_TYPE_NAMES),
    help='Input channel. If omitted, channel_index is used.',
)
parser.add_argument(
    '--different_view',
    type=str,
    default=None,
    choices=sorted(DIFFERENT_VIEW_NAMES),
    help='Embedding-space near/far view.',
)
parser.add_argument(
    '--fixed_residual_alpha',
    type=float,
    default=None,
    help='Alpha for fixed_residual_view: near_embedding - alpha * far_embedding.',
)
parser.add_argument('--wandb_entity', type=str, default=os.environ.get('WANDB_ENTITY'))
parser.add_argument('--wandb_project', type=str, default=os.environ.get('WANDB_PROJECT', 'residual view'))
parser.add_argument('--wandb_name', type=str, default=None)
parser.add_argument('--use_wandb', action="store_true", default=False)
parser.add_argument('--no_wandb', action="store_true", default=False)
parser.add_argument('--cache_only', action="store_true", default=False)

add_config_defaults(parser, config_defaults)
args = parser.parse_args()
model_name = args.model_name
pt_model_dir = args.pretrained_model_dir
if model_name.lower() == "beats_ft1":
    single_best_layer = 4
else:
    single_best_layer = 5

channel_index=args.channel_index
if args.input_type is None:
    if channel_index == 0:
        input_type = "near"
    elif channel_index == 1:
        input_type = "far"
    else:
        raise ValueError(
            f"Unsupported channel_index={channel_index}. "
            "Use --input_type to select near or far input."
        )
else:
    input_type = args.input_type
dataset_name = args.dataset_name
train_split = args.train_split
top_k = args.top_k
batch_size = args.batch_size
pooling_feature = "temporal"
temporal_pooling = True
spectral_pooling = args.spectral_pooling
if spectral_pooling:
    pooling_feature = "spectral"
elif args.temporal_pooling:
    pooling_feature = "temporal"
eval_split = args.eval_split
audio_length = args.audio_length
n_mix_support = args.n_mix_support
alpha = args.alpha
save_official = args.save_official
selected_machine_names = [
    machine_name.strip()
    for machine_name in args.machine_names.split(',')
    if machine_name.strip()
]
if not selected_machine_names:
    raise ValueError("--machine_names must be 'all' or a comma-separated list.")
different_view = args.different_view
fixed_residual_alpha = args.fixed_residual_alpha
if different_view in SCALED_DIFFERENT_VIEW_NAMES and fixed_residual_alpha is None:
    raise ValueError(
        "--fixed_residual_alpha is required for scaled different_view modes: "
        f"{sorted(SCALED_DIFFERENT_VIEW_NAMES)}."
    )
if different_view not in SCALED_DIFFERENT_VIEW_NAMES and fixed_residual_alpha is not None:
    raise ValueError(
        "--fixed_residual_alpha is only valid with scaled different_view modes: "
        f"{sorted(SCALED_DIFFERENT_VIEW_NAMES)}."
    )

view_condition = f'_differentview{different_view}' if different_view else ''
fixed_residual_condition = (
    f'_fixedresidualalpha{fixed_residual_alpha}'
    if fixed_residual_alpha is not None else ''
)
crop_condition = ""
log_condition = f'DNASD_model_name{model_name}_input{input_type}{view_condition}{fixed_residual_condition}_topk{top_k}_trainsplit{train_split}_evalsplit{eval_split}_pooling{pooling_feature}_n_mix_support{n_mix_support}_alpha{alpha}'
save_dir = f"out/{dataset_name}_test/patch_diff_{model_name}/{log_condition}/log_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

wandb_enabled = args.use_wandb and not args.no_wandb
wandb_run = None
if wandb_enabled:
    import wandb
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        dir=save_dir,
    )
    wandb_run.summary['score_selection'] = 'raw_final_best_layer_wise'
    wandb_run.summary['score_mode'] = 'raw_knn_distance'
    wandb_run.summary['dev_selective_best_each_machine'] = 'diagnostic_only'
    wandb_run.summary['input_type'] = input_type
    wandb_run.summary['machine_names'] = args.machine_names
    wandb_run.summary['different_view'] = different_view
    wandb_run.summary['fixed_residual_alpha'] = fixed_residual_alpha

if dataset_name == "dcase2026":
    datasets = get_dcase2026(train_split)
else:
    raise ValueError(
        f"Unsupported dataset_name={dataset_name}. "
        "This public reproduction package includes only the DCASE2026 Task 2 loader."
    )
train_data = datasets['train']
if eval_split not in datasets:
    raise ValueError(
        f"dataset_name={dataset_name} has no eval_split={eval_split}. "
        "Check local test wavs and ground-truth availability."
    )
test_data = datasets[eval_split]

train_file_attrs = np.array(train_data['file_attrs'])
train_machine_names = np.array(train_data['machine_names'])

# ======== model ========
print(f'model name used: {model_name}')
feature_extractor = build_feature_extractor(model_name, pt_model_dir)
model = feature_extractor.model

print(sum([p.numel() for p in model.parameters() if p.requires_grad]))


# ======== distance matrix ========
# original
# def calc_dist_matrix(x, y):
#     """Calculate Euclidean distance matrix with torch.tensor"""
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#     dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
#     return dist_matrix

def calc_dist_matrix(x, y):
    """Efficient Euclidean distance matrix calculation using broadcasting"""
    x_norm = (x**2).sum(dim=1, keepdim=True)  # Shape: (N, 1)
    y_norm = (y**2).sum(dim=1, keepdim=True).T  # Shape: (1, M)
    dist_sq = x_norm + y_norm - 2 * x @ y.T
    dist_matrix = torch.sqrt(torch.clamp(dist_sq, min=0.0))  # Shape: (N, M)
    return dist_matrix

# ======== evaluation function ========
def eval_score(gt_list, scores):
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, scores)
    img_roc_auc = roc_auc_score(gt_list, scores)
    pauc = roc_auc_score(gt_list, scores, max_fpr=0.1)
    precision, recall, thresholds = precision_recall_curve(gt_list, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
    idx = np.argmax(f1_scores)
    return img_roc_auc, pauc, f1_scores[idx]


device = "cuda"
model.to(device)
model.eval()

df_log = pd.DataFrame()

from sklearn.preprocessing import LabelEncoder

# ======= Evaluation =======
machine_names = np.unique(train_data["machine_names"])
if selected_machine_names != ['all']:
    unknown_machine_names = sorted(set(selected_machine_names) - set(machine_names))
    if unknown_machine_names:
        raise ValueError(
            f"Unknown machine_names={unknown_machine_names}. "
            f"Available machine_names={machine_names.tolist()}."
        )
    machine_names = np.array(selected_machine_names)
print(f'machine names: {machine_names}')
train_input_types = expand_input_type(input_type, "train")
eval_input_types = expand_input_type(input_type, "eval")
if different_view is not None:
    train_input_types = PAIR_INPUT_TYPES
    eval_input_types = PAIR_INPUT_TYPES
historical_mono = dataset_name in {"dcase2023", "dcase2024", "dcase2025"}
if historical_mono and different_view is not None:
    raise ValueError("Historical DCASE mono eval does not support different_view.")
dataset_train_input_types = None if historical_mono else train_input_types
dataset_eval_input_types = None if historical_mono else eval_input_types
print(f'input_type: {input_type}')
print(f'different_view: {different_view}')
print(f'train input types: {train_input_types}')
print(f'eval input types: {eval_input_types}')
print(f'historical_mono: {historical_mono}')

for class_name in machine_names:

    train_dataset = AudioDataset(
        file_list=train_data["file_list"],
        label_list=train_data["label_list"],
        machine_list=train_data["machine_names"],
        source_list=train_data["source_list"],
        machine_name=class_name,
        audio_length=audio_length,
        channel_index=args.channel_index,
        input_types=dataset_train_input_types,
    )
    test_dataset = AudioDataset(
        file_list=test_data["file_list"],
        label_list=test_data["label_list"],
        machine_list=test_data["machine_names"],
        source_list=test_data["source_list"],
        machine_name=class_name,
        audio_length=audio_length,
        channel_index=args.channel_index,
        input_types=dataset_eval_input_types,
    )
    train_class_ids_machine = train_file_attrs[train_machine_names == class_name]
    all_attrs = np.concatenate([train_class_ids_machine])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_attrs)
    train_class_ids = label_encoder.transform(train_class_ids_machine)

    class_tensor = torch.from_numpy(train_class_ids)
    unique_class_tensor = torch.unique(class_tensor)
    # print(f'unique class tensor: {unique_class_tensor}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False)
    print(f'class name: {class_name}')
    train_source_list_for_scoring = train_dataset.source_list
    test_output_files = test_dataset.file_list
    if different_view is not None:
        train_source_list_for_scoring = collapse_paired_sequence(
            train_dataset.source_list
        )
        test_output_files = collapse_paired_sequence(test_dataset.file_list)

    train_source_mask_np = np.array(train_source_list_for_scoring) == 'source'
    train_source_indices_np = np.flatnonzero(train_source_mask_np)
    train_target_indices_np = np.flatnonzero(~train_source_mask_np)
    train_source_indices = torch.from_numpy(train_source_indices_np).to(device)
    train_target_indices = torch.from_numpy(train_target_indices_np).to(device)
    print(f"train source refs: {len(train_source_indices_np)}")
    print(f"train target refs: {len(train_target_indices_np)}")

    # extract train set features
    print(f'len train dataset: {len(train_dataset)}')
    train_feature_layers = []
    train_all_lasts = []

    # save the embeddings if None
    save_path = os.path.join(
        "cache_memory",
        args.save_path
        + f'_{model_name}_input{input_type}'
        + crop_condition
        + ('_differentview_pair' if different_view else ''),
    )
    save_dir_path = f"temp_{dataset_name}_{train_split}"
    os.makedirs(os.path.join(save_path, save_dir_path), exist_ok=True)
    train_feature_filepath = os.path.join(save_path, save_dir_path, f'train_{pooling_feature}_%s.pkl' % class_name)
    if not os.path.exists(train_feature_filepath):
        for batch in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            x = batch['input']
            x = x.to(device)

            # forward
            with torch.no_grad():
                train_feature_layers.append(feature_extractor.extract_batch(x, pooling_feature))

        torch.cuda.empty_cache()
        train_feature_layers = torch.cat(train_feature_layers, 1).flatten(2)
        print(f"train_all_lasts.size(): {train_feature_layers.size()}")
        # save extracted feature
        print(f'save train set feature to: {train_feature_filepath}')
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_feature_layers, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_feature_layers = pickle.load(f)
        print(f"train_all_lasts.size(): {train_feature_layers.size()}")

    if different_view is not None:
        train_feature_layers = make_different_view_features(
            train_feature_layers,
            different_view,
            fixed_residual_alpha,
        )
        print(f"different_view train features: {train_feature_layers.size()}")

    gt_list = []
    test_outputs = []
    test_feature_layers = []
    test_all_lasts = []

    save_dir_path = f"temp_{dataset_name}_{eval_split}"
    os.makedirs(os.path.join(save_path, save_dir_path), exist_ok=True)
    test_feature_filepath = os.path.join(save_path, save_dir_path, f'{eval_split}_{pooling_feature}_%s.pkl' % class_name)

    if not os.path.exists(test_feature_filepath):
        # extract test set features
        for batch in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            x = batch['input']
            x = x.to(device)
            y = batch['label']
            gt_list.extend(y.cpu().detach().numpy())

            # forward
            with torch.no_grad():
                test_feature_layers.append(feature_extractor.extract_batch(x, pooling_feature))

        torch.cuda.empty_cache()

        test_feature_layers = torch.cat(test_feature_layers, 1).flatten(2)
        print(f"test_all_lasts.size(): {test_feature_layers.size()}")
        # save extracted feature
        print(f'save test set feature to: {test_feature_filepath}')
        with open(test_feature_filepath, 'wb') as f:
            pickle.dump(test_feature_layers, f)
    else:
        print('load test set feature from: %s' % test_feature_filepath)
        with open(test_feature_filepath, 'rb') as f:
            test_feature_layers = pickle.load(f)
        print(f"train_all_lasts.size(): {test_feature_layers.size()}")
        # get ground truth
        for batch in test_dataloader:
            y = batch['label']
            gt_list.extend(y.cpu().detach().numpy())

    if different_view is not None:
        test_feature_layers = make_different_view_features(
            test_feature_layers,
            different_view,
            fixed_residual_alpha,
        )
        gt_list = collapse_paired_sequence(gt_list)
        print(f"different_view test features: {test_feature_layers.size()}")

    if args.cache_only:
        print(f"cache_only complete: class={class_name}")
        continue

    if eval_split in {'valid', 'test', 'eval'}:
        machine_list = np.array(test_data['machine_names'])
        source_list = np.array(test_data['source_list'])
        source_list_machine = source_list[machine_list == class_name].reshape(-1)
        source_list_machine = np.array(source_list_machine == 'source')
        gt_list = np.asarray(gt_list).reshape(-1)
    else:
        raise ValueError(f"Unsupported eval_split={eval_split}")

    has_eval_labels = set(np.unique(gt_list)).issubset({0, 1}) and (
        len(np.unique(gt_list)) == 2
    )
    if has_eval_labels:
        gt_source = gt_list[(source_list_machine == True) | (gt_list != 0)]
        gt_target = gt_list[(source_list_machine == False) | (gt_list != 0)]
    else:
        print(
            f"No binary labels for eval_split={eval_split}, "
            f"class={class_name}; saving anomaly scores only."
        )
        gt_source = np.array([])
        gt_target = np.array([])
    results_an = pd.DataFrame()
    results_an['output1'] = [f.split('/')[-1] for f in test_output_files]
    results_dec = pd.DataFrame()
    results_dec['output1'] = [f.split('/')[-1] for f in test_output_files]
    best_score = 0
    best_layer = None
    for num_layer, (train_all_lasts, test_all_lasts) in enumerate(zip(train_feature_layers, test_feature_layers)):
        print(f' layer {num_layer+1}')
        train_all_lasts = train_all_lasts.to(device, non_blocking=True)
        test_all_lasts = test_all_lasts.to(device, non_blocking=True)
        current_source_indices = train_source_indices
        current_target_indices = train_target_indices

        augmented_target_samples = []
        if n_mix_support is not None:
            source_train_features = train_all_lasts[current_source_indices]
            target_train_features = train_all_lasts[current_target_indices]

            # Compute distance matrix between target and source features
            ST_dist_matrix = calc_dist_matrix(
                torch.flatten(target_train_features, 1),
                torch.flatten(source_train_features, 1)
            )

            # Get top-k nearest source features for each target feature
            topk_values, topk_indexes = torch.topk(ST_dist_matrix, k=n_mix_support, dim=1, largest=False)

            # Perform mixup augmentation
            for i, topk_index in enumerate(topk_indexes):
                nearest_sources = source_train_features[topk_index]
                mixed_samples = alpha * target_train_features[i] + (1 - alpha) * nearest_sources
                augmented_target_samples.append(mixed_samples)

            # Stack augmented samples and update training features
            augmented_target_samples = torch.cat(augmented_target_samples, dim=0)
            print(f"augmented_target_samples: {augmented_target_samples.size()}")

            aug_start = train_all_lasts.shape[0]
            aug_stop = aug_start + augmented_target_samples.shape[0]
            augmented_indices = torch.arange(
                aug_start,
                aug_stop,
                device=device,
            )
            train_all_lasts = torch.cat([train_all_lasts, augmented_target_samples], dim=0)
            current_target_indices = torch.cat(
                [current_target_indices, augmented_indices],
                dim=0,
            )

        train_ref_features = torch.flatten(train_all_lasts, 1)
        dist_matrix = calc_dist_matrix(
            torch.flatten(test_all_lasts, 1),
            train_ref_features,
        )
        source_dist = dist_matrix[:, current_source_indices]
        target_dist = dist_matrix[:, current_target_indices]
        print(f"source_dist.size(): {source_dist.size()}")
        print(f"target_dist.size(): {target_dist.size()}")

        topk_value_source, topk_index_source = torch.topk(
            source_dist,
            k=top_k,
            dim=1,
            largest=False,
        )
        topk_value_target, topk_index_target = torch.topk(
            target_dist,
            k=top_k,
            dim=1,
            largest=False,
        )
        topk_index_source = current_source_indices[topk_index_source]
        topk_index_target = current_target_indices[topk_index_target]

        topk_indexes = torch.cat(
            [topk_index_source, topk_index_target],
            1,
        ).cpu()
        value_source = torch.mean(topk_value_source, 1).cpu().detach().numpy()
        value_target = torch.mean(topk_value_target, 1).cpu().detach().numpy()

        scores = np.minimum(value_source, value_target)
        min_indices = np.argmin(
            np.stack((value_source, value_target), axis=-1),
            axis=-1,
        )
        topk_indexes_raw = topk_indexes[
            np.arange(topk_indexes.shape[0]),
            min_indices,
        ]

        if has_eval_labels:
            # by including anomalous sample of different domain, we pick the same threshold
            score_source = scores[(source_list_machine == True) | (gt_list != 0)]
            score_target = scores[(source_list_machine == False) | (gt_list != 0)]

            auc_all, pauc_all, f1_all = eval_score(gt_list, scores)
            auc_source, pauc_source, f1_source = eval_score(gt_source, score_source)
            auc_target, pauc_target, f1_target = eval_score(gt_target, score_target)

            official_score = hmean([auc_source, auc_target, pauc_all])

            print('AUC: %.4f, PAUC: %.4f, F1: %.4f' % (auc_all, pauc_all, f1_all))
            print('AUC source: %.4f, PAUC source: %.4f, F1 source: %.4f' % (auc_source, pauc_source, f1_source))
            print('AUC target: %.4f, PAUC target: %.4f, F1 target: %.4f' % (auc_target, pauc_target, f1_target))
        else:
            auc_all = pauc_all = f1_all = np.nan
            auc_source = pauc_source = f1_source = np.nan
            auc_target = pauc_target = f1_target = np.nan
            official_score = np.nan

        # ===== find the retrieval accuracy
        topk_indexes_source = topk_indexes_raw[source_list_machine]
        topk_indexes_target = topk_indexes_raw[~source_list_machine]
        source = 0

        if n_mix_support is not None:
            acc_source = 0
            acc_target = 0
        else:
            print(f'len topk indexes source: {len(topk_indexes_source)}')
            for id, ix in enumerate(topk_indexes_source):
                ret_name = train_dataset.file_list[ix].split('/')[-1]
                # print(ix, ret_name)
                if 'source' in ret_name:
                    source+=1
            acc_source = source / len(topk_indexes_source)

            target = 0
            print(f'len topk indexes target: {len(topk_indexes_target)}')
            # print(topk_indexes_target)
            for id, ix in enumerate(topk_indexes_target):
                # print(f'ix: {ix}')
                ret_name = train_dataset.file_list[ix].split('/')[-1]
                # print(ix, ret_name)
                if 'target' in ret_name:
                    target+=1
            acc_target = target / len(topk_indexes_target)
            print(f'retrieval acc source: {acc_source}, target: {acc_target}')

        df_log = pd.concat([df_log, pd.DataFrame({
            'layer': [num_layer+1],
            'machine': [class_name],
            'auc_source': [auc_source],
            'auc_target': [auc_target],
            'pauc': [pauc_all],
            'official_score': [official_score],
            'auc': [auc_all],
            'pauc_source': [pauc_source],
            'pauc_target': [pauc_target],
            'acc_source': [acc_source],
            'acc_target': [acc_target],
            'f1': [f1_all],
            'f1_source': [f1_source],
            'f1_target': [f1_target],
            })]
        )
        df_log.to_csv(f'{save_dir}/result.csv', index=False)

        # ======== create challenge submission files ========
        # Save raw anomaly scores for Baseline late fusion.
        if save_official:
            sub_condition = (
                f'team_Baseline_{model_name}_raw_pooling{pooling_feature}'
                f'_topk{top_k}'
                f'_input{input_type}'
                f'_differentview{different_view}'
                f'{fixed_residual_condition}'
                f'_channel{channel_index}'
                f'_n_mix_support{n_mix_support}_alpha{alpha}'
            )
            sub_path_sb = (
                './dcase2023_task2_evaluator/teams/submission/'
                f'{sub_condition}_single_best'
            )
            sub_path_mb = (
                './dcase2023_task2_evaluator/teams/submission/'
                f'{sub_condition}_machine_best'
            )
            if not os.path.exists(sub_path_sb):
                os.makedirs(sub_path_sb)
            if not os.path.exists(sub_path_mb):
                os.makedirs(sub_path_mb)

            results_an[num_layer+1] = [str(s) for s in scores]
            layer_score_path = os.path.join(
                save_dir,
                'anomaly_scores',
                f'layer_{num_layer+1:02d}',
            )
            os.makedirs(layer_score_path, exist_ok=True)
            results_an_layer = pd.DataFrame()
            results_an_layer['output1'] = results_an['output1']
            results_an_layer[num_layer+1] = results_an[num_layer+1]
            results_an_layer.to_csv(
                layer_score_path + '/anomaly_score_'
                + class_name + '_section_00' + '_test.csv',
                encoding='utf-8',
                index=False,
                header=False,
            )

            # decision results
            if has_eval_labels:
                precision, recall, thresholds = precision_recall_curve(
                    gt_list,
                    scores,
                )
                f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
                idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[idx]
                # threshold = np.percentile(train_scores, q=90)
                decisions = scores > optimal_threshold
            else:
                decisions = np.zeros_like(scores, dtype=bool)
            results_dec[num_layer+1] = [str(int(s)) for s in decisions]

            if num_layer+1 == single_best_layer:
                results_an_single = pd.DataFrame()
                results_an_single['output1'] = results_an['output1']
                results_an_single[num_layer+1] = results_an[num_layer+1]
                results_dec_single = pd.DataFrame()
                results_dec_single['output1'] = results_dec['output1']
                results_dec_single[num_layer+1] = results_dec[num_layer+1]
                results_an_single.to_csv(sub_path_sb + '/anomaly_score_' + class_name + '_section_00' + '_test.csv',
                    encoding='utf-8', index=False, header=False)
                results_dec_single.to_csv(sub_path_sb + '/decision_result_' + class_name + '_section_00' + '_test.csv',
                    encoding='utf-8', index=False, header=False)

            if has_eval_labels and official_score > best_score:
                best_score = official_score
                best_layer = num_layer+1

    if save_official and best_layer is not None:
        # save file
        print(f'best layer: {best_layer}')
        print(f'best score: {best_score}')
        results_an_best = pd.DataFrame()
        results_an_best['output1'] = results_an['output1']
        results_an_best[best_layer] = results_an[best_layer]
        results_dec_best = pd.DataFrame()
        results_dec_best['output1'] = results_dec['output1']
        results_dec_best[best_layer] = results_dec[best_layer]
        results_an_best.to_csv(sub_path_mb + '/anomaly_score_' + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)
        results_dec_best.to_csv(sub_path_mb + '/decision_result_' + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)
        # save to DCASE evaluator path
        if not os.path.exists(sub_path_mb):
            os.makedirs(sub_path_mb)
        results_an_best.to_csv(sub_path_mb + '/anomaly_score_' + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)
        results_dec_best.to_csv(sub_path_mb + '/decision_result_' + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)

# ======== Summary with raw kNN scores ========
if args.cache_only:
    print("cache_only pipeline complete")
    if wandb_run is not None:
        wandb_run.finish()
    raise SystemExit(0)

df_test = df_log.reset_index(drop=True)
columns = df_test.columns
df_avg = df_test.groupby('layer')[columns[2:]].agg(hmean)
df_avg['oc'] = df_avg[['auc_source', 'auc_target', 'pauc']].agg(hmean, axis=1) # just in case
print(df_avg)

df_log.to_csv(f'{save_dir}/result.csv', index=False)
df_avg.to_csv(f'{save_dir}/df_avg.csv', index=False)
if df_avg['official_score'].notna().any():
    print('SELECTED FINAL LAYER-WISE SCORE')
    best_layer_wise = df_avg['oc'].argmax()+1 # index + 1 to get the layer
    machine_layer_wise = df_test[df_test['layer'] == best_layer_wise][['machine', 'auc', 'pauc', 'auc_source', 'auc_target', 'official_score']]
    print(machine_layer_wise)

    # print(f' All layer wise score:')
    final_best_layer_wise = pd.DataFrame(df_avg.iloc[df_avg['official_score'].argmax()]).T
    # print(final_best_layer_wise)

    # dev-selective diagnostic only: do not use as the final Baseline score.
    best_each_machine = df_test.loc[df_test.groupby('machine')['official_score'].idxmax()]
    # print(best_each_machine)

    # print(f'FINAL BEST SCORE')
    # final_best_score = pd.DataFrame(best_each_machine[['auc_source', 'auc_target', 'pauc', 'official_score']].agg(hmean)).T
    # print(final_best_score)

    machine_layer_wise.to_csv(f'{save_dir}/machine_layer_wise.csv', index=False)
    final_best_layer_wise.to_csv(f'{save_dir}/final_best_layer_wise.csv', index=False)
    best_each_machine.to_csv(f'{save_dir}/best_each_machine.csv', index=False)
    # final_best_score.to_csv(f'{save_dir}/final_best_score.csv', index=False)

    if wandb_run is not None:
        wandb.log({
            f'raw_knn/final_layer_wise/{key}': float(value)
            for key, value in final_best_layer_wise.iloc[0].items()
        })
        wandb.log({
            'raw_knn/df_avg':
            wandb.Table(dataframe=df_avg.reset_index())
        })
        wandb.log({
            'raw_knn/best_each_machine':
            wandb.Table(dataframe=best_each_machine.reset_index(drop=True))
        })
        artifact = wandb.Artifact(
            f'{wandb_run.name}-dcase2026-raw-knn-results',
            type='result',
        )
        for filename in os.listdir(save_dir):
            if filename.endswith('.csv'):
                artifact.add_file(os.path.join(save_dir, filename))
        wandb_run.log_artifact(artifact)
        wandb.finish()
else:
    print('No binary eval labels; skip raw metric summary.')
    pd.DataFrame().to_csv(f'{save_dir}/machine_layer_wise.csv', index=False)
    pd.DataFrame().to_csv(f'{save_dir}/final_best_layer_wise.csv', index=False)
    pd.DataFrame().to_csv(f'{save_dir}/best_each_machine.csv', index=False)

print(f"Done")
