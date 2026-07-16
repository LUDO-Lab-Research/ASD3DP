import glob
import os
import re
from pathlib import Path

import numpy as np
import yaml


def get_filename_list(dir_path, pattern="*", ext="wav"):
    file_path_pattern = os.path.join(dir_path, f"{pattern}.{ext}")
    return sorted(glob.glob(file_path_pattern))


def get_meta_list(file_list):
    label_list = []
    source_list = []
    for file_path in file_list:
        filename = Path(file_path).name
        if "normal" in filename:
            label_list.append(0)
        elif "anomaly" in filename:
            label_list.append(1)
        else:
            label_list.append(-1)

        if "source" in filename:
            source_list.append("source")
        elif "target" in filename:
            source_list.append("target")
        else:
            source_list.append("unknown")
    return label_list, source_list


def get_data_list(split_dirs):
    file_list = []
    for split_dir in split_dirs:
        file_list.extend(get_filename_list(split_dir, ext="wav"))
    label_list, source_list = get_meta_list(file_list)
    return file_list, label_list, source_list


def get_attributes(file_list, source_list):
    machine_list = []
    attrs = []
    for idx, file_path in enumerate(file_list):
        source = str(source_list[idx])
        path = Path(file_path)
        machine = path.parent.parent.name
        filename = path.name
        section = re.findall(r"section_[0-9][0-9]", filename)[0]
        tail = filename.split(".wav")[0].split("_")[6:]
        attr_id = "_".join(tail) if tail else "noAttribute"
        machine_id = f"{machine}_{section.split('_')[-1]}"
        machine_list.append(machine)
        attrs.append("###".join([machine_id, attr_id, source]))
    return np.array(machine_list), np.array(attrs)


def prepare_data(split_dirs):
    file_list, label_list, source_list = get_data_list(split_dirs)
    machine_names, file_attrs = get_attributes(file_list, source_list)
    return file_list, label_list, source_list, machine_names, file_attrs


def make_split(file_list, label_list, source_list, machine_names, file_attrs):
    return {
        "file_list": file_list,
        "label_list": label_list,
        "source_list": source_list,
        "machine_names": machine_names,
        "file_attrs": file_attrs,
    }


def get_dcase2026(train_split="train", config_path=None):
    if config_path is None:
        config_path = os.environ.get(
            "DCASE2026_CONFIG",
            "./config/data_config_2026.yaml",
        )
    with open(config_path, "r", encoding="utf-8") as ymlfile:
        data_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if train_split == "train":
        train_dirs = sorted(data_config["dcase2026_train_dirs"])
    elif train_split == "eval_train":
        train_dirs = sorted(data_config["dcase2026_eval_train_dirs"])
    else:
        raise ValueError(
            "DCASE2026 Baseline currently supports train or eval_train."
        )

    valid_dirs = sorted(data_config["dcase2026_valid_dirs"])
    eval_dirs = sorted(data_config.get("dcase2026_eval_dirs", []))

    train = prepare_data(train_dirs)
    valid = prepare_data(valid_dirs)

    splits = {
        "train": make_split(*train),
        "valid": make_split(*valid),
    }
    if eval_dirs:
        eval_data = prepare_data(eval_dirs)
        splits["eval"] = make_split(*eval_data)

    return splits
