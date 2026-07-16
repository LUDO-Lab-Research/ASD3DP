import numpy as np
from tqdm import tqdm
import torch
import soundfile as sf
from torch.utils.data import Dataset

from src.datasets.clip_sampling import fixed_10s_sample
from src.residual_view.input_type import make_input_type


class AudioDataset(Dataset):
    def __init__(
            self,
            file_list,
            label_list,
            machine_list,
            source_list=None,
            class_ids=None,
            load_in_memory=False,
            machine_name=None,
            audio_length=None,
            extracted_features=None,
            domain=None,
            channel_index=None,
            input_types=None,
        ):
        self.machine_name = machine_name
        self.extracted_features = None
        self.domain = domain
        self.channel_index = channel_index
        self.input_types = input_types
        # Filter by machine name
        if self.machine_name is not None:
            self.file_list = [file for file, machine in zip(file_list, machine_list) if machine == self.machine_name]
            self.label_list = [label for label, machine in zip(label_list, machine_list) if machine == self.machine_name]
            self.machine_list = [machine for machine in machine_list if machine == self.machine_name]

            if source_list is not None:
                self.source_list = [source for source, machine in zip(source_list, machine_list) if machine == self.machine_name]

            if class_ids is not None:
                self.class_ids = [class_id for class_id, machine in zip(class_ids, machine_list) if machine == self.machine_name]

            if extracted_features is not None:
                self.extracted_features = [feature for feature, machine in zip(extracted_features, machine_list) if machine == self.machine_name]
        else:
            self.file_list = file_list
            self.label_list = label_list
            self.machine_list = machine_list

            if source_list is not None:
                self.source_list = source_list

            if class_ids is not None:
                self.class_ids = class_ids

            if extracted_features is not None:
                self.extracted_features = extracted_features

        if source_list is not None:
            if domain == "source":
                self.file_list = [file for file, source in zip(self.file_list, self.source_list) if source == "source"]
                self.label_list = [label for label, source in zip(self.label_list, self.source_list) if source == "source"]
                self.machine_list = [machine for machine, source in zip(self.machine_list, self.source_list) if source == "source"]
                self.source_list = [source for source in self.source_list if source == "source"]
                if class_ids is not None:
                    self.class_ids = [class_id for class_id, source in zip(class_ids, self.source_list) if source == "source"]
                if extracted_features is not None:
                    self.extracted_features = [feature for feature, source in zip(self.extracted_features, self.source_list) if source == "source"]

            elif domain == "target":
                self.file_list = [file for file, source in zip(self.file_list, self.source_list) if source == "target"]
                self.label_list = [label for label, source in zip(self.label_list, self.source_list) if source == "target"]
                self.machine_list = [machine for machine, source in zip(self.machine_list, self.source_list) if source == "target"]
                self.source_list = [source for source in self.source_list if source == "target"]
                if class_ids is not None:
                    self.class_ids = [class_id for class_id, source in zip(class_ids, self.source_list) if source == "target"]
                if extracted_features is not None:
                    self.extracted_features = [feature for feature, source in zip(self.extracted_features, self.source_list) if source == "target"]

        self.load_in_memory = load_in_memory
        self.audio_length = audio_length

        if self.input_types is not None:
            self._expand_input_types()

        if self.extracted_features is not None:
            self.data_list = [self.get_extracted_data(feature, label, class_id) for feature, label, class_id in
                              tqdm(zip(self.extracted_features, self.label_list, self.class_ids))] if load_in_memory else []
        else:
            self.data_list = [
                self.get_data(file, label, self._get_input_type(index))
                for index, (file, label) in enumerate(
                    zip(self.file_list, self.label_list)
                )
            ] if load_in_memory else []

        print(f"Number of samples: {len(self)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        if self.extracted_features is not None:
            data_item = self.data_list[item] if self.load_in_memory else self.get_extracted_data(
                self.extracted_features[item], self.label_list[item], self.class_ids[item]
            )

        else:
            data_item = self.data_list[item] if self.load_in_memory else self.get_data(
                self.file_list[item],
                self.label_list[item],
                self._get_input_type(item),
            )
        return data_item

    def get_data(self, filename, label, input_type=None):
        wave, sr = sf.read(filename, always_2d=True)
        if sr != 16000:
            raise ValueError(f"Sample rate is not 16kHz: {sr}")
        if input_type is not None:
            wave = make_input_type(wave, input_type)
        elif self.channel_index is not None:
            if self.channel_index < 0 or self.channel_index >= wave.shape[1]:
                raise ValueError(
                    f"Invalid channel_index={self.channel_index} "
                    f"for {filename} with shape {wave.shape}"
                )
            wave = wave[:, self.channel_index]
        elif wave.shape[1] == 1:
            wave = wave[:, 0]
        else:
            raise ValueError(
                f"Multi-channel audio requires channel_index: {filename}"
            )

        if self.audio_length is not None:
            wave = fixed_10s_sample(wave, audio_length=self.audio_length)
        wave = torch.from_numpy(np.asarray(wave, dtype=np.float32))

        return {
            "input": wave, # some are not 10 secs
            "label": torch.Tensor([label]).long(),
        }

    def get_extracted_data(self, feature, label, class_id):
        return {
            "input": feature, # some are not 10 secs
            "label": torch.Tensor([label]).long(),
            # "class_id": torch.Tensor([class_id]).long(),
            "class_id": class_id,
        }

    def adjust_size(self, wave, new_size):
        audio_length = new_size
        if wave.shape[0] < audio_length:
            wave = np.pad(wave, (0, audio_length-wave.shape[0]), 'constant')
        else:
            wave = wave[:audio_length]
        return wave

    def _expand_input_types(self):
        """Duplicate metadata so each file can expose one or more inputs."""
        expanded_files = []
        expanded_labels = []
        expanded_machines = []
        expanded_input_types = []
        expanded_sources = [] if hasattr(self, "source_list") else None
        expanded_class_ids = [] if hasattr(self, "class_ids") else None

        for idx, filename in enumerate(self.file_list):
            for input_type in self.input_types:
                expanded_files.append(filename)
                expanded_labels.append(self.label_list[idx])
                expanded_machines.append(self.machine_list[idx])
                expanded_input_types.append(input_type)
                if expanded_sources is not None:
                    expanded_sources.append(self.source_list[idx])
                if expanded_class_ids is not None:
                    expanded_class_ids.append(self.class_ids[idx])

        self.file_list = expanded_files
        self.label_list = expanded_labels
        self.machine_list = expanded_machines
        self.input_type_list = expanded_input_types
        if expanded_sources is not None:
            self.source_list = expanded_sources
        if expanded_class_ids is not None:
            self.class_ids = expanded_class_ids

    def _get_input_type(self, item):
        if hasattr(self, "input_type_list"):
            return self.input_type_list[item]
        return None
