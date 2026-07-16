import os
import sys
import glob
import json
import types
import importlib.util
from importlib import import_module

import torch


class BeatsFeatureExtractor:
    def __init__(self, model_name, pretrained_model_dir):
        from beats.BEATs import BEATs, BEATsConfig

        ckpt_path = self._checkpoint_path(model_name, pretrained_model_dir)
        checkpoint = torch.load(ckpt_path)
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model_name = model_name
        self.ckpt_path = ckpt_path

    @staticmethod
    def _checkpoint_path(model_name, pretrained_model_dir):
        ckpt_names = {
            "beats": "BEATs_iter3_plus_AS2M.pt",
            "beats_ft1": "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
            "beats_ft2": "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            "beats_iter3": "BEATs_iter3.pt",
            "beats_iter3_ft1": "BEATs_iter3_finetuned_on_AS2M_cpt1.pt",
            "beats_iter3_ft2": "BEATs_iter3_finetuned_on_AS2M_cpt2.pt",
        }
        if model_name not in ckpt_names:
            raise ValueError(f"Unsupported BEATs model_name: {model_name}")
        return os.path.join(pretrained_model_dir, ckpt_names[model_name])

    def extract_batch(self, x, pooling_feature=None, score_layer=None):
        bs = x.size(0)
        padding_mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        _, _, attns = self.model.extract_features(
            x,
            padding_mask=padding_mask,
            need_weights=True,
            layer=11,
        )

        if pooling_feature == "temporal":
            out_layers = [
                f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(1).cpu().unsqueeze(0)
                for f_layer in attns
            ][1:]
        elif pooling_feature == "spectral":
            out_layers = [
                f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(2).cpu().unsqueeze(0)
                for f_layer in attns
            ][1:]
        else:
            out_layers = [
                f_layer[0].transpose(0, 1).mean(1).cpu().unsqueeze(0)
                for f_layer in attns
            ][1:]
        if score_layer is not None:
            out_layers = [out_layers[score_layer - 1]]

        return torch.cat(out_layers, 0)


class CedFeatureExtractor:
    _CHECKPOINTS = {
        "ced_tiny": "audiotransformer_tiny_mAP_4814.pt",
        "ced_base": "audiotransformer_base_mAP_4999.pt",
    }

    def __init__(self, pretrained_model_dir, model_name="ced_tiny"):
        vendor_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "vendors", "ced"
            )
        )
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

        from einops import rearrange
        from models.audiotransformer import ced_base, ced_tiny

        ckpt_path = os.path.join(pretrained_model_dir, self._CHECKPOINTS[model_name])
        model_factory = {"ced_tiny": ced_tiny, "ced_base": ced_base}[model_name]
        self.model = model_factory(pretrained=True, pretrained_url=ckpt_path)
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self._rearrange = rearrange

    def extract_batch(self, x, pooling_feature="temporal", score_layer=None):
        pooling_feature = self._normalize_pooling(pooling_feature)
        token_layers = self._extract_token_layers(x)

        if pooling_feature == "temporal":
            out_layers = self._temporal_pool_ced_layers(token_layers)
        else:
            raise ValueError(
                "CED wrapper currently supports GenRep-style Temporal Pooling only; "
                f"got pooling_feature={pooling_feature}."
            )

        return torch.cat(out_layers, 0)

    @staticmethod
    def _normalize_pooling(pooling_feature):
        if pooling_feature is None:
            return "temporal"
        return pooling_feature

    def _extract_token_layers(self, x):
        x = self._waveform_to_patch_tokens(x)
        token_layers = []
        for block in self.model.blocks:
            x = block(x)
            token_layers.append(self.model.norm(x))
        return token_layers

    def _waveform_to_patch_tokens(self, x):
        x = self.model.front_end(x)
        x = self._rearrange(x, "b f t -> b 1 f t")
        x = self.model.init_bn(x)
        if x.shape[-1] > self.model.maximal_allowed_length:
            raise ValueError(
                "CED input is longer than its target_length. Keep the current 10s "
                "policy or handle this in the clip-length ablation."
            )

        x = self.model.patch_embed(x)
        # Vendor CED forward_features treats this as [B, C, F, T]
        # and then flattens the frequency-time grid into F*T tokens.
        _, _, freq_patches, time_patches = x.shape
        expected_freq_patches = 4
        expected_time_patches = 62
        if (freq_patches, time_patches) != (
            expected_freq_patches,
            expected_time_patches,
        ):
            raise RuntimeError(
                f"{self.model_name} 10s Baseline expects patch grid 4 x 62; "
                f"got {freq_patches} x {time_patches}."
            )
        x = x + self.model.time_pos_embed[:, :, :, :time_patches]
        x = x + self.model.freq_pos_embed[:, :, :, :]
        x = self._rearrange(x, "b c f t -> b (f t) c")
        if self.model.pooling == "token":
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.model.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        return self.model.pos_drop(x)

    def _temporal_pool_ced_layers(self, token_layers):
        expected_layers = 12
        if len(token_layers) != expected_layers:
            raise RuntimeError(
                f"{self.model_name} wrapper expects 12 block outputs; "
                f"got {len(token_layers)}."
            )

        out_layers = []
        for layer_idx in range(expected_layers):
            layer_x = token_layers[layer_idx]
            pooled = self._temporal_pool(layer_x)
            out_layers.append(pooled.cpu().unsqueeze(0))
        return out_layers

    def _temporal_pool(self, layer_x):
        batch_size = layer_x.size(0)
        freq_bins = 4
        time_bins = 62
        target_pooled_groups = 8
        embed_dim = layer_x.size(-1)
        expected_tokens = freq_bins * time_bins
        if layer_x.size(1) != expected_tokens:
            raise RuntimeError(
                f"{self.model_name} Temporal Pooling expects 4 x 62 patch tokens; "
                f"got {layer_x.size(1)} tokens."
            )
        if target_pooled_groups % freq_bins != 0:
            raise RuntimeError(
                "CED target pooled groups must divide the frequency grid; "
                f"got {target_pooled_groups} groups and {freq_bins} freq bins."
            )

        time_groups = target_pooled_groups // freq_bins
        if time_bins % time_groups != 0:
            raise RuntimeError(
                "CED time bins must divide into the target pooled groups; "
                f"got {time_bins} time bins and {time_groups} time groups."
            )
        frames_per_group = time_bins // time_groups

        # The rule is 8 pooled groups. For CED's 4 x 62 grid this becomes
        # 4 frequency bins x 2 derived time groups, then average within
        # each time group.
        layer_x = layer_x.reshape(
            batch_size,
            freq_bins,
            time_groups,
            frames_per_group,
            embed_dim,
        )
        layer_x = layer_x.mean(3).reshape(
            batch_size, target_pooled_groups, embed_dim
        )
        return layer_x.flatten(1)


class LoadOnlyFeatureExtractor:
    def __init__(self, model, model_name, ckpt_path):
        self.model = model
        self.model_name = model_name
        self.ckpt_path = ckpt_path

    def extract_batch(self, x, pooling_feature="temporal", score_layer=None):
        if pooling_feature is None:
            pooling_feature = "temporal"
        raise NotImplementedError(
            f"{self.model_name} is load-only for now with requested "
            f"pooling_feature={pooling_feature}. Add encoder-specific "
            "preprocessing/layer extraction before running feature extraction."
        )


class DaShengFeatureExtractor:
    def __init__(self, model_name="dasheng_base"):
        try:
            from dasheng import dasheng_06B, dasheng_12B, dasheng_base
        except ImportError as exc:
            raise ImportError(
                "DaSheng is not installed. Install it with `python -m pip install dasheng` "
                "before running --model_name dasheng_base."
            ) from exc

        factories = {
            "dasheng_base": dasheng_base,
            "dasheng_06b": dasheng_06B,
            "dasheng_12b": dasheng_12B,
        }
        if model_name not in factories:
            raise ValueError(f"Unsupported DaSheng model_name: {model_name}")
        self.model = factories[model_name]()
        self.model_name = model_name
        self.ckpt_path = "dasheng package default checkpoint"

    def extract_batch(self, x, pooling_feature="temporal", score_layer=None):
        if score_layer not in {None, 1}:
            raise ValueError(
                f"{self.model_name} exposes one final feature layer in this wrapper; "
                f"got score_layer={score_layer}."
            )
        pooling_feature = "temporal" if pooling_feature is None else pooling_feature
        if pooling_feature != "temporal":
            raise ValueError(
                f"{self.model_name} wrapper currently supports temporal pooling only; "
                f"got pooling_feature={pooling_feature}."
            )

        features = self.model(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if not torch.is_tensor(features):
            raise TypeError(f"{self.model_name} returned non-tensor features: {type(features)}")
        if features.dim() == 3:
            features = features.mean(1)
        elif features.dim() > 3:
            features = features.flatten(1)
        elif features.dim() != 2:
            raise RuntimeError(
                f"{self.model_name} expected [B, T, D] or [B, D] features; "
                f"got shape={tuple(features.shape)}."
            )
        return features.cpu().unsqueeze(0)


class AudioMaePlusPlusFeatureExtractor:
    _CONFIGS = {
        "audiomaepp_tiny": "maepp_tiny_200_16x4",
        "audiomaepp_tiny_200_16x4": "maepp_tiny_200_16x4",
        "audiomaepp_base": "maepp_base_200_16x4",
        "audiomaepp_base_200_16x4": "maepp_base_200_16x4",
        "audiomaepp_large": "maepp_large_200_16x4",
        "audiomaepp_large_200_16x4": "maepp_large_200_16x4",
    }

    def __init__(self, model_name, pretrained_model_dir):
        vendor_candidates = [
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "vendors",
                    "audiomae-plusplus-official",
                )
            ),
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "vendors",
                    "audiomaepp",
                )
            ),
        ]
        for vendor_path in vendor_candidates:
            if os.path.isdir(vendor_path) and vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)

        try:
            RuntimeMAE = self._import_runtime_mae()
        except ImportError as exc:
            raise ImportError(
                "AudioMAE++ HEAR API is not available. Clone "
                "https://github.com/SarthakYadav/audiomae-plusplus-official into "
                "`vendors/audiomae-plusplus-official` and run `uv sync` there."
            ) from exc

        if model_name not in self._CONFIGS:
            raise ValueError(f"Unsupported AudioMAE++ model_name: {model_name}")
        config_name = self._CONFIGS[model_name]
        config = import_module(f"configs.{config_name}").get_config()
        pretrained_dir = os.environ.get("PT_MAEPP_MODEL_DIR", pretrained_model_dir)
        if not pretrained_dir or not os.path.isdir(pretrained_dir):
            raise FileNotFoundError(
                "AudioMAE++ pretrained directory is missing. Set PT_MAEPP_MODEL_DIR "
                "or pass --pretrained_model_dir to the downloaded pretrained-weight folder."
            )
        config_dir = os.path.join(pretrained_dir, config_name)
        if os.path.isdir(config_dir):
            pretrained_dir = config_dir
        ckpt_glob = os.path.join(pretrained_dir, "checkpoints", "*.pth")
        if not glob.glob(ckpt_glob):
            raise FileNotFoundError(
                "AudioMAE++ checkpoint is missing. Expected at least one file "
                f"matching {ckpt_glob}. PT_MAEPP_MODEL_DIR may point either to "
                "the weight root or directly to the config-specific directory."
            )

        self.model = RuntimeMAE(config, pretrained_dir)
        self.model_name = model_name
        self.config_name = config_name
        self.ckpt_path = pretrained_dir

    def extract_batch(self, x, pooling_feature="temporal", score_layer=None):
        if score_layer not in {None, 1}:
            raise ValueError(
                f"{self.model_name} exposes one scene-embedding layer in this wrapper; "
                f"got score_layer={score_layer}."
            )
        pooling_feature = "temporal" if pooling_feature is None else pooling_feature
        if pooling_feature != "temporal":
            raise ValueError(
                f"{self.model_name} wrapper currently supports temporal pooling only; "
                f"got pooling_feature={pooling_feature}."
            )

        features = self.model.get_scene_embeddings(x)
        features = self._first_tensor(features)
        if features.dim() == 3:
            features = features.mean(1)
        elif features.dim() > 3:
            features = features.flatten(1)
        elif features.dim() != 2:
            raise RuntimeError(
                f"{self.model_name} expected [B, T, D] or [B, D] features; "
                f"got shape={tuple(features.shape)}."
            )
        return features.cpu().unsqueeze(0)

    @staticmethod
    def _import_runtime_mae():
        repo_src_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "src" or name.startswith("src.")
        }
        for name in repo_src_modules:
            sys.modules.pop(name, None)
        for name in list(sys.modules):
            if name == "hear_api" or name.startswith("hear_api."):
                sys.modules.pop(name, None)
        try:
            from hear_api import RuntimeMAE
        finally:
            for name in list(sys.modules):
                if name == "src" or name.startswith("src."):
                    sys.modules.pop(name, None)
            sys.modules.update(repo_src_modules)
        return RuntimeMAE

    @staticmethod
    def _first_tensor(features):
        if torch.is_tensor(features):
            return features
        if isinstance(features, dict):
            for key in ["scene_embedding", "scene_embeddings", "embedding", "embeddings"]:
                value = features.get(key)
                if torch.is_tensor(value):
                    return value
            for value in features.values():
                if torch.is_tensor(value):
                    return value
        if isinstance(features, (tuple, list)):
            for value in features:
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"AudioMAE++ returned no tensor features: {type(features)}")


class FisherFeatureExtractor:
    _MODEL_IDS = {
        "fisher_tiny": "jiangab/FISHER-tiny-0723",
        "fisher_small": "jiangab/FISHER-small-0723",
    }
    _REVISIONS = {
        "fisher_tiny": "f9ac81c66b13f97386e4e39b1777e64ba4c98459",
        "fisher_small": "c21d26b6a2f0593ddeea23871d66b76899c07280",
    }

    def __init__(self, model_name="fisher_small", pretrained_model_dir=None):
        try:
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
        except ImportError as exc:
            raise ImportError(
                "FISHER requires transformers. Install it with "
                "`python -m pip install transformers` before running "
                "--model_name fisher_small."
            ) from exc

        if model_name not in self._MODEL_IDS:
            raise ValueError(f"Unsupported FISHER model_name: {model_name}")

        model_id = os.environ.get("FISHER_MODEL_ID", self._MODEL_IDS[model_name])
        revision = os.environ.get(
            "FISHER_REVISION",
            self._REVISIONS[model_name],
        )
        cache_dir = os.environ.get("FISHER_CACHE_DIR")
        if cache_dir is None and pretrained_model_dir:
            cache_dir = pretrained_model_dir

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model_cls = get_class_from_dynamic_module(
            "modeling_fisher.FISHERModel",
            model_id,
            revision=revision,
        )
        if not hasattr(model_cls, "all_tied_weights_keys"):
            # FISHER remote code was authored against a newer Transformers API.
            # Current local Transformers expects this attribute during
            # `from_pretrained` finalization even when there are no tied weights.
            model_cls.all_tied_weights_keys = {}

        self.model = model_cls.from_pretrained(
            model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
        )
        self.model_name = model_name
        self.model_id = model_id
        self.revision = revision
        self.ckpt_path = f"{model_id}@{revision}"

    def extract_batch(self, x, pooling_feature="temporal", score_layer=None):
        if score_layer not in {None, 1}:
            raise ValueError(
                f"{self.model_name} exposes one extract_features layer in this "
                f"wrapper; got score_layer={score_layer}."
            )
        pooling_feature = "temporal" if pooling_feature is None else pooling_feature
        if pooling_feature != "temporal":
            raise ValueError(
                f"{self.model_name} wrapper currently supports temporal pooling "
                f"only; got pooling_feature={pooling_feature}."
            )

        spec = self._waveform_to_fisher_spec(x)
        with torch.autocast(device_type=x.device.type, enabled=x.is_cuda):
            features = self.model.extract_features(spec)
        features = self._first_tensor(features)
        if features.dim() == 3:
            features = features.mean(1)
        elif features.dim() > 3:
            features = features.flatten(1)
        elif features.dim() != 2:
            raise RuntimeError(
                f"{self.model_name} expected [B, T, D] or [B, D] features; "
                f"got shape={tuple(features.shape)}."
            )
        return features.cpu().unsqueeze(0)

    def _waveform_to_fisher_spec(self, x):
        import torch.nn.functional as F
        import torchaudio

        sample_rate = int(os.environ.get("FISHER_SAMPLE_RATE", "16000"))
        if sample_rate != 16000:
            raise ValueError(
                "This DCASE wrapper expects 16 kHz waveforms before FISHER "
                f"preprocessing; got FISHER_SAMPLE_RATE={sample_rate}."
            )

        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=25 * sample_rate // 1000,
            win_length=None,
            hop_length=10 * sample_rate // 1000,
            power=1,
            center=False,
        ).to(x.device)

        x = x - x.mean(dim=-1, keepdim=True)
        spec = torch.log(torch.abs(spectrogram(x)) + 1e-10)
        spec = spec.transpose(-2, -1)
        spec = (spec + 3.017344307886898) / (2.1531635155379805 * 2)
        if spec.shape[-2] > 1024:
            spec = spec[:, :1024]
        band_width = int(self.model.cfg.band_width)
        if spec.shape[-1] < band_width:
            spec = F.pad(spec, (0, band_width - spec.shape[-1]))
        return spec.unsqueeze(1)

    @staticmethod
    def _first_tensor(features):
        if torch.is_tensor(features):
            return features
        if isinstance(features, dict):
            for key in ["features", "last_hidden_state", "embedding", "embeddings"]:
                value = features.get(key)
                if torch.is_tensor(value):
                    return value
            for value in features.values():
                if torch.is_tensor(value):
                    return value
        if isinstance(features, (tuple, list)):
            for value in features:
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"FISHER returned no tensor features: {type(features)}")


def _find_file(base_dir, candidates):
    for candidate in candidates:
        path = os.path.join(base_dir, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these files exists under {base_dir}: {candidates}")


def _load_eat_snapshot(snapshot_dir, package_name):
    module = types.ModuleType(package_name)
    module.__path__ = [snapshot_dir]
    sys.modules[package_name] = module

    for name in ["configuration_eat", "model_core", "eat_model", "modeling_eat"]:
        file_path = os.path.join(snapshot_dir, f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", file_path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{name}"] = loaded
        spec.loader.exec_module(loaded)

    config_cls = sys.modules[f"{package_name}.configuration_eat"].EATConfig
    model_cls = sys.modules[f"{package_name}.modeling_eat"].EATModel

    with open(os.path.join(snapshot_dir, "config.json")) as f:
        config = config_cls(**json.load(f))
    model = model_cls(config)

    from safetensors.torch import load_file

    ckpt_path = os.path.join(snapshot_dir, "model.safetensors")
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    return model, ckpt_path


class EatFeatureExtractor:
    def __init__(self, model_name, pretrained_model_dir):
        if model_name == "eat_large":
            snapshot_dir = _find_file(
                pretrained_model_dir, ["EAT-large_epoch20_pretrain"]
            )
            package_name = "residual_view_eat_large"
        elif model_name == "sslam":
            snapshot_dir = _find_file(pretrained_model_dir, ["SSLAM_pretrain"])
            package_name = "residual_view_sslam"
        else:
            raise ValueError(f"Unsupported EAT-style model_name: {model_name}")

        import torchaudio.compliance.kaldi as ta_kaldi

        model, ckpt_path = _load_eat_snapshot(snapshot_dir, package_name)
        self.model = model
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self._ta_kaldi = ta_kaldi

    def extract_batch(self, x, pooling_feature="temporal", score_layer=None):
        pooling_feature = self._normalize_pooling(pooling_feature)
        rms_time_weights = None
        if pooling_feature == "rms_temporal":
            rms_time_weights = self._rms_time_weights(x)
        token_layers = self._extract_patch_token_layers(x, score_layer)

        supported_pooling = {
            "temporal",
            "spectral",
            "gem",
            "rdp",
            "rdp_gem",
            "rms_temporal",
        }
        if pooling_feature not in supported_pooling:
            raise ValueError(
                f"{self.model_name} wrapper supports "
                f"{sorted(supported_pooling)}; "
                f"got pooling_feature={pooling_feature}."
            )
        if self.model_name != "sslam" and pooling_feature != "temporal":
            raise ValueError(
                "Pooling ablation is currently opened only for SSLAM; "
                f"got model_name={self.model_name}, "
                f"pooling_feature={pooling_feature}."
            )

        expected_layers = 1 if score_layer is not None else None
        out_layers = self._pool_eat_layers(
            token_layers,
            pooling_feature,
            rms_time_weights,
            expected_layers,
        )

        return torch.cat(out_layers, 0)

    @staticmethod
    def _normalize_pooling(pooling_feature):
        if pooling_feature is None:
            return "temporal"
        return pooling_feature

    def _waveform_to_mel(self, x):
        import torch.nn.functional as F

        target_length = 1024
        norm_mean = -4.268
        norm_std = 4.569
        mels = []
        for waveform in x:
            waveform = waveform - waveform.mean()
            mel = self._ta_kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
            n_frames = mel.shape[0]
            if n_frames < target_length:
                mel = F.pad(mel, (0, 0, 0, target_length - n_frames))
            else:
                mel = mel[:target_length, :]
            mel = (mel - norm_mean) / (norm_std * 2)
            mels.append(mel)
        return torch.stack(mels, dim=0).unsqueeze(1)

    def _extract_patch_token_layers(self, x, score_layer=None):
        x = self._waveform_to_mel(x)
        core_model = self.model.model
        if score_layer is not None and (
            score_layer < 1 or score_layer > core_model.config.depth
        ):
            raise ValueError(
                f"score_layer={score_layer} is outside available "
                f"layers 1..{core_model.config.depth}"
            )
        x = core_model.local_encoder(x)
        self._check_patch_grid(x)

        if core_model.fixed_positional_encoder is not None:
            x = x + core_model.fixed_positional_encoder(x, None)[:, :x.size(1), :]
        cls_token = core_model.extra_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = core_model.pre_norm(x)
        x = core_model.pos_drop(x)

        token_layers = []
        for layer_idx, block in enumerate(core_model.blocks, start=1):
            x, _ = block(x)
            if score_layer is None or layer_idx == score_layer:
                token_layers.append(x[:, 1:])
            if score_layer is not None and layer_idx == score_layer:
                break
        return token_layers

    def _check_patch_grid(self, x):
        time_bins = 64
        freq_bins = 8
        expected_tokens = time_bins * freq_bins
        if x.size(1) != expected_tokens:
            raise RuntimeError(
                f"{self.model_name} 10s Baseline expects patch grid "
                f"64 x 8; got {x.size(1)} tokens."
            )

    def _pool_eat_layers(
        self,
        token_layers,
        pooling_feature,
        rms_time_weights=None,
        expected_layers=None,
    ):
        if expected_layers is None:
            expected_layers = self.model.model.config.depth
        if len(token_layers) != expected_layers:
            raise RuntimeError(
                f"{self.model_name} wrapper expects {expected_layers} "
                f"block outputs; got {len(token_layers)}."
            )

        out_layers = []
        for layer_idx in range(expected_layers):
            layer_x = token_layers[layer_idx]
            if pooling_feature == "temporal":
                # Original GenRep-style temporal pooling:
                # SSLAM patch tokens are [B, 64 time, 8 frequency, D].
                # Average the time axis and keep the 8 frequency groups.
                pooled = (
                    self._sslam_patch_grid(layer_x, "Temporal Pooling")
                    .mean(1)
                    .flatten(1)
                )
            elif pooling_feature == "spectral":
                pooled = self._spectral_pool(layer_x)
            elif pooling_feature == "gem":
                pooled = self._gem_pool(layer_x)
            elif pooling_feature == "rdp":
                pooled = self._rdp_pool(layer_x)
            elif pooling_feature == "rdp_gem":
                pooled = self._rdp_gem_pool(layer_x)
            elif pooling_feature == "rms_temporal":
                pooled = self._rms_temporal_pool(layer_x, rms_time_weights)
            out_layers.append(pooled.cpu().unsqueeze(0))
        return out_layers

    def _sslam_patch_grid(self, layer_x, pooling_name):
        batch_size = layer_x.size(0)
        time_bins = 64
        freq_bins = 8
        embed_dim = layer_x.size(-1)
        expected_tokens = time_bins * freq_bins
        if layer_x.size(1) != expected_tokens:
            raise RuntimeError(
                f"{self.model_name} {pooling_name} expects "
                f"64 x 8 patch tokens; got {layer_x.size(1)} tokens."
            )

        # EAT/SSLAM local_encoder flattens [T, F] patch grid as T*F tokens.
        return layer_x.reshape(batch_size, time_bins, freq_bins, embed_dim)

    def _spectral_pool(self, layer_x):
        # Opposite view: average frequency patches and keep temporal groups.
        layer_x = self._sslam_patch_grid(layer_x, "Spectral Pooling")
        return layer_x.mean(2).flatten(1)

    def _rms_time_weights(self, waveform):
        batch_size = waveform.size(0)
        time_bins = 64
        frames = waveform.reshape(batch_size, time_bins, -1)
        energy = frames.pow(2).mean(2).sqrt()
        return energy / energy.sum(1, keepdim=True).clamp_min(1e-12)

    def _rms_temporal_pool(self, layer_x, rms_time_weights):
        # Acoustic-region variant of temporal pooling.
        # Keep the same 64 x 8 SSLAM grid, but replace uniform time mean
        # with waveform RMS weights over the 64 time regions.
        layer_x = self._sslam_patch_grid(layer_x, "RMS Temporal Pooling")
        weights = rms_time_weights.view(layer_x.size(0), -1, 1, 1)
        return (layer_x * weights).sum(1).flatten(1)

    def _gem_pool(self, layer_x):
        # GeM default p=3 follows the pooling ablation paper's EAT setting.
        p = 3.0
        layer_x = self._sslam_patch_grid(layer_x, "GeM Pooling").clamp_min(0)
        return layer_x.pow(p).mean(1).clamp_min(1e-12).pow(1.0 / p).flatten(1)

    def _rdp_weights(self, layer_x):
        # RDP default gamma=1 follows the pooling ablation paper's EAT setting.
        gamma = 1.0
        batch_size, time_bins, freq_bins, embed_dim = layer_x.shape
        time_vectors = layer_x.reshape(batch_size, time_bins, freq_bins * embed_dim)
        center = time_vectors.mean(1, keepdim=True)
        distance = torch.norm(time_vectors - center, p=2, dim=2)
        min_distance = distance.min(1, keepdim=True).values
        max_distance = distance.max(1, keepdim=True).values
        normalized_distance = (
            (distance - min_distance)
            / (max_distance - min_distance).clamp_min(1e-12)
        )
        weights = (1.0 + normalized_distance).pow(gamma)
        return weights / weights.sum(1, keepdim=True).clamp_min(1e-12)

    def _rdp_pool(self, layer_x):
        layer_x = self._sslam_patch_grid(layer_x, "RDP Pooling")
        weights = self._rdp_weights(layer_x).view(layer_x.size(0), -1, 1, 1)
        return (layer_x * weights).sum(1).flatten(1)

    def _rdp_gem_pool(self, layer_x):
        # RDP chooses time weights; GeM p=3 pools positive activations.
        p = 3.0
        layer_x = self._sslam_patch_grid(layer_x, "RDP + GeM Pooling")
        weights = self._rdp_weights(layer_x).view(layer_x.size(0), -1, 1, 1)
        numerator = layer_x.clamp_min(0).pow(p).mul(weights).sum(1)
        denominator = weights.sum(1).clamp_min(1e-12)
        return (numerator / denominator).clamp_min(1e-12).pow(1.0 / p).flatten(1)


class M2dClapFeatureExtractor:
    def __init__(self, pretrained_model_dir):
        vendor_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "vendors", "m2d", "examples"
            )
        )
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

        import portable_m2d

        ckpt_path = _find_file(
            pretrained_model_dir,
            [
                "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/"
                "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth",
                "checkpoint-30.pth",
            ],
        )
        self.model = portable_m2d.PortableM2D(ckpt_path)
        self.model_name = "m2d_clap"
        self.ckpt_path = ckpt_path
        self._rearrange = portable_m2d.rearrange

    def extract_batch(self, x, pooling_feature="temporal"):
        pooling_feature = self._normalize_pooling(pooling_feature)
        token_layers = self._extract_patch_token_layers(x)

        if pooling_feature == "temporal":
            out_layers = self._temporal_pool_m2d_layers(token_layers)
        else:
            raise ValueError(
                "M2D-CLAP wrapper currently supports GenRep-style "
                f"Temporal Pooling only; got pooling_feature={pooling_feature}."
            )

        return torch.cat(out_layers, 0)

    @staticmethod
    def _normalize_pooling(pooling_feature):
        if pooling_feature is None:
            return "temporal"
        return pooling_feature

    def _extract_patch_token_layers(self, x):
        x = self.model.to_normalized_feature(x)
        unit_frames = self.model.cfg.input_size[1]
        if x.shape[-1] != unit_frames:
            raise RuntimeError(
                "M2D-CLAP 10s Baseline expects 1001 log-mel frames; "
                f"got {x.shape[-1]}."
            )

        backbone = self.model.backbone
        x = backbone.patch_embed(x)
        self._check_patch_grid(x)

        pos_embed = backbone.pos_embed[:, 1:, :]
        x = x + pos_embed[:, :x.shape[1], :]
        cls_token = backbone.cls_token + backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        token_layers = []
        for block in backbone.blocks:
            x = block(x)
            token_layers.append(backbone.norm(x)[:, 1:])
        return token_layers

    def _check_patch_grid(self, x):
        freq_bins = 5
        time_bins = 62
        expected_tokens = freq_bins * time_bins
        if x.size(1) != expected_tokens:
            raise RuntimeError(
                "M2D-CLAP 10s Baseline expects patch grid 5 x 62; "
                f"got {x.size(1)} tokens."
            )

    def _temporal_pool_m2d_layers(self, token_layers):
        expected_layers = 12
        if len(token_layers) != expected_layers:
            raise RuntimeError(
                "M2D-CLAP wrapper expects 12 block outputs; "
                f"got {len(token_layers)}."
            )

        out_layers = []
        for layer_idx in range(expected_layers):
            layer_x = token_layers[layer_idx]
            pooled = self._temporal_pool(layer_x)
            out_layers.append(pooled.cpu().unsqueeze(0))
        return out_layers

    def _temporal_pool(self, layer_x):
        batch_size = layer_x.size(0)
        freq_bins = 5
        time_bins = 62
        embed_dim = layer_x.size(-1)
        expected_tokens = freq_bins * time_bins
        if layer_x.size(1) != expected_tokens:
            raise RuntimeError(
                "M2D-CLAP Temporal Pooling expects 5 x 62 patch tokens; "
                f"got {layer_x.size(1)} tokens."
            )

        # M2D runtime flattens patch tokens as [F, T].
        # Therefore mean(2) below averages the time axis.
        layer_x = layer_x.reshape(batch_size, freq_bins, time_bins, embed_dim)
        return layer_x.mean(2).flatten(1)


def build_feature_extractor(model_name, pretrained_model_dir):
    if model_name.startswith("beats"):
        return BeatsFeatureExtractor(model_name, pretrained_model_dir)
    if model_name in {"ced_tiny", "ced_base"}:
        return CedFeatureExtractor(pretrained_model_dir, model_name=model_name)
    if model_name in {"eat_large", "sslam"}:
        return EatFeatureExtractor(model_name, pretrained_model_dir)
    if model_name == "m2d_clap":
        return M2dClapFeatureExtractor(pretrained_model_dir)
    if model_name in {"dasheng_base", "dasheng_06b", "dasheng_12b"}:
        return DaShengFeatureExtractor(model_name=model_name)
    if model_name in {
        "audiomaepp_tiny",
        "audiomaepp_tiny_200_16x4",
        "audiomaepp_base",
        "audiomaepp_base_200_16x4",
    }:
        return AudioMaePlusPlusFeatureExtractor(model_name, pretrained_model_dir)
    if model_name in {"fisher_tiny", "fisher_small"}:
        return FisherFeatureExtractor(model_name, pretrained_model_dir)
    raise NotImplementedError(
        f"Encoder wrapper is not implemented yet for model_name={model_name}. "
        "Add an encoder-specific extractor instead of reusing BEATs extraction."
    )
