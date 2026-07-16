import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml


RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))

from run_projection_select import load_projection_config, make_views


class Run21ViewTests(unittest.TestCase):
    def test_make_views_maps_even_rows_to_near_and_odd_rows_to_far(self):
        pair = torch.tensor([[[1.0, 2.0]], [[0.5, 1.5]], [[3.0, 4.0]], [[1.0, 1.0]]])
        views = make_views(pair.unsqueeze(0), layer=1, residual_alpha=0.5)
        torch.testing.assert_close(views["near"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        torch.testing.assert_close(views["far"], torch.tensor([[0.5, 1.5], [1.0, 1.0]]))
        torch.testing.assert_close(views["residual"], torch.tensor([[0.75, 1.25], [2.5, 3.5]]))

    def test_config_accepts_near_and_residual_modes_without_memmix(self):
        config = {
            "train_split": "train",
            "eval_split": "valid",
            "cache_root_template": "cache/{model_name}",
            "score_dir": "out",
            "n_mix_support": 0,
            "systems": [
                {"system_id": "near", "mode": "near", "encoders": [{"model_name": "sslam", "layer": 12}]},
                {"system_id": "residual", "mode": "residual", "encoders": [{"model_name": "sslam", "layer": 12}]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            loaded = load_projection_config(path)
        self.assertEqual(loaded.n_mix_support, 0)
        self.assertEqual([system.mode for system in loaded.systems], ["near", "residual"])


if __name__ == "__main__":
    unittest.main()
