import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "runner.py"
SPEC = importlib.util.spec_from_file_location("run20_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class RunnerContractTest(unittest.TestCase):
    def test_runner_uses_packaged_dataset_loader(self):
        self.assertEqual(runner.SOURCE_DIR, MODULE_PATH.parent)
        self.assertTrue((runner.SOURCE_DIR / "datasets.py").is_file())

    def test_file_score_is_mean_of_frame_vector_mse(self):
        frame_scores = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(runner.reduce_file_score(frame_scores), 2.5)

    def test_gamma_threshold_is_90_percent_quantile(self):
        scores = np.array([0.8, 1.0, 1.2, 1.4, 1.7, 2.0])
        params, threshold = runner.fit_gamma_threshold(scores, quantile=0.9)
        self.assertEqual(len(params), 3)
        self.assertGreater(threshold, 0.0)
        self.assertAlmostEqual(
            threshold,
            runner.gamma_quantile(params, 0.9),
            places=12,
        )

    def test_checkpoint_epochs_are_locked(self):
        self.assertEqual(runner.CHECKPOINT_EPOCHS, (20, 40, 60, 80, 100))

    def test_no_validation_argument_exists(self):
        options = {action.dest for action in runner.build_parser()._actions}
        self.assertNotIn("validation_split", options)
        self.assertNotIn("val_manifest", options)

    def test_smoke_mode_forces_one_epoch(self):
        args = runner.build_parser().parse_args(
            [
                "--raw-root", "/tmp/raw",
                "--cache-root", "/tmp/cache",
                "--output-dir", "/tmp/output",
                "--seed", "23711",
                "--wandb-project", "smoke",
                "--smoke-one-epoch",
            ]
        )
        self.assertEqual(runner.effective_epochs(args), 1)

    def test_official_state_dict_contract(self):
        model = runner.AutoEncoder(640)
        self.assertEqual(tuple(model.cov_source.shape), (128, 128))
        self.assertEqual(tuple(model.cov_target.shape), (128, 128))
        self.assertFalse(model.cov_source.requires_grad)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 300696)


if __name__ == "__main__":
    unittest.main()
