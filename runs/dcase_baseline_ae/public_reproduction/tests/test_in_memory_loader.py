import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import random
import torch


RUN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_DIR / "src"))

from runner import capture_rng_state, iter_resident_batches, load_contiguous_features, restore_rng_state


def make_store(tmp_path: Path, arrays: list[np.ndarray], declared_rows=None, dims=3) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    for index, array in enumerate(arrays):
        np.save(store / f"chunk_{index:06d}.npy", array.astype(np.float32))
    rows = sum(len(array) for array in arrays) if declared_rows is None else declared_rows
    (store / "DONE.json").write_text(
        json.dumps({"rows": rows, "chunks": len(arrays), "config": {"n_mels": dims, "frames": 1}})
    )
    return store


class LoaderAndResumeTests(unittest.TestCase):
    def test_load_contiguous_features_preserves_chunk_and_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            first = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            second = np.array([[7, 8, 9]], dtype=np.float32)
            result = load_contiguous_features(make_store(Path(directory), [first, second]))
            np.testing.assert_array_equal(result, np.concatenate([first, second]))
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(result.flags.c_contiguous)

    def test_load_contiguous_features_rejects_declared_row_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = make_store(Path(directory), [np.ones((2, 3), dtype=np.float32)], declared_rows=3)
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                load_contiguous_features(store)

    def test_load_contiguous_features_rejects_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = make_store(Path(directory), [np.ones((2, 4), dtype=np.float32)], dims=3)
            with self.assertRaisesRegex(ValueError, "feature dimension mismatch"):
                load_contiguous_features(store)

    def test_rng_state_round_trip_restores_python_numpy_and_torch(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        state = capture_rng_state()
        expected = (random.random(), np.random.random(), torch.rand(3))
        random.random(), np.random.random(), torch.rand(3)
        restore_rng_state(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_resident_batches_cover_each_row_once_and_keep_final_partial_batch(self):
        features = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        torch.manual_seed(21)
        batches = list(iter_resident_batches(features, batch_size=4, device=torch.device("cpu")))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        observed = torch.cat(batches)[:, 0].div(3).to(torch.int64)
        self.assertEqual(sorted(observed.tolist()), list(range(10)))


if __name__ == "__main__":
    unittest.main()
