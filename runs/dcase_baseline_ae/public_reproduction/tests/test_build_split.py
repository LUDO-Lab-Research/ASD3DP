import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_split.py"
SPEC = importlib.util.spec_from_file_location("build_split", MODULE_PATH)
build_split = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_split)


class SplitHelpersTest(unittest.TestCase):
    def test_staging_refuses_to_replace_existing_output_by_default(self):
        stage_script = MODULE_PATH.parent / "stage_split.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(stage_script),
                    "--train-manifest",
                    str(Path(directory) / "train.csv"),
                    "--test-manifest",
                    str(Path(directory) / "test.csv"),
                    "--release-root",
                    directory,
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("pass --replace-output", completed.stderr)

    def test_equal_session_quotas_sum_to_context_quota(self):
        quotas = build_split.equal_quotas(["s1", "s2", "s3"], 250)
        self.assertEqual(sum(quotas.values()), 250)
        self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)

    def test_timeline_sample_covers_ordered_range(self):
        rows = [{"clip_index": str(i)} for i in range(1, 101)]
        selected = build_split.sample_timeline(rows, 10, seed=13711)
        indices = sorted(int(row["clip_index"]) for row in selected)
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(set(indices)), 10)
        self.assertLessEqual(indices[0], 10)
        self.assertGreaterEqual(indices[-1], 91)

    def test_slow_motion_is_train_only(self):
        self.assertEqual(
            build_split.HELD_OUT_NORMAL_SESSIONS["motion_slow_no_filament"],
            (),
        )
        self.assertIn(
            "asd3dp_s0016",
            build_split.TRAIN_ONLY_NORMAL_SESSIONS,
        )

    def test_manifest_audio_path_is_relative_to_release_root(self):
        root = Path("/release/asd3dp_v0.4.1")
        row = {"clip_audio_file": "clipped/audio_event_clean/normal/sample.wav"}
        self.assertEqual(
            build_split.resolve_audio_path(root, row),
            root / "clipped/audio_event_clean/normal/sample.wav",
        )

    def test_each_context_has_250_training_rows(self):
        rows = []
        for context, sessions in {
            "print_slow_loaded": ["s1", "s2"],
            "print_fast_loaded": ["s3", "s4"],
            "motion_slow_no_filament": ["s5"],
            "motion_fast_no_filament": ["s6", "s7"],
        }.items():
            for session in sessions:
                rows.extend(
                    {
                        "protocol_axis": context,
                        "release_session_id": session,
                        "clip_index": str(i),
                    }
                    for i in range(1, 301)
                )
        selected = build_split.sample_training_rows(rows, seed=13711)
        counts = {}
        for row in selected:
            counts[row["protocol_axis"]] = counts.get(row["protocol_axis"], 0) + 1
        self.assertEqual(counts, {context: 250 for context in build_split.CONTEXTS})

    def test_balanced_group_sample_is_equal_across_sessions(self):
        rows = [
            {"release_session_id": session, "clip_index": str(i)}
            for session in ("s1", "s2", "s3")
            for i in range(1, 101)
        ]
        selected = build_split.sample_balanced_sessions(rows, 100, seed=13711)
        counts = {}
        for row in selected:
            session = row["release_session_id"]
            counts[session] = counts.get(session, 0) + 1
        self.assertEqual(sum(counts.values()), 100)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_capacity_aware_quotas_redistribute_short_session(self):
        quotas = build_split.capacity_aware_quotas({"s1": 1, "s2": 100, "s3": 100}, 100)
        self.assertEqual(sum(quotas.values()), 100)
        self.assertEqual(quotas["s1"], 1)
        self.assertLessEqual(abs(quotas["s2"] - quotas["s3"]), 1)


if __name__ == "__main__":
    unittest.main()
