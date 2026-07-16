import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))

from stage_asd3dp_pairs import channel_path, write_pair


class StagePairTests(unittest.TestCase):
    def test_channel_path_maps_front_clip_to_rear_clip(self):
        front = Path("clipped/audio_event_clean/normal/x_ch2_normal.wav")
        self.assertEqual(
            channel_path(front, "ch1"),
            Path("clipped/audio_event_clean/normal/x_ch1_normal.wav"),
        )

    def test_channel_path_maps_front_clip_to_right_and_left(self):
        front = Path("clipped/audio_event_clean/normal/x_ch2_normal.wav")
        self.assertEqual(channel_path(front, "ch3").name, "x_ch3_normal.wav")
        self.assertEqual(channel_path(front, "ch4").name, "x_ch4_normal.wav")

    def test_write_pair_resamples_and_places_rear_before_front(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rear = root / "rear.wav"
            front = root / "front.wav"
            out = root / "pair.wav"
            sf.write(rear, np.full(48000, 0.25, dtype=np.float32), 48000)
            sf.write(front, np.full(48000, -0.5, dtype=np.float32), 48000)
            write_pair(rear, front, out, target_sr=16000)
            wave, sr = sf.read(out, always_2d=True)
            self.assertEqual(sr, 16000)
            self.assertEqual(wave.shape, (16000, 2))
            self.assertAlmostEqual(float(wave[8000, 0]), 0.25, places=3)
            self.assertAlmostEqual(float(wave[8000, 1]), -0.5, places=3)


if __name__ == "__main__":
    unittest.main()
