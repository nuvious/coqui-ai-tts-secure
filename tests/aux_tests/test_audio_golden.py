"""Golden-value test for the numpy audio pipeline.

``tests/data/golden/audio_*.npy`` hold the mel spectrogram, linear spectrogram
and f0 contour that ``AudioProcessor`` produced for the first two seconds of
``tests/inputs/example_1.wav`` with the audio section of
``tests/inputs/test_config.json``. The test asserts shape equality and
``np.allclose`` within 1e-3 so that a dependency bump that changes the numbers
is caught.

Regenerate the fixtures after an intentional change with::

    python -m tests.aux_tests.test_audio_golden --update
"""

import os
import sys
import unittest

import numpy as np

from tests import get_tests_data_path, get_tests_input_path
from TTS.config import read_json_with_comments
from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio.processor import AudioProcessor

GOLDEN_DIR = os.path.join(get_tests_data_path(), "golden")
CONFIG_PATH = os.path.join(get_tests_input_path(), "test_config.json")
WAV_PATH = os.path.join(get_tests_input_path(), "example_1.wav")
WAV_SECONDS = 2
FEATURES = ("melspectrogram", "spectrogram", "f0")
UPDATE_COMMAND = "python -m tests.aux_tests.test_audio_golden --update"
ATOL = 1e-3
RTOL = 1e-3


def fixture_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"audio_{name}.npy")


def build_processor() -> AudioProcessor:
    audio = read_json_with_comments(CONFIG_PATH)["audio"]
    audio.pop("audio_processor", None)
    # BaseAudioConfig supplies the pitch_fmin and pitch_fmax defaults that compute_f0 requires.
    config = BaseAudioConfig(**audio)
    return AudioProcessor(verbose=False, **config)


def load_clip(ap: AudioProcessor) -> np.ndarray:
    wav = ap.load_wav(WAV_PATH)
    return wav[: WAV_SECONDS * ap.sample_rate]


def compute_features() -> dict:
    ap = build_processor()
    wav = load_clip(ap)
    return {
        "melspectrogram": ap.melspectrogram(wav).astype(np.float32),
        "spectrogram": ap.spectrogram(wav).astype(np.float32),
        "f0": ap.compute_f0(wav).astype(np.float32),
    }


def write_fixtures() -> None:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name, value in compute_features().items():
        np.save(fixture_path(name), value)


class TestAudioGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = compute_features()

    def _assert_matches_fixture(self, name: str):
        path = fixture_path(name)
        self.assertTrue(os.path.isfile(path), f"missing fixture {path}; run `{UPDATE_COMMAND}`")
        expected = np.load(path)
        actual = self.features[name]
        self.assertEqual(actual.shape, expected.shape, f"{name} shape changed")
        self.assertTrue(np.all(np.isfinite(actual)), f"{name} contains non-finite values")
        if not np.allclose(actual, expected, atol=ATOL, rtol=RTOL):
            max_abs = float(np.max(np.abs(actual - expected)))
            self.fail(
                f"{name} drifted from fixture (max abs diff {max_abs:.6g}); if intentional run `{UPDATE_COMMAND}`"
            )

    def test_melspectrogram_matches_fixture(self):
        self._assert_matches_fixture("melspectrogram")

    def test_spectrogram_matches_fixture(self):
        self._assert_matches_fixture("spectrogram")

    def test_f0_matches_fixture(self):
        self._assert_matches_fixture("f0")

    def test_frame_counts_agree(self):
        mel, spec, f0 = (self.features[name] for name in FEATURES)
        self.assertEqual(mel.shape[1], spec.shape[1])
        self.assertEqual(f0.shape[0], mel.shape[1])


if __name__ == "__main__":
    if "--update" in sys.argv:
        sys.argv.remove("--update")
        write_fixtures()
        print(f"wrote {', '.join(fixture_path(name) for name in FEATURES)}")
    else:
        unittest.main()
