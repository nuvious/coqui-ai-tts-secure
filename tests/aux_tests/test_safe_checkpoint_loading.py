import os
import pickle
import re
import unittest
from collections import OrderedDict

import numpy as np
import torch

from tests import get_tests_data_path, get_tests_input_path, get_tests_output_path
from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_fsspec, register_safe_globals, safe_np_load_dict, safe_torch_load

TTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "TTS")


class NotAllowed:
    """A class that is deliberately outside the checkpoint allowlist."""

    def __init__(self, value=0):
        self.value = value

    def __reduce__(self):
        return (NotAllowed, (self.value,))


def _write_npy_object(path, obj):
    array = np.empty((), dtype=object)
    array[()] = obj
    np.save(path, array, allow_pickle=True)


class TestSafeCheckpointLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_dir = os.path.join(get_tests_output_path(), "safe_checkpoint_loading")
        os.makedirs(cls.out_dir, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.out_dir, name)

    def test_trainer_style_checkpoint(self):
        config = VitsConfig(datasets=[BaseDatasetConfig()], characters=CharactersConfig())
        state = {
            "config": config.to_dict(),
            "model": OrderedDict([("layer.weight", torch.randn(4, 3)), ("layer.bias", torch.zeros(4))]),
            "optimizer": {"state": {}, "param_groups": [{"lr": 1e-3, "params": [0, 1]}]},
            "step": 100,
            "epoch": 2,
            "lr": 1e-3,
            "date": "2024-01-01 00:00:00",
        }
        path = self._path("trainer_style.pth")
        torch.save(state, path)
        loaded = load_fsspec(path, map_location="cpu")
        self.assertIsInstance(loaded["config"], dict)
        self.assertEqual(loaded["config"]["model"], "vits")
        self.assertIsInstance(loaded["model"], OrderedDict)
        self.assertTrue(torch.equal(loaded["model"]["layer.weight"], state["model"]["layer.weight"]))
        self.assertEqual(loaded["optimizer"]["param_groups"][0]["lr"], 1e-3)
        self.assertEqual(loaded["step"], 100)
        self.assertEqual(loaded["date"], "2024-01-01 00:00:00")

    def test_radam_optimizer_instance_checkpoint(self):
        from TTS.utils.radam import RAdam  # pylint: disable=import-outside-toplevel

        optimizer = RAdam([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
        path = self._path("radam_instance.pth")
        torch.save({"model": OrderedDict([("w", torch.zeros(2))]), "optimizer": optimizer}, path)
        loaded = load_fsspec(path, map_location="cpu")
        self.assertIsInstance(loaded["optimizer"], RAdam)
        self.assertEqual(loaded["optimizer"].param_groups[0]["lr"], 1e-3)

    def test_config_instance_checkpoint(self):
        config = VitsConfig(datasets=[BaseDatasetConfig(formatter="ljspeech")], characters=CharactersConfig())
        path = self._path("config_instance.pth")
        torch.save({"config": config, "model": OrderedDict([("w", torch.ones(2))])}, path)
        loaded = load_fsspec(path, map_location="cpu")
        self.assertIsInstance(loaded["config"], VitsConfig)
        self.assertIsInstance(loaded["config"].datasets[0], BaseDatasetConfig)
        self.assertEqual(loaded["config"].datasets[0].formatter, "ljspeech")
        self.assertIsInstance(loaded["config"].characters, CharactersConfig)
        self.assertEqual(loaded["config"].to_dict(), config.to_dict())

    def test_numpy_checkpoint(self):
        state = {
            "array": np.arange(6, dtype=np.float32).reshape(2, 3),
            "mean": np.float64(1.5),
            "std": np.float64(0.25),
            "count": np.int64(7),
        }
        path = self._path("numpy_values.pth")
        torch.save(state, path)
        loaded = load_fsspec(path, map_location="cpu")
        self.assertIsInstance(loaded["array"], np.ndarray)
        np.testing.assert_array_equal(loaded["array"], state["array"])
        self.assertIsInstance(loaded["mean"], np.float64)
        self.assertEqual(loaded["mean"], 1.5)
        self.assertIsInstance(loaded["count"], np.int64)

    def test_d_vector_checkpoint(self):
        state = {
            f"speaker_{i}.wav": {"name": f"speaker_{i}", "embedding": np.random.rand(8).tolist()} for i in range(3)
        }
        path = self._path("d_vectors.pth")
        torch.save(state, path)
        loaded = load_fsspec(path, map_location="cpu")
        self.assertEqual(loaded.keys(), state.keys())
        self.assertEqual(loaded["speaker_0.wav"]["embedding"], state["speaker_0.wav"]["embedding"])

    def test_safe_torch_load_file_object(self):
        path = self._path("file_object.pth")
        torch.save({"w": torch.ones(2)}, path)
        with open(path, "rb") as f:
            loaded = safe_torch_load(f, map_location="cpu")
        self.assertTrue(torch.equal(loaded["w"], torch.ones(2)))

    def test_pickle_module_is_rejected(self):
        path = self._path("file_object.pth")
        torch.save({"w": torch.ones(2)}, path)
        with self.assertRaises(ValueError):
            load_fsspec(path, pickle_module=pickle)

    def test_real_fixtures(self):
        speakers = load_fsspec(os.path.join(get_tests_data_path(), "dummy_speakers.pth"))
        self.assertGreater(len(speakers), 0)
        first = next(iter(speakers.values()))
        self.assertIn("embedding", first)

        stats = safe_np_load_dict(os.path.join(get_tests_data_path(), "ljspeech", "f0_cache", "pitch_stats.npy"))
        self.assertIsInstance(stats["mean"], np.floating)
        self.assertIsInstance(stats["std"], np.floating)

        ap = AudioProcessor(**BaseAudioConfig(mel_fmax=8000, pitch_fmax=640, pitch_fmin=1))
        mel_mean, mel_std, linear_mean, linear_std, stats_config = ap.load_stats(
            os.path.join(get_tests_input_path(), "scale_stats.npy")
        )
        self.assertIsInstance(mel_mean, np.ndarray)
        self.assertIsInstance(mel_std, np.ndarray)
        self.assertIsInstance(linear_mean, np.ndarray)
        self.assertIsInstance(linear_std, np.ndarray)
        self.assertIsInstance(stats_config, dict)

    def test_unlisted_global_in_checkpoint_raises(self):
        path = self._path("unlisted_global.pth")
        torch.save({"payload": NotAllowed(3), "w": torch.ones(1)}, path)
        with self.assertRaises(RuntimeError) as ctx:
            load_fsspec(path, map_location="cpu")
        self.assertIn("Unsupported global", str(ctx.exception))
        self.assertIn("NotAllowed", str(ctx.exception))
        self.assertIn("Trainer", str(ctx.exception))

    def test_unlisted_global_in_npy_raises(self):
        path = self._path("unlisted_global.npy")
        _write_npy_object(path, {"mean": NotAllowed(1)})
        with self.assertRaises(pickle.UnpicklingError) as ctx:
            safe_np_load_dict(path)
        self.assertIn("NotAllowed", str(ctx.exception))

    def test_npy_must_hold_dict(self):
        path = self._path("not_a_dict.npy")
        _write_npy_object(path, [1, 2, 3])
        with self.assertRaises(ValueError):
            safe_np_load_dict(path)
        plain = self._path("plain_array.npy")
        np.save(plain, np.zeros(3))
        with self.assertRaises(ValueError):
            safe_np_load_dict(plain)

    def test_register_safe_globals_is_idempotent(self):
        register_safe_globals()
        before = len(torch.serialization.get_safe_globals())
        register_safe_globals()
        self.assertEqual(len(torch.serialization.get_safe_globals()), before)
        registered = {
            entry[0] if isinstance(entry, tuple) else entry for entry in torch.serialization.get_safe_globals()
        }
        self.assertIn(VitsConfig, registered)
        self.assertIn(CharactersConfig, registered)
        self.assertIn(np.ndarray, registered)
        self.assertIn(np.dtype, registered)

    def test_no_unsafe_loaders_in_source(self):
        torch_load = re.compile(r"\btorch\.load\(")
        np_load = re.compile(r"\bnp\.load\(|\bnumpy\.load\(")
        offenders = []
        for dirpath, _, filenames in os.walk(TTS_ROOT):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                file_path = os.path.join(dirpath, filename)
                relative = os.path.relpath(file_path, TTS_ROOT)
                with open(file_path, encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        if torch_load.search(line) and relative != os.path.join("utils", "io.py"):
                            offenders.append(f"{relative}:{lineno}: {line.strip()}")
                        if np_load.search(line) and "allow_pickle=True" in line:
                            offenders.append(f"{relative}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], "unsafe deserialization calls found:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
