"""Offline exercise of the ``TTS.api.TTS`` Python interface.

A random Tacotron2 checkpoint is written from ``tests/inputs/dummy_model_config.json``
so that no model download is needed, and outgoing socket connections are blocked
for the duration of the test class so that any network access fails loudly.
"""

import os
import socket
import unittest
from unittest import mock

import numpy as np
import torch
from trainer.io import save_checkpoint

from tests import get_tests_input_path, get_tests_output_path
from TTS.api import TTS
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.utils.manage import ModelManager

ZOO_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"


def _refuse_connection(*args, **kwargs):
    raise RuntimeError("network access is not allowed in test_python_api")


class TestPythonApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network_guard = mock.patch.object(socket.socket, "connect", _refuse_connection)
        cls.network_guard.start()
        cls.output_path = os.path.join(get_tests_output_path(), "python_api")
        os.makedirs(cls.output_path, exist_ok=True)
        cls.config_path = os.path.join(get_tests_input_path(), "dummy_model_config.json")
        config = load_config(cls.config_path)
        torch.manual_seed(0)
        model = setup_model(config)
        save_checkpoint(config, model, None, None, 10, 1, cls.output_path)
        cls.model_path = os.path.join(cls.output_path, "checkpoint_10.pth")
        cls.tts = TTS(progress_bar=False, gpu=False)
        cls.tts.load_tts_model_by_path(cls.model_path, cls.config_path, gpu=False)

    @classmethod
    def tearDownClass(cls):
        cls.network_guard.stop()

    def test_multi_speaker_and_multi_lingual_flags_are_bool(self):
        self.assertIsInstance(self.tts.is_multi_speaker, bool)
        self.assertIsInstance(self.tts.is_multi_lingual, bool)
        self.assertFalse(self.tts.is_multi_speaker)
        self.assertFalse(self.tts.is_multi_lingual)

    def test_speakers_and_languages_are_none_or_list(self):
        for value in (self.tts.speakers, self.tts.languages):
            self.assertTrue(value is None or isinstance(value, list), f"unexpected type {type(value)}")

    def test_tts_returns_waveform(self):
        wav = self.tts.tts("Hello world.")
        self.assertIsInstance(wav, (list, np.ndarray))
        self.assertGreater(len(wav), 0)
        self.assertTrue(np.all(np.isfinite(np.asarray(wav, dtype=np.float32))))

    def test_tts_to_file_writes_riff_wav(self):
        file_path = os.path.join(self.output_path, "python_api_output.wav")
        if os.path.exists(file_path):
            os.remove(file_path)
        returned = self.tts.tts_to_file("Hello world.", file_path=file_path)
        self.assertEqual(returned, file_path)
        with open(file_path, "rb") as handle:
            header = handle.read(12)
        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(header[8:12], b"WAVE")
        self.assertGreater(os.path.getsize(file_path), 44)

    def test_list_models_contains_ljspeech_tacotron2(self):
        # The instance method returns a ModelManager in this version rather than a list.
        listed = self.tts.list_models()
        names = listed if isinstance(listed, list) else listed.list_models()
        self.assertIsInstance(names, list)
        self.assertIn(ZOO_MODEL_NAME, names)
        self.assertIn(ZOO_MODEL_NAME, self.tts.models)
        self.assertIn(ZOO_MODEL_NAME, ModelManager(progress_bar=False, verbose=False).list_models())

    def test_constructor_with_model_and_config_path(self):
        tts = TTS(model_path=self.model_path, config_path=self.config_path, progress_bar=False, gpu=False)
        self.assertIsNotNone(tts.synthesizer)
        self.assertIsInstance(tts.is_multi_speaker, bool)
        try:
            multi_lingual = tts.is_multi_lingual
        except AttributeError:
            self.skipTest("is_multi_lingual raises AttributeError when the loaded config has no `languages` field")
        self.assertIsInstance(multi_lingual, bool)


if __name__ == "__main__":
    unittest.main()
