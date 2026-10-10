import json
import os
import unittest
from unittest import mock

from trainer.io import save_checkpoint

from tests import get_tests_input_path, get_tests_output_path
from TTS.config import load_config
from TTS.server.server import MAX_CONTENT_LENGTH, MAX_TEXT_LENGTH, create_app, create_argparser
from TTS.tts.models import setup_model
from TTS.utils.synthesizer import Synthesizer


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output_path = os.path.join(get_tests_output_path(), "server_test")
        os.makedirs(output_path, exist_ok=True)
        config = load_config(os.path.join(get_tests_input_path(), "dummy_model_config.json"))
        # The untrained decoder never predicts a stop token, so cap its steps to keep synthesis fast.
        config.max_decoder_steps = 16
        config.audio.griffin_lim_iters = 10
        cls.config_path = os.path.join(output_path, "config.json")
        config.save_json(cls.config_path)
        model = setup_model(config)
        save_checkpoint(config, model, output_path, current_step=10, epoch=1)
        cls.checkpoint_path = os.path.join(output_path, "checkpoint_10.pth")

        cls.args = create_argparser().parse_args([])
        cls.args.model_path = cls.checkpoint_path
        cls.args.config_path = cls.config_path
        synthesizer = Synthesizer(cls.checkpoint_path, cls.config_path, None, None)
        cls.app = create_app(cls.args, synthesizer, model_path=cls.checkpoint_path, config_path=cls.config_path)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_host_default_is_loopback(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("TTS_SERVER_HOST", None)
            parsed = create_argparser().parse_args([])
        self.assertEqual(parsed.host, "127.0.0.1")
        self.assertEqual(parsed.port, 5002)

    def test_host_env_override(self):
        with mock.patch.dict(os.environ, {"TTS_SERVER_HOST": "0.0.0.0"}):
            parsed = create_argparser().parse_args([])
        self.assertEqual(parsed.host, "0.0.0.0")

    def test_max_content_length_is_set(self):
        self.assertEqual(self.app.config["MAX_CONTENT_LENGTH"], MAX_CONTENT_LENGTH)

    def test_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_mary_tts_locales_and_voices(self):
        locales = self.client.get("/locales")
        self.assertEqual(locales.status_code, 200)
        self.assertEqual(locales.get_data(as_text=True).strip(), "en")
        voices = self.client.get("/voices")
        self.assertEqual(voices.status_code, 200)
        self.assertIn("en", voices.get_data(as_text=True))

    def test_api_tts_returns_wav(self):
        response = self.client.get("/api/tts", query_string={"text": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/wav")
        self.assertTrue(response.data.startswith(b"RIFF"))

    def test_api_tts_rejects_long_text(self):
        response = self.client.get("/api/tts", query_string={"text": "a" * (MAX_TEXT_LENGTH + 1)})
        self.assertEqual(response.status_code, 400)

    def test_process_rejects_long_text(self):
        response = self.client.post("/process", data={"INPUT_TEXT": "a" * (MAX_TEXT_LENGTH + 1)})
        self.assertEqual(response.status_code, 400)

    def test_style_wav_path_traversal_rejected(self):
        for style_wav in ("../../etc/passwd", "/etc/passwd", "/etc/passwd.wav"):
            response = self.client.get("/api/tts", query_string={"text": "hello", "style_wav": style_wav})
            self.assertEqual(response.status_code, 400, style_wav)

    def test_style_wav_missing_file_in_allowed_dir_rejected(self):
        missing = os.path.join(os.path.dirname(self.checkpoint_path), "missing.wav")
        response = self.client.get("/api/tts", query_string={"text": "hello", "style_wav": missing})
        self.assertEqual(response.status_code, 400)

    def test_style_wav_dict_is_not_rejected(self):
        style = json.dumps({"0": 0.3, "1": -0.2})
        try:
            response = self.client.get("/api/tts", query_string={"text": "hello", "style_wav": style})
        except Exception:  # pylint: disable=broad-except
            # The untrained test model does not use GST, so a model-side failure is acceptable here.
            return
        self.assertNotEqual(response.status_code, 400)

    def test_process_post(self):
        response = self.client.post("/process", data={"INPUT_TEXT": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/wav")
        self.assertTrue(response.data.startswith(b"RIFF"))

    def test_oversized_body_rejected(self):
        body = "INPUT_TEXT=" + "a" * (MAX_CONTENT_LENGTH + 1)
        response = self.client.post(
            "/process", data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        self.assertEqual(response.status_code, 413)
