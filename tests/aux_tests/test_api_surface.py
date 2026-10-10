"""Guardrail test for the public API surface of the library.

The snapshot in ``tests/data/api_surface.json`` records the public methods and
signatures of ``TTS.api.TTS``, the option names of the ``tts`` and
``tts-server`` command lines, a handful of load-bearing signatures from the
utility layer and the model-zoo names read from ``TTS/.models.json``. The test
fails with a unified diff whenever the live surface drifts from the snapshot.

Regenerate the snapshot after an intentional API change with::

    python -m tests.aux_tests.test_api_surface --update
"""

import difflib
import inspect
import json
import os
import re
import sys
import unittest

from tests import get_tests_data_path
from TTS.api import TTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

FIXTURE_PATH = os.path.join(get_tests_data_path(), "api_surface.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI_SOURCE = os.path.join(REPO_ROOT, "TTS", "bin", "synthesize.py")
SERVER_SOURCE = os.path.join(REPO_ROOT, "TTS", "server", "server.py")
UPDATE_COMMAND = "python -m tests.aux_tests.test_api_surface --update"

# Matches the first option string of each add_argument call, across line breaks.
ADD_ARGUMENT_RE = re.compile(r"""add_argument\(\s*["'](-{1,2}[\w-]+)["']""")


def _signature(obj) -> str:
    return str(inspect.signature(obj))


def _public_members(cls) -> dict:
    """Describe the public callables and properties defined directly on a class."""
    members = {}
    for name, obj in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(obj, property):
            members[name] = {"kind": "property"}
        elif isinstance(obj, staticmethod):
            members[name] = {"kind": "staticmethod", "signature": _signature(obj.__func__)}
        elif isinstance(obj, classmethod):
            members[name] = {"kind": "classmethod", "signature": _signature(obj.__func__)}
        elif inspect.isfunction(obj):
            members[name] = {"kind": "method", "signature": _signature(obj)}
    return members


def _argparse_options(source_path: str) -> list:
    with open(source_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return sorted(set(ADD_ARGUMENT_RE.findall(source)))


def build_snapshot() -> dict:
    manager = ModelManager(progress_bar=False, verbose=False)
    return {
        "TTS.api.TTS": _public_members(TTS),
        "TTS.bin.synthesize.options": _argparse_options(CLI_SOURCE),
        "TTS.server.server.options": _argparse_options(SERVER_SOURCE),
        "TTS.utils.synthesizer.Synthesizer": {
            "__init__": _signature(Synthesizer.__init__),
            "tts": _signature(Synthesizer.tts),
        },
        "TTS.utils.manage.ModelManager": _public_members(ModelManager),
        "TTS.utils.audio.AudioProcessor.__init__": _signature(AudioProcessor.__init__),
        "TTS.tts.utils.text.tokenizer.TTSTokenizer.__init__": _signature(TTSTokenizer.__init__),
        "model_zoo": sorted(manager.list_models()),
    }


def render_snapshot(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def write_fixture() -> None:
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as handle:
        handle.write(render_snapshot(build_snapshot()))


class TestApiSurface(unittest.TestCase):
    def test_snapshot_matches_fixture(self):
        self.assertTrue(os.path.isfile(FIXTURE_PATH), f"missing fixture {FIXTURE_PATH}; run `{UPDATE_COMMAND}`")
        with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
            expected = handle.read()
        actual = render_snapshot(build_snapshot())
        if actual != expected:
            diff = "\n".join(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="tests/data/api_surface.json",
                    tofile="live API surface",
                    lineterm="",
                )
            )
            self.fail(f"public API surface changed; if intentional run `{UPDATE_COMMAND}`\n{diff}")

    def test_snapshot_is_populated(self):
        snapshot = build_snapshot()
        self.assertIn("tts", snapshot["TTS.api.TTS"])
        self.assertIn("tts_to_file", snapshot["TTS.api.TTS"])
        self.assertIn("--text", snapshot["TTS.bin.synthesize.options"])
        self.assertIn("--port", snapshot["TTS.server.server.options"])
        self.assertIn("tts_models/en/ljspeech/tacotron2-DDC", snapshot["model_zoo"])


if __name__ == "__main__":
    if "--update" in sys.argv:
        sys.argv.remove("--update")
        write_fixture()
        print(f"wrote {FIXTURE_PATH}")
    else:
        unittest.main()
