"""Golden-value test for text cleaning, phonemization and tokenization.

``tests/data/golden/tokenizer_ids.json`` records, for a fixed set of English
sentences, the cleaned text and token ids of a grapheme ``TTSTokenizer`` using
``english_cleaners`` and the phonemized text and token ids of an IPA
``TTSTokenizer`` driven by ``ESpeak("en-us")``. The test asserts exact
equality so that a change in the cleaners, the espeak backend or the character
tables is caught.

Regenerate the fixture after an intentional change with::

    python -m tests.text_tests.test_tokenizer_golden --update
"""

import json
import os
import sys
import unittest

from tests import get_tests_data_path
from TTS.tts.utils.text.characters import Graphemes, IPAPhonemes
from TTS.tts.utils.text.cleaners import english_cleaners
from TTS.tts.utils.text.phonemizers import ESpeak
from TTS.tts.utils.text.tokenizer import TTSTokenizer

FIXTURE_PATH = os.path.join(get_tests_data_path(), "golden", "tokenizer_ids.json")
UPDATE_COMMAND = "python -m tests.text_tests.test_tokenizer_golden --update"

SENTENCES = [
    "Dr. Smith paid $12.50 for 3 apples on Jan. 2nd, 1999.",
    "Hello world! How are you today?",
    "The quick brown fox jumps over the lazy dog; it's 10:45 a.m.",
    "Mr. and Mrs. Jones live at 221B Baker St., London.",
    "In 2024, roughly 60 percent of users (about 1,200 people) said yes: it works.",
]


def build_tokenizers():
    graphemes = TTSTokenizer(use_phonemes=False, characters=Graphemes(), text_cleaner=english_cleaners)
    phonemizer = ESpeak("en-us")
    phonemes = TTSTokenizer(
        use_phonemes=True, characters=IPAPhonemes(), phonemizer=phonemizer, text_cleaner=english_cleaners
    )
    return graphemes, phonemes


def build_snapshot() -> dict:
    graphemes, phonemes = build_tokenizers()
    entries = []
    for text in SENTENCES:
        cleaned = english_cleaners(text)
        entries.append(
            {
                "text": text,
                "cleaned": cleaned,
                "grapheme_ids": graphemes.text_to_ids(text),
                "phonemized": phonemes.phonemizer.phonemize(cleaned, separator=""),
                "phoneme_ids": phonemes.text_to_ids(text),
            }
        )
    return {
        "phonemizer_backend": phonemes.phonemizer.backend,
        "phonemizer_version": phonemes.phonemizer.backend_version,
        "sentences": entries,
    }


def render_snapshot(snapshot: dict) -> str:
    """Serialize the snapshot with each id list on a single line."""
    lines = [
        "{",
        f'  "phonemizer_backend": {json.dumps(snapshot["phonemizer_backend"])},',
        f'  "phonemizer_version": {json.dumps(snapshot["phonemizer_version"])},',
        '  "sentences": [',
    ]
    for index, entry in enumerate(snapshot["sentences"]):
        lines.append("    {")
        for key in ("text", "cleaned", "grapheme_ids", "phonemized", "phoneme_ids"):
            comma = "," if key != "phoneme_ids" else ""
            lines.append(f'      "{key}": {json.dumps(entry[key], ensure_ascii=False)}{comma}')
        lines.append("    }" + ("," if index < len(snapshot["sentences"]) - 1 else ""))
    lines.extend(["  ]", "}"])
    return "\n".join(lines) + "\n"


def write_fixture() -> None:
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as handle:
        handle.write(render_snapshot(build_snapshot()))


class TestTokenizerGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = build_snapshot()
        if not os.path.isfile(FIXTURE_PATH):
            raise FileNotFoundError(f"missing fixture {FIXTURE_PATH}; run `{UPDATE_COMMAND}`")
        with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
            cls.expected = json.load(handle)

    def _pairs(self):
        self.assertEqual(len(self.snapshot["sentences"]), len(self.expected["sentences"]))
        for actual, expected in zip(self.snapshot["sentences"], self.expected["sentences"]):
            self.assertEqual(actual["text"], expected["text"], "sentence list changed; regenerate the fixture")
            yield actual, expected

    def test_cleaned_text_matches_fixture(self):
        for actual, expected in self._pairs():
            with self.subTest(text=actual["text"]):
                self.assertEqual(actual["cleaned"], expected["cleaned"])

    def test_grapheme_ids_match_fixture(self):
        for actual, expected in self._pairs():
            with self.subTest(text=actual["text"]):
                self.assertEqual(actual["grapheme_ids"], expected["grapheme_ids"])

    def _require_same_phonemizer(self):
        """Phoneme output depends on the installed espeak-ng release, so only an identical release is compared."""
        actual = (self.snapshot["phonemizer_backend"], self.snapshot["phonemizer_version"])
        expected = (self.expected["phonemizer_backend"], self.expected.get("phonemizer_version"))
        if actual != expected:
            self.skipTest(f"fixture was generated with {expected}, installed phonemizer is {actual}")

    def test_phonemized_text_matches_fixture(self):
        self._require_same_phonemizer()
        for actual, expected in self._pairs():
            with self.subTest(text=actual["text"]):
                self.assertEqual(actual["phonemized"], expected["phonemized"])

    def test_phoneme_ids_match_fixture(self):
        self._require_same_phonemizer()
        for actual, expected in self._pairs():
            with self.subTest(text=actual["text"]):
                self.assertEqual(actual["phoneme_ids"], expected["phoneme_ids"])


if __name__ == "__main__":
    if "--update" in sys.argv:
        sys.argv.remove("--update")
        write_fixture()
        print(f"wrote {FIXTURE_PATH}")
    else:
        unittest.main()
