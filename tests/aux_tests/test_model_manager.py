import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from TTS.utils.download import extract_archive, safe_extract_archive
from TTS.utils.manage import ModelManager

MODEL_BYTES = b"fake checkpoint bytes"
CONFIG_BYTES = json.dumps({"model": "fake"}).encode("utf-8")


class FakeResponse:
    """Minimal stand-in for a streamed requests response."""

    def __init__(self, payload: bytes, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, block_size):
        for start in range(0, len(self.payload), block_size):
            yield self.payload[start : start + block_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def make_tar(entries, links=()):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in entries:
            if data is None:
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                t.addfile(info)
            else:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            t.addfile(info)
    return buf.getvalue()


def write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


class TestArchiveDownloads(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.out = os.path.join(self.root, "out")
        os.makedirs(self.out)

    def tearDown(self):
        self._tmp.cleanup()

    def test_zip_download_extracts_to_root(self):
        payload = make_zip(
            [("nested/", b""), ("nested/model_file.pth", MODEL_BYTES), ("nested/config.json", CONFIG_BYTES)]
        )
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(payload)) as get:
            ModelManager._download_zip_file("https://example.invalid/model.zip", self.out, False)
        self.assertEqual(get.call_count, 1)
        self.assertIsNotNone(get.call_args.kwargs.get("timeout"))
        self.assertEqual(sorted(os.listdir(self.out)), ["config.json", "model_file.pth"])
        with open(os.path.join(self.out, "model_file.pth"), "rb") as f:
            self.assertEqual(f.read(), MODEL_BYTES)

    def test_tar_download_extracts_to_root(self):
        payload = make_tar([("eng", None), ("eng/model.pth", MODEL_BYTES), ("eng/config.json", CONFIG_BYTES)])
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(payload)) as get:
            ModelManager._download_tar_file("https://example.invalid/eng.tar.gz", self.out, False)
        self.assertIsNotNone(get.call_args.kwargs.get("timeout"))
        self.assertEqual(sorted(os.listdir(self.out)), ["config.json", "model.pth"])

    def test_zip_traversal_rejected(self):
        payload = make_zip([("../evil.txt", b"evil"), ("model_file.pth", MODEL_BYTES)])
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(payload)):
            with self.assertRaises(ValueError) as ctx:
                ModelManager._download_zip_file("https://example.invalid/model.zip", self.out, False)
        self.assertIn("evil.txt", str(ctx.exception))
        self.assertFalse(os.path.exists(os.path.join(self.root, "evil.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "model_file.pth")))

    def test_tar_absolute_member_rejected(self):
        evil_target = os.path.join(self.root, "evil.txt")
        payload = make_tar([("eng", None), (evil_target, b"evil"), ("eng/model.pth", MODEL_BYTES)])
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(payload)):
            with self.assertRaises(ValueError) as ctx:
                ModelManager._download_tar_file("https://example.invalid/eng.tar.gz", self.out, False)
        self.assertIn("evil.txt", str(ctx.exception))
        self.assertFalse(os.path.exists(evil_target))
        self.assertFalse(os.path.exists(os.path.join(self.out, "eng")))

    def test_safe_extract_rejects_symlink_outside(self):
        payload = make_tar([("data/ok.txt", b"ok")], links=[("data/escape", "../../outside")])
        archive = os.path.join(self.root, "links.tar.gz")
        write_bytes(archive, payload)
        with self.assertRaises(ValueError) as ctx:
            safe_extract_archive(archive, self.out)
        self.assertIn("escape", str(ctx.exception))
        self.assertEqual(os.listdir(self.out), [])

    def test_safe_extract_allows_symlink_inside(self):
        payload = make_tar([("data/ok.txt", b"ok")], links=[("data/alias", "ok.txt")])
        archive = os.path.join(self.root, "links.tar.gz")
        write_bytes(archive, payload)
        names = safe_extract_archive(archive, self.out)
        self.assertEqual(names, ["data/ok.txt", "data/alias"])
        self.assertTrue(os.path.islink(os.path.join(self.out, "data", "alias")))

    def test_extract_archive_rejects_traversal(self):
        zip_path = os.path.join(self.root, "evil.zip")
        write_bytes(zip_path, make_zip([("../evil.txt", b"evil")]))
        with self.assertRaises(ValueError):
            extract_archive(zip_path, self.out)
        self.assertFalse(os.path.exists(os.path.join(self.root, "evil.txt")))

        tar_path = os.path.join(self.root, "evil.tar.gz")
        write_bytes(tar_path, make_tar([("../evil.txt", b"evil")]))
        with self.assertRaises(ValueError):
            extract_archive(tar_path, self.out)
        self.assertFalse(os.path.exists(os.path.join(self.root, "evil.txt")))

    def test_extract_archive_still_extracts_clean_archives(self):
        zip_path = os.path.join(self.root, "ok.zip")
        write_bytes(zip_path, make_zip([("a/b.txt", b"b")]))
        files = extract_archive(zip_path, self.out)
        self.assertEqual(files, ["a/b.txt"])
        self.assertTrue(os.path.isfile(os.path.join(self.out, "a", "b.txt")))


class TestModelManagerHash(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.payload = make_zip([("model_file.pth", MODEL_BYTES), ("config.json", CONFIG_BYTES)])
        self.model_dir = os.path.join(self.root, "tts", "tts_models--en--test--fake")

    def tearDown(self):
        self._tmp.cleanup()

    def _manager(self, model_hash):
        models = {
            "tts_models": {
                "en": {
                    "test": {
                        "fake": {
                            "github_rls_url": "https://example.invalid/fake.zip",
                            "model_hash": model_hash,
                            "license": "mit",
                        }
                    }
                }
            }
        }
        models_file = os.path.join(self.root, ".models.json")
        with open(models_file, "w", encoding="utf-8") as f:
            json.dump(models, f)
        return ModelManager(models_file=models_file, output_prefix=self.root, verbose=False)

    def test_download_with_matching_md5(self):
        good_hash = hashlib.md5(MODEL_BYTES, usedforsecurity=False).hexdigest()
        manager = self._manager(good_hash)
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(self.payload)) as get:
            model_path, config_path, _ = manager.download_model("tts_models/en/test/fake")
            self.assertEqual(get.call_count, 1)
            manager.download_model("tts_models/en/test/fake")
            self.assertEqual(get.call_count, 1)
        self.assertEqual(model_path, os.path.join(self.model_dir, "model_file.pth"))
        self.assertEqual(config_path, os.path.join(self.model_dir, "config.json"))
        with open(os.path.join(self.model_dir, "hash.md5"), encoding="utf-8") as f:
            self.assertEqual(f.read(), good_hash)

    def test_download_with_matching_sha256(self):
        manager = self._manager(hashlib.sha256(MODEL_BYTES).hexdigest())
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(self.payload)):
            manager.download_model("tts_models/en/test/fake")
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, "model_file.pth")))

    def test_download_with_wrong_hash_removes_folder(self):
        manager = self._manager("0" * 32)
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(self.payload)):
            with self.assertRaises(RuntimeError) as ctx:
                manager.download_model("tts_models/en/test/fake")
        self.assertIn("mismatch", str(ctx.exception).lower())
        self.assertFalse(os.path.exists(self.model_dir))

    def test_download_without_hash_keeps_working(self):
        manager = self._manager(None)
        del manager.models_dict["tts_models"]["en"]["test"]["fake"]["model_hash"]
        with patch("TTS.utils.manage.requests.get", return_value=FakeResponse(self.payload)):
            manager.download_model("tts_models/en/test/fake")
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, "model_file.pth")))
        self.assertFalse(os.path.exists(os.path.join(self.model_dir, "hash.md5")))


if __name__ == "__main__":
    unittest.main()
