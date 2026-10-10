"""Download every model in the zoo and report any checkpoint global outside the weights_only allowlist."""

import argparse
import os
import sys

import torch

from TTS.utils.io import load_fsspec, register_safe_globals
from TTS.utils.manage import ModelManager

CHECKPOINT_SUFFIXES = (".pth", ".pt", ".pth.tar")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download every model listed by ModelManager and check each checkpoint file with "
            "torch.serialization.get_unsafe_globals_in_checkpoint after registering the project allowlist. "
            "Models whose name contains 'xtts' are skipped unless COQUI_TOS_AGREED=1 is set."
        )
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory to download models into (default: TTS user data dir)."
    )
    parser.add_argument("--filter", default=None, help="Only check models whose name contains this substring.")
    parser.add_argument(
        "--skip-download", action="store_true", help="Only scan files already present in the download dir."
    )
    return parser.parse_args(argv)


def find_checkpoints(root):
    for dirpath, _, filenames in os.walk(root):
        # Archives built on macOS carry AppleDouble metadata files that are not checkpoints.
        if "__MACOSX" in dirpath.split(os.sep):
            continue
        for filename in filenames:
            if filename.endswith(CHECKPOINT_SUFFIXES) and not filename.startswith("._"):
                yield os.path.join(dirpath, filename)


def download_models(manager, model_names):
    for name in model_names:
        print(f" > Downloading {name}")
        try:
            manager.download_model(name)
        except Exception as e:  # pylint: disable=broad-except
            print(f" ! Failed to download {name}: {e}")


def check_file(path):
    """Return the unlisted globals of a checkpoint, falling back to a strict load for legacy (non zip) files."""
    try:
        return sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        load_fsspec(path, map_location="cpu")
        return []
    except Exception as e:  # pylint: disable=broad-except
        return [f"<error: {str(e).splitlines()[0][:160]}>"]


def main(argv=None):
    args = parse_args(argv)
    manager = ModelManager(output_prefix=args.output_dir, progress_bar=True)
    model_names = manager.list_models()
    if args.filter:
        model_names = [name for name in model_names if args.filter in name]
    if os.environ.get("COQUI_TOS_AGREED") != "1":
        model_names = [name for name in model_names if "xtts" not in name.lower()]
    if not args.skip_download:
        download_models(manager, model_names)

    register_safe_globals()
    rows = [(path, check_file(path)) for path in sorted(find_checkpoints(manager.output_prefix))]
    width = max((len(os.path.relpath(path, manager.output_prefix)) for path, _ in rows), default=10)
    print(f"{'checkpoint':<{width}}  unlisted globals")
    for path, unsafe in rows:
        print(f"{os.path.relpath(path, manager.output_prefix):<{width}}  {', '.join(unsafe) if unsafe else 'none'}")
    failures = [path for path, unsafe in rows if unsafe]
    print(f"\n{len(rows)} checkpoint files scanned, {len(failures)} with unlisted globals")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
