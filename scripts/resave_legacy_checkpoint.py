"""Re-save a legacy checkpoint so that it loads under the strict weights_only loader.

Released Tacotron2 checkpoints from 2020 and 2021 pickle an RAdam optimizer whose state is a
collections.defaultdict, which torch's restricted unpickler cannot rebuild. This maintainer tool
loads such a file with the unrestricted unpickler, so run it only on a checkpoint you trust and only
in an isolated environment, then publish the re-saved file. The output keeps the model weights and
the config and drops optimizer and scheduler objects, which inference never needs.
"""

import argparse
import sys
from collections import OrderedDict

import torch

from TTS.utils.io import load_fsspec

KEPT_KEYS = ("model", "config", "step", "epoch", "date", "r", "model_loss", "characters")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the legacy checkpoint.")
    parser.add_argument("output", help="Where to write the re-saved checkpoint.")
    parser.add_argument(
        "--i-trust-this-file",
        action="store_true",
        help="Acknowledge that the input is unpickled without restrictions and may execute code.",
    )
    return parser.parse_args(argv)


def sanitize(state):
    clean = {}
    for key in KEPT_KEYS:
        if key in state:
            clean[key] = state[key]
    if isinstance(clean.get("model"), dict):
        clean["model"] = OrderedDict((k, v) for k, v in clean["model"].items() if isinstance(v, torch.Tensor))
    config = clean.get("config")
    if config is not None and hasattr(config, "to_dict"):
        clean["config"] = config.to_dict()
    return clean


def main(argv=None):
    args = parse_args(argv)
    if not args.i_trust_this_file:
        print(" [!] Refusing to run without --i-trust-this-file; the input is unpickled without restrictions.")
        return 2
    state = torch.load(args.input, map_location="cpu", weights_only=False)  # nosec B614 # nosemgrep
    clean = sanitize(state)
    dropped = sorted(set(state) - set(clean))
    torch.save(clean, args.output)
    load_fsspec(args.output, map_location="cpu")
    print(f" > Wrote {args.output}; kept {sorted(clean)}; dropped {dropped}; strict load verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
