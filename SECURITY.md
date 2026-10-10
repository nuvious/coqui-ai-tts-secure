# Security policy

This fork of Coqui TTS 0.22.0 exists to keep that release usable without known
vulnerabilities. It does not add features, it keeps the public Python API, the
`tts` and `tts-server` command lines, the model formats, and the supported Python
range (3.10 to 3.13). The maintained community fork at
https://github.com/idiap/coqui-ai-TTS is the right place for new development.

## Reporting a vulnerability

Open a private security advisory on the GitHub repository
(Security tab, "Report a vulnerability"). Please include the affected file or
dependency, a reproduction, and the impact you see. You can expect an
acknowledgement within a week.

## Supported versions

Only the current `dev` branch and the latest tagged release receive fixes.

## How dependencies are kept current

- `requirements*.txt` hold the version floors the package installs with; a floor
  is raised to the first patched release whenever a dependency gets a CVE.
- The training loop comes from `coqui-tts-trainer` (the maintained fork of
  `trainer`, same `trainer` import name) because `trainer` 0.0.36 does not
  install on Python 3.12 or newer.
- `requirements/lock-py3.10.txt` through `lock-py3.13.txt` pin the exact set
  used by CI and the Docker images. Regenerate them with `make lock`.
- `make audit` runs `pip-audit` on every lock file. The `security` GitHub
  workflow runs the same audit on every push and weekly, together with Bandit,
  Ruff security rules, Semgrep, CodeQL, Trivy, and zizmor.
- Dependabot opens weekly pull requests for Python packages, GitHub Actions, and
  Docker base images. A pull request is mergeable when the `tests` workflow and
  the `security` workflow are green.
- `make api-check` (griffe) and `tests/aux_tests/test_api_surface.py` fail when a
  public signature changes, which is how backwards compatibility is enforced.

## Hardening decisions

### Checkpoint loading is strict

Every checkpoint is loaded with `torch.load(weights_only=True)`. An allowlist
registered in `TTS/utils/io.py` covers this project's own configuration classes
and the numpy objects that embeddings and statistics files contain. There is no
option to disable the check.

A checkpoint that contains any other Python object fails with an error that
names the unsupported global. Seven released Tacotron2 models from 2020 and 2021
(`de/thorsten/tacotron2-DCA`, `en/ek1/tacotron2`, `en/sam/tacotron-DDC`,
`es/mai/tacotron2-DDC`, `fr/mai/tacotron2-DDC`, `nl/mai/tacotron2-DDC`,
`zh-CN/baker/tacotron2-DDC-GST`) pickle an optimizer whose state torch's
restricted unpickler cannot rebuild, so they do not load until re-saved;
`scripts/resave_legacy_checkpoint.py` is the maintainer tool for that, and the
re-saved files need to be published and referenced from `TTS/.models.json`.
Every other model in the zoo loads under the strict loader (checked with
`scripts/check_zoo_safe_globals.py` on 2026-10-10). Checkpoints written by the Trainer, the released
model zoo, speaker embedding files, and `scale_stats.npy` all load as before. If
you have an older checkpoint that fails, load it once in an isolated environment
you trust and re-save it with the Trainer; the result stores the configuration as
a plain dictionary and loads under the strict loader.

### Model downloads

Archives are extracted only after every member has been checked to stay inside
the target directory. When `.models.json` carries a `model_hash`, the downloaded
model file is verified against it. Hugging Face model references used by Bark and
Tortoise are pinned to a specific revision. All HTTP requests carry a timeout.

### XTTS and transformers 5

The XTTS streaming generator was ported to the transformers 5 generation API
because every 4.x release carries unfixed advisories. transformers 5 requires
Python 3.10 or newer, which is one reason Python 3.9 support was dropped.

### The demo server

`tts-server` listens on `127.0.0.1` by default. Pass `--host 0.0.0.0` or set
`TTS_SERVER_HOST` to expose it; the Docker images set that variable already.
Requests are capped in size and text length, the Werkzeug debugger is never
enabled, and the `style_wav` parameter can only reference files next to the
loaded model or under the working directory. Importing `TTS.server.server` no
longer parses the command line or downloads a model; the Flask application is
built by `create_app` when `main()` runs.

## Accepted risks

- `assert` statements in library code (Bandit B101, Ruff S101) are kept: the
  project uses them for argument validation throughout and removing them would
  change error behavior.
- The `random` module (B311, S311) is used for data augmentation and sampling,
  never for secrets.
- `TTS/tts/utils/monotonic_align/core.pyx` disables Cython bounds checking in
  its inner loops for performance. Inputs come from the model's own tensors, not
  from user data.
- `nltk` has an advisory with no fixed release at the time of writing; it is only
  used for tokenizing text passed to the Tortoise model.
- Python 3.9 reached end of life in October 2025 and many dependencies no longer
  publish fixed releases for it, so it is no longer supported.
