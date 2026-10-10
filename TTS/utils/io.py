import importlib
import os
import pickle
import pkgutil
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Union

import fsspec
import numpy as np
import torch
from coqpit import Coqpit

from TTS.utils.generic_utils import get_user_data_dir

CONFIG_PACKAGES = (
    "TTS.tts.configs",
    "TTS.vocoder.configs",
    "TTS.encoder.configs",
    "TTS.vc.configs",
)

CONFIG_MODULES = (
    "TTS.config.shared_configs",
    "TTS.tts.models.xtts",
    "TTS.tts.models.vits",
    "TTS.tts.models.forward_tts",
    "TTS.tts.models.align_tts",
    "TTS.tts.models.delightful_tts",
    "TTS.tts.models.tortoise",
    "TTS.tts.models.bark",
    "TTS.vocoder.models.wavegrad",
    "TTS.vocoder.models.wavernn",
    "TTS.tts.layers.xtts.trainer.gpt_trainer",
    "TTS.tts.layers.bark.model",
    "TTS.tts.layers.bark.model_fine",
)

UNSAFE_CHECKPOINT_GUIDANCE = (
    "Checkpoints are loaded with torch.load(weights_only=True) and only this project's config classes and numpy "
    "arrays are allowed; there is no option to disable this check. Re-save the checkpoint from a trusted "
    "environment with the Trainer (which stores the config as a plain dict) so that it only contains tensors, "
    "numbers, strings and containers."
)

_safe_globals_registered = False


class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def _numpy_safe_globals() -> Dict[str, Callable]:
    """Return the numpy callables a pickled ndarray or numpy scalar refers to, keyed by every module path numpy 1 and 2 use."""
    try:
        multiarray = importlib.import_module("numpy._core.multiarray")
    except ImportError:
        multiarray = importlib.import_module("numpy.core.multiarray")
    allowed = {
        "numpy.ndarray": np.ndarray,
        "numpy.dtype": np.dtype,
        "numpy.core.multiarray._reconstruct": multiarray._reconstruct,
        "numpy.core.multiarray.scalar": multiarray.scalar,
        "numpy._core.multiarray._reconstruct": multiarray._reconstruct,
        "numpy._core.multiarray.scalar": multiarray.scalar,
    }
    if hasattr(np, "dtypes"):
        for name in dir(np.dtypes):
            if name.endswith("DType"):
                allowed[f"numpy.dtypes.{name}"] = getattr(np.dtypes, name)
    return allowed


def _import_or_warn(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:  # pylint: disable=broad-except
        # A model whose optional dependencies are broken must not block loading every other checkpoint.
        warnings.warn(f"Could not import {module_name}; its config classes will not be loadable from checkpoints: {e}")
        return None


def _iter_config_modules():
    module_names = list(CONFIG_MODULES)
    for package_name in CONFIG_PACKAGES:
        package = _import_or_warn(package_name)
        if package is None:
            continue
        yield package
        module_names.extend(
            module_info.name for module_info in pkgutil.iter_modules(package.__path__, prefix=package_name + ".")
        )
    for module_name in module_names:
        module = _import_or_warn(module_name)
        if module is not None:
            yield module


def _project_safe_globals() -> Dict[str, Callable]:
    """Return every Coqpit subclass this project defines, keyed by its import path."""
    allowed = {}
    for module in _iter_config_modules():
        for attribute in vars(module).values():
            if (
                isinstance(attribute, type)
                and issubclass(attribute, Coqpit)
                and attribute.__module__.startswith("TTS.")
            ):
                allowed[f"{attribute.__module__}.{attribute.__qualname__}"] = attribute
    # TortoiseArgs holds a vocoder enum whose members wrap these classes.
    vocoder = _import_or_warn("TTS.tts.layers.tortoise.vocoder")
    if vocoder is not None:
        for attribute in (vocoder.VocType, vocoder.VocConf, vocoder.UnivNetGenerator):
            allowed[f"{attribute.__module__}.{attribute.__qualname__}"] = attribute
    # Released Tacotron2 checkpoints from 2020 and 2021 pickle the RAdam optimizer instance itself.
    radam = _import_or_warn("TTS.utils.radam")
    if radam is not None:
        allowed["TTS.utils.radam.RAdam"] = radam.RAdam
        allowed["collections.defaultdict"] = defaultdict
        allowed["builtins.dict"] = dict
    return allowed


def register_safe_globals() -> None:
    """Allow this project's config classes and numpy objects to be unpickled by torch.load(weights_only=True).

    Registration happens once per process. Each project class is also registered under the historical
    `mozilla_voice_tts` package name so checkpoints saved by that fork still load.
    """
    global _safe_globals_registered  # pylint: disable=global-statement
    if _safe_globals_registered:
        return
    safe_globals = []
    for name, callable_ in _numpy_safe_globals().items():
        safe_globals.append((callable_, name))
    for name, callable_ in _project_safe_globals().items():
        safe_globals.append((callable_, name))
        safe_globals.append((callable_, name.replace("TTS", "mozilla_voice_tts", 1)))
    torch.serialization.add_safe_globals(safe_globals)
    _safe_globals_registered = True


def safe_torch_load(f, map_location=None, **kwargs) -> Any:
    """Load a checkpoint with torch.load(weights_only=True) after registering the project allowlist.

    Args:
        f: File path or file-like object.
        map_location: Forwarded to torch.load.
        **kwargs: Keyword arguments forwarded to torch.load. `weights_only` is always True and `pickle_module`
            is rejected because it cannot be combined with `weights_only`.

    Raises:
        RuntimeError: If the checkpoint refers to a global outside the allowlist.
    """
    if "pickle_module" in kwargs:
        raise ValueError("pickle_module cannot be used: checkpoints are always loaded with weights_only=True.")
    kwargs["weights_only"] = True
    register_safe_globals()
    try:
        # weights_only is forced above, so this is the one sanctioned torch.load call.
        return torch.load(f, map_location=map_location, **kwargs)  # nosec B614 # nosemgrep
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Refusing to load checkpoint: {e}\n{UNSAFE_CHECKPOINT_GUIDANCE}") from e


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that only resolves the globals needed to rebuild numpy arrays and scalars."""

    def __init__(self, file, allowed: Dict[str, Callable]):
        super().__init__(file)
        self._allowed = allowed

    def find_class(self, module, name):
        key = f"{module}.{name}"
        if key in self._allowed:
            return self._allowed[key]
        raise pickle.UnpicklingError(
            f"Unsupported global: {key} is not allowed in a statistics file. Only numpy arrays and scalars inside a "
            "dict are accepted; regenerate the file with TTS/bin/compute_statistics.py in a trusted environment."
        )


def safe_np_load_dict(path: str) -> Dict:
    """Load a `.npy` file holding a 0-d object array that wraps a dict, without allowing arbitrary pickles.

    This replaces the pickle-enabled `np.load` path for the statistics files this project writes.

    Args:
        path: Path to the `.npy` file.

    Returns:
        The dict stored in the file.
    """
    npformat = np.lib.format
    with fsspec.open(path, "rb") as f:
        version = npformat.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = npformat.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            shape, _, dtype = npformat.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported .npy format version {version} in {path}")
        if not dtype.hasobject or shape != ():
            raise ValueError(f"{path} is not a 0-d object array; found shape {shape} and dtype {dtype}")
        array = _NumpyOnlyUnpickler(f, _numpy_safe_globals()).load()
    if not isinstance(array, np.ndarray) or array.shape != ():
        raise ValueError(f"{path} does not contain a 0-d numpy array")
    stats = array.item()
    if not isinstance(stats, dict):
        raise ValueError(f"{path} does not contain a dict; found {type(stats).__name__}")
    return stats


def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Checkpoints are always loaded with `weights_only=True`; see `safe_torch_load`.

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return safe_torch_load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return safe_torch_load(f, map_location=map_location, **kwargs)


def load_checkpoint(
    model, checkpoint_path, use_cuda=False, eval=False, cache=False
):  # pylint: disable=redefined-builtin
    state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
    model.load_state_dict(state["model"])
    if use_cuda:
        model.cuda()
    if eval:
        model.eval()
    return model, state
