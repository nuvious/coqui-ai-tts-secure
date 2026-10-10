import os

import requests
from tqdm import tqdm

REQUEST_TIMEOUT = (10, 300)

DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tortoise", "models")
MODELS_DIR = os.environ.get("TORTOISE_MODELS_DIR", DEFAULT_MODELS_DIR)
MODELS_DIR = "/data/speech_synth/models/"
MODELS = {
    "autoregressive.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth",
    "classifier.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth",
    "clvp2.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth",
    "diffusion_decoder.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth",
    "vocoder.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth",
    "rlg_auto.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth",
    "rlg_diffuser.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth",
}


def download_models(specific_models=None):
    """
    Call to download all the models that Tortoise uses.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    for model_name, url in MODELS.items():
        if specific_models is not None and model_name not in specific_models:
            continue
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            continue
        print(f"Downloading {model_name} from {url}...")
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, total=total or None) as t,
                open(model_path, "wb") as f,
            ):
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    t.update(len(chunk))
        print("Done.")


def get_model_path(model_name, models_dir=MODELS_DIR):
    """
    Get path to given model, download it if it doesn't exist.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path) and models_dir == MODELS_DIR:
        download_models([model_name])
    return model_path
