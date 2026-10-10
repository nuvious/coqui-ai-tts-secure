# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer

import os.path
import shutil

import huggingface_hub
import requests

REQUEST_TIMEOUT = (10, 300)
# Pinning the tokenizer repo to a commit keeps a later upstream push from silently changing what gets loaded.
TOKENIZER_REPO_REVISION = "c26e70f3311c6973ca86511dd18b6a8ee073e830"


class HubertManager:
    @staticmethod
    def make_sure_hubert_installed(
        download_url: str = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt", model_path: str = ""
    ):
        if not os.path.isfile(model_path):
            print("Downloading HuBERT base model")
            with requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            print("Downloaded HuBERT")
            return model_path
        return None

    @staticmethod
    def make_sure_tokenizer_installed(
        model: str = "quantifier_hubert_base_ls960_14.pth",
        repo: str = "GitMylo/bark-voice-cloning",
        model_path: str = "",
    ):
        model_dir = os.path.dirname(model_path)
        if not os.path.isfile(model_path):
            print("Downloading HuBERT custom tokenizer")
            huggingface_hub.hf_hub_download(
                repo,
                model,
                revision=TOKENIZER_REPO_REVISION,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )
            shutil.move(os.path.join(model_dir, model), model_path)
            print("Downloaded tokenizer")
            return model_path
        return None
