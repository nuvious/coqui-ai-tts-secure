import numpy as np
import soundfile as sf
import torch


def load_audio_tensor(path: str):
    """Load an audio file as a float32 tensor shaped [channels, samples] in [-1, 1], like torchaudio.load did.

    torchaudio 2.9 moved file decoding into the optional torchcodec package, so soundfile is used instead.

    Args:
        path (str): Path to the audio file.

    Returns:
        Tuple[torch.Tensor, int]: Waveform and sample rate.
    """
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sample_rate)


def save_audio_tensor(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a float tensor shaped [channels, samples] or [samples] as a 32-bit float WAV, like torchaudio.save did.

    Args:
        path (str): Output path.
        waveform (torch.Tensor): Audio samples in [-1, 1].
        sample_rate (int): Sample rate in Hz.
    """
    data = waveform.detach().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    sf.write(path, data.T, sample_rate, subtype="FLOAT")
