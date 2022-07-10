import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_spectrogram(spectrogram, ap=None, fig_size=(16, 10), output_fig=False):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram_ = spectrogram.detach().cpu().numpy().squeeze().T
    else:
        spectrogram_ = spectrogram.T
    spectrogram_ = spectrogram_.astype(np.float32) if spectrogram_.dtype == np.float16 else spectrogram_
    if ap is not None:
        spectrogram_ = ap.denormalize(spectrogram_)  # pylint: disable=protected-access
    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram_, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    if not output_fig:
        plt.close()
    return fig