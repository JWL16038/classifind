"""
The visualiser used to display the plots of the audio file.
"""
import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchaudio import transforms

N_FFT = 2048


def plot_spectrum(musicdata):
    """
    Visualises the spectrum
    """
    fourier_transform = np.abs(
        librosa.stft(musicdata.waveform[:N_FFT], hop_length=N_FFT + 1)
    )
    plt.figure(figsize=(12, 4))
    plt.plot(fourier_transform)
    plt.title("Spectrum")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.show()


def plot_waveform_spectogram(musicdata):
    """
    Visualises the waveform and the spectrogram in one plot
    """
    spectrogram = transforms.Spectrogram(n_fft=N_FFT)
    spec = spectrogram(musicdata.waveform)
    fig, axes = plt.subplots(2, 1)
    waveform = musicdata.waveform.numpy()
    _, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / musicdata.sample_rate
    axes[0].plot(time_axis, waveform[0], linewidth=1)
    axes[0].grid(True)
    axes[0].set_xlim([0, time_axis[-1]])
    axes[0].set_xlabel("Time (secs)")
    axes[0].set_title("Original Waveform")
    axes[1].imshow(
        librosa.power_to_db(spec[0]),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    axes[1].set_title("Spectrogram")
    axes[1].set_ylabel("Frequency Bin")
    fig.tight_layout()
    plt.show()
