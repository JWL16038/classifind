"""
Feature extractor
"""
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchaudio import transforms

N_FFT = 2048
HOP_LENGTH = 512

ABSOLUTE_PATH = Path().resolve().parent.parent
RAW_PATH = Path("data/raw/classical_music_files")
PROCESSED_PATH = Path("data/processed/classical_music_files")
FULL_RAW_PATH = ABSOLUTE_PATH.joinpath(RAW_PATH)
FULL_PROCESSED_PATH = ABSOLUTE_PATH.joinpath(PROCESSED_PATH)


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


def calculate_mfccs(musicdata):
    """
    Calculates the Mel-frequency cepstral coefficients (MFCC) of the waveform for the music data
    """
    transform = transforms.MFCC(
        sample_rate=musicdata.sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )
    mfcc = transform(musicdata.waveform)
    return mfcc


def calculate_db(musicdata):
    """
    Calculates the DB from amplitude for the waveform
    """
    amplitudetodb = transforms.AmplitudeToDB(stype="amplitude", top_db=80)
    spec = amplitudetodb(musicdata.waveform)
    return spec


def calculate_zero_crossing_rate(musicdata):
    """
    Calculates the zero crossing rate for the waveform
    """
    return librosa.feature.zero_crossing_rate(musicdata.waveform.numpy())


def calculate_chromagram(musicdata):
    """
    Calculates the chromagram for the waveform
    """
    return librosa.feature.chroma_stft(
        y=musicdata.waveform.numpy(), sr=musicdata.sample_rate
    )
