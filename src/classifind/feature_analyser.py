"""
Feature extractor
"""
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np

N_FFT = 2048
HOP_LENGTH = 512

ABSOLUTE_PATH = Path().resolve().parent.parent
RAW_PATH = Path("data/raw/classical_music_files")
PROCESSED_PATH = Path("data/processed")
FULL_RAW_PATH = ABSOLUTE_PATH.joinpath(RAW_PATH)
FULL_PROCESSED_PATH = ABSOLUTE_PATH.joinpath(PROCESSED_PATH)


def calculate_constantq_transform(musicdata, visualise=False):
    """
    Visualises the Constant-Q Transform
    """
    hop_length = 512
    cqt = librosa.cqt(
        musicdata.timeseries, sr=musicdata.sample_rate, n_bins=72, hop_length=hop_length
    )

    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    if visualise:
        fig, axes = plt.subplots(figsize=(15, 5))
        img = librosa.display.specshow(
            log_cqt,
            sr=musicdata.sample_rate,
            x_axis="time",
            y_axis="cqt_note",
            cmap="coolwarm",
            ax=axes,
        )
        axes.set_title("Constant-Q power spectrum")
        fig.colorbar(img, ax=axes, format="%+2.0f dB")
        plt.show()
    return log_cqt


def plot_waveform(musicdata):
    """
    Visualises the wave of the plot
    """
    _, axes = plt.subplots()
    librosa.display.waveshow(
        y=musicdata.timeseries, sr=musicdata.sample_rate, color="blue", axis="time"
    )
    axes.set_xlabel("Time (seconds)")
    axes.set_title(f"Timeseries plot for {musicdata.title}")
    plt.show()


def plot_spectrum(musicdata):
    """
    Visualises the spectrum
    """
    fourier_transform = np.abs(
        librosa.stft(musicdata.timeseries[:N_FFT], hop_length=N_FFT + 1)
    )
    plt.figure(figsize=(12, 4))
    plt.plot(fourier_transform)
    plt.title("Spectrum")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectogram(musicdata, log=False):
    """
    Visualises the spectrogram
    """
    stft_audio = librosa.stft(musicdata.timeseries, n_fft=N_FFT, hop_length=HOP_LENGTH)
    db_data = librosa.power_to_db(np.abs(stft_audio) ** 2)
    plt.figure(figsize=(12, 4))
    if log:
        librosa.display.specshow(
            db_data,
            sr=musicdata.sample_rate,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="log",
        )
    else:
        librosa.display.specshow(
            db_data,
            sr=musicdata.sample_rate,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="linear",
        )
    plt.colorbar(format="%+2.0f dB")
    plt.show()


def calculate_mfccs(musicdata, visualise=False):
    """
    Visualises the Mel-frequency cepstral coefficients (MFCC) plot
    """
    plt.figure(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(
        y=musicdata.timeseries, sr=musicdata.sample_rate, n_mfcc=13
    )  # computed MFCCs over frames.
    if visualise:
        librosa.display.specshow(mfccs, sr=musicdata.sample_rate, x_axis="time")
        plt.colorbar(format="%+2.0f dB")
        plt.show()
    return mfccs
