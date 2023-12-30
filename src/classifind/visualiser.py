"""
The visualiser used to display the plots of the audio file.

Some code taken from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
"""
import librosa
import torch
from matplotlib import pyplot as plt

N_FFT = 2048


def plot_waveform(waveform, sample_rate, xlim=None, ylim=None):
    """
    Visualises the waveform
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for channel in range(num_channels):
        axes[channel].plot(time_axis, waveform[channel], linewidth=1)
        axes[channel].grid(True)
        if num_channels > 1:
            axes[channel].set_ylabel(f"Channel {channel+1}")
        if xlim:
            axes[channel].set_xlim(xlim)
        if ylim:
            axes[channel].set_ylim(ylim)
    figure.suptitle("Waveform")
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, xlim=None):
    """
    Visualises the specgram
    """
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape
    #   time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for channel in range(num_channels):
        axes[channel].specgram(waveform[channel], Fs=sample_rate)
        if num_channels > 1:
            axes[channel].set_ylabel(f"Channel {channel+1}")
        if xlim:
            axes[channel].set_xlim(xlim)
    figure.suptitle("Spectrogram")
    plt.show(block=False)


def plot_spectrogram(spec, aspect="auto", xmax=None):
    """
    Visualises the spectrogram
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title("Spectrogram (db)")
    axs.set_ylabel("Frequency bin")
    axs.set_xlabel("Frame")
    image = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(image, ax=axs)
    plt.show(block=False)


def plot_mel_fbank(fbank):
    """
    Visualises the Mel F Bank
    """
    _, axs = plt.subplots(1, 1)
    axs.set_title("Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("Frequency bin")
    axs.set_xlabel("Mel bin")
    plt.show(block=False)


def plot_pitch(waveform, sample_rate, pitch):
    """
    Visualises the pitch
    """
    _, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")
    axis2.legend(loc=0)
    plt.show(block=False)
