"""
Feature extractor
"""
import os
import logging
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from classifind.dataset import ClassicalMusicDataset, MusicData

N_FFT = 2048
HOP_LENGTH = 512

ABSOLUTE_PATH = Path().resolve().parent.parent
RAW_PATH = Path("data/raw/classical_music_files")
PROCESSED_PATH = Path("data/processed")
FULL_RAW_PATH = ABSOLUTE_PATH.joinpath(RAW_PATH)
FULL_PROCESSED_PATH = ABSOLUTE_PATH.joinpath(PROCESSED_PATH)


def visualise_waveform(musicdata):
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


def visualise_spectrum(musicdata):
    """
    Visualises the spectrum
    """
    plt.figure(figsize=(12, 4))
    fourier_transform = np.abs(
        librosa.stft(musicdata.timeseries[:N_FFT], hop_length=N_FFT + 1)
    )
    plt.plot(fourier_transform)
    plt.title("Spectrum")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.show()


def visualise_spectogram(musicdata, log=False):
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


def visualise_mfccs(musicdata, scale=False):
    """
    Visualises the Mel-frequency cepstral coefficients (MFCC) plot
    """
    plt.figure(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(
        y=musicdata.timeseries, sr=musicdata.sample_rate, n_mfcc=13
    )  # computed MFCCs over frames.
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        mfccs = scaler.fit_transform(mfccs)
    librosa.display.specshow(mfccs, sr=musicdata.sample_rate, x_axis="time")
    plt.colorbar(format="%+2.0f dB")
    plt.show()


def process_audiofiles(dataframe):
    """
    Processes all the audio files that are listed in the CSV file through Librosa
    """
    dataset = ClassicalMusicDataset()
    for _, row in dataframe.iterrows():
        filename = row["filename"]
        composer = row["composer"]
        path = FULL_RAW_PATH.joinpath(row["path"], filename)
        logging.info("Processing: %s", filename)
        timeseries, sample_rate = librosa.load(path)
        musicdata = MusicData(filename, composer, timeseries, sample_rate)
        dataset.add_instance(musicdata)
    return dataset


def read_metadata():
    """
    Reads in the metadata CSV
    """
    logging.info("Reading in metadata...")
    try:
        csv = pd.read_csv(
            os.path.join(FULL_PROCESSED_PATH, "classical_music_files/metadata.csv")
        )
        # Temporary code to grab 1% of the dataset for testing purposes
        csv = csv.sample(frac=0.01, random_state=0)
        return csv
    except IOError as error:
        raise error


def run():
    """
    Executes the feature extractor
    """
    df = read_metadata()
    data = process_audiofiles(df)
    print(data.num_instances())


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Loading feature extractor")
    run()
