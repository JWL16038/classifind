"""
Data parser
"""
import os
import logging
from pathlib import Path
import pandas as pd
import torchaudio
from classifind.dataset import ClassicalMusicDataset, MusicData

SPLIT_DURATION = 30

# =====================================
# Paths
# =====================================
ABSOLUTE_PATH = Path().resolve().parent
RELATIVE_PATH = Path("data/raw/classical_music_files")
RELATIVE_PROCESSED_PATH = Path("data/processed/classical_music_files")
FULL_RAW_PATH = ABSOLUTE_PATH / RELATIVE_PATH
FULL_PROCESSED_PATH = ABSOLUTE_PATH / RELATIVE_PROCESSED_PATH


def process_audiofiles(dataframe):
    """
    Processes all the audio files that are listed in the CSV file through Librosa
    """
    dataset = ClassicalMusicDataset()
    for _, row in dataframe.iterrows():
        path = FULL_RAW_PATH.joinpath(row["path"], row["filename"])
        logging.info("Processing: %s", row["filename"])
        segments = split_audiofile(path, row)
        dataset.add_segments(segments)
    return dataset


def read_metadata(sample_amount=None):
    """
    Reads in the metadata CSV
    """
    logging.info("Reading in metadata...")
    try:
        csv = pd.read_csv(os.path.join(FULL_PROCESSED_PATH, "metadata.csv"))
        if sample_amount is not None:
            # Temporary code to grab X% of the dataset for testing purposes
            csv = csv.sample(frac=sample_amount, random_state=0)
        return csv
    except IOError as error:
        raise error


def split_audiofile(path, row):
    """
    Splits the audio file into 30 second segments for training (this number can be changed).
    """
    segments = []
    filename = row["filename"]
    composer = row["composer"]
    composer_enc = row["composer_enc"]
    waveform, sample_rate = torchaudio.load(path)

    # Calculate the number of segments
    num_segments = waveform.size(1) // (SPLIT_DURATION * sample_rate)

    for i in range(num_segments):
        # Calculate start and end times for each segment
        start_sample = i * SPLIT_DURATION * sample_rate
        end_sample = min((i + 1) * SPLIT_DURATION * sample_rate, waveform.size(1))

        # Extract the segment from the audio
        segment = waveform[:, int(start_sample) : int(end_sample)]

        musicdata = MusicData(
            filename,
            i,
            composer,
            composer_enc,
            segment,
            sample_rate,
            start_sample,
            end_sample,
        )
        segments.append(musicdata)
    return segments
