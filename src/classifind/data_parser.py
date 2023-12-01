"""
Data parser
"""
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
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
    timeseries, sample_rate = librosa.load(path)
    duration = librosa.get_duration(y=timeseries)

    # Calculate the number of segments
    num_segments = int(np.ceil(duration / SPLIT_DURATION))

    for i in range(num_segments):
        # Calculate start and end times for each segment
        start_time = i * SPLIT_DURATION
        end_time = min((i + 1) * SPLIT_DURATION, duration)

        # Extract the segment from the audio
        segment = timeseries[
            int(start_time * sample_rate) : int(end_time * sample_rate)
        ]

        # # Save the segment as a new WAV file
        # segment_output_path = f"{output_path}_segment_{i + 1}.wav"
        # librosa.output.write_wav(segment_output_path, segment, musicdata.sample_rate)

        musicdata = MusicData(
            filename,
            i,
            composer,
            composer_enc,
            segment,
            sample_rate,
            start_time,
            end_time,
        )
        segments.append(musicdata)
    return segments
