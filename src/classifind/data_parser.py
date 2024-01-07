"""
Data parser
"""
import logging
import math
import os
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd
import torchaudio
import noisereduce as nr
from pydub import AudioSegment
from classifind.dataset import ClassicalMusicDataset, MusicData

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
        segments = load_split_audiofile(path, row)
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
            csv = csv.sample(frac=sample_amount, random_state=1)
        return csv
    except IOError as error:
        raise error


def denoise_audio(audio_segment, reduce_proportion=0.30) -> AudioSegment:
    """
    Removes as much noise from the audio segment using noisereduce
    """
    # Convert audio to numpy array
    samples = np.array(audio_segment.get_array_of_samples())

    reduced_noise = nr.reduce_noise(
        samples,
        sr=audio_segment.frame_rate,
        prop_decrease=reduce_proportion,
        use_torch=True,
    )

    # Convert reduced noise signal back to audio
    return AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels,
    )


def detect_leading_silence(sound, silence_threshold=-32.0, chunk_size=10):
    """
    Detects leading silence in the audio clip
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound

    https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    """
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[
        trim_ms : trim_ms + chunk_size
    ].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms


def trim_audio(audio_segment, total_duration) -> AudioSegment:
    """
    Trims the audio clip for any beginning and ending silences
    """
    start_trim = detect_leading_silence(audio_segment)
    end_trim = detect_leading_silence(audio_segment.reverse())
    return audio_segment[start_trim : total_duration - end_trim]


def delete_all_chunks():
    """
    Wipes all chunks from the processed directory, while keeping the composer's subdirectories.
    """
    for path, _, files in os.walk(FULL_PROCESSED_PATH):
        for item in files:
            if item.endswith(".mp3"):
                os.remove(FULL_PROCESSED_PATH.joinpath(path, item))
    logging.info("All chunks successfully deleted")


def save_chunk(audio_segment, save_path, start_time, end_time, force_reload=False):
    """
    Saves the audio segment chunk in the specified path to save the audio chunk as an mp3 file.
    """
    path = rf"{FULL_PROCESSED_PATH.joinpath(save_path)}"
    if not force_reload and os.path.isfile(path):
        return
    chunk = audio_segment[start_time:end_time]
    chunk = cast(
        AudioSegment, chunk
    )  # Sanity check that chunk is still an audiosegment to avoid type errors
    if os.path.isfile(path):
        os.remove(path)
    chunk.export(path, format="mp3")
    logging.debug("Chunk successfully saved as %s", path)


def load_split_audiofile(path, entry, split_duration=30, force_reload=False):
    """
    Splits the audio file into 30 second segments for training (this number can be changed).
    """
    audio_chunks = []

    if not os.path.isdir(FULL_PROCESSED_PATH.joinpath(entry["composer"])):
        os.mkdir(FULL_PROCESSED_PATH.joinpath(entry["composer"]))

    if path.with_suffix(".mp3"):
        audio_segment = AudioSegment.from_mp3(path)
    elif path.with_suffix(".wav"):
        audio_segment = AudioSegment.from_wav(path)
    else:
        raise ValueError(f"File with path ({path}) extension not supported")

    total_duration = len(audio_segment)
    audio_segment = denoise_audio(audio_segment)
    trimmed_segment = trim_audio(audio_segment, total_duration)

    # Calculate the number of chunks
    split_duration_ms = split_duration * 1000
    num_chunks = math.ceil(total_duration / split_duration_ms)

    # Iterate through the chunks and extract each segment
    for i in range(num_chunks):
        start_time = i * split_duration_ms
        end_time = (i + 1) * split_duration_ms
        save_path = f"{entry['composer']}/{entry['title']}_chunk_{i}.mp3"
        save_chunk(trimmed_segment, save_path, start_time, end_time, force_reload)
        waveform, sample_rate = torchaudio.load(FULL_PROCESSED_PATH.joinpath(save_path))
        musicdata = MusicData(
            entry["title"],
            i,
            entry["composer"],
            entry["composer_enc"],
            waveform,
            sample_rate,
            start_time,
            end_time,
        )
        audio_chunks.append(musicdata)
    return audio_chunks
