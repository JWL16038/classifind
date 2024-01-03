"""
Data parser
"""
import logging
import os
from pathlib import Path
import pandas as pd
import torchaudio
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
            csv = csv.sample(frac=sample_amount, random_state=0)
        return csv
    except IOError as error:
        raise error


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    """
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

    sample_rate = audio_segment.frame_rate
    print(f"Sample rate: {sample_rate}")

    # https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    start_trim = detect_leading_silence(audio_segment)
    end_trim = detect_leading_silence(audio_segment.reverse())
    duration = len(audio_segment)
    audio_segment = audio_segment[start_trim : duration - end_trim]

    split_duration_ms = split_duration * 1000
    total_duration = len(audio_segment)

    # Calculate the number of chunks
    num_chunks = total_duration // split_duration_ms

    # Iterate through the chunks and extract each segment
    for i in range(num_chunks):
        start_time = i * split_duration_ms
        end_time = (i + 1) * split_duration_ms
        new_filename = f"{entry['title']}_chunk_{i}.mp3"
        path = FULL_PROCESSED_PATH.joinpath(
            f"{entry['composer']}/{new_filename}"
        ).as_posix()
        if not os.path.isfile(path) or force_reload:
            chunk = audio_segment[start_time:end_time]
            chunk.export(path, format="mp3")
        waveform, sample_rate = torchaudio.load(path)
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
