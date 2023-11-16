"""
Metadata Scraper
"""
import os
import logging
from pathlib import Path
import eyed3

ABSOLUTE_PATH = Path().resolve().parent.parent.parent
RELATIVE_PATH = Path("data/raw/classical_music_files")
FULL_RAW_PATH = ABSOLUTE_PATH / RELATIVE_PATH

audio_extensions = [".mp3", ".wav", ".flac"]


class MusicFile:
    """
    Class to record each file information
    """

    def __init__(self, title, artist, album):
        self.title = title
        self.artist = artist
        self.album = album

    def get_title(self):
        """
        Gets the title of the audio file
        """
        return self.title

    def get_artist(self):
        """
        Gets the artist(s) of the audio file
        """
        return self.artist

    def get_album(self):
        """
        Gets the album of the audio file
        """
        return self.album


def parse_longpath(path):
    """
    Converts the current path to a long path.

    The function to convert the path was taken from this stackoverflow answer:
    https://stackoverflow.com/questions/55815617/pathlib-path-rglob-fails-on-long-file-paths-in-windows
    """
    normalized = os.fspath(path.resolve())
    if not normalized.startswith("\\\\?\\"):
        normalized = "\\\\?\\" + normalized
    return Path(normalized)


def extract_audio_metadata(file_path):
    """
    Extracts metadata information from the audio file
    """
    try:
        combined_path = FULL_RAW_PATH / file_path
        print(os.name == "nt")
        if os.name == "nt":
            combined_path = parse_longpath(combined_path)
        audiofile = eyed3.load(combined_path)
        tag = audiofile.tag
        return tag.title, tag.artist, tag.album
    except OSError as error:
        raise error


def process_files():
    """
    Process all files and to the file list
    """
    file_paths = []
    audio_files = []
    for path in FULL_RAW_PATH.glob("**/*"):
        if path.suffix.lower() in audio_extensions:
            file_path = path.relative_to(FULL_RAW_PATH)
            file_paths.append(file_path)

    logging.info("Read in %d music files", len(file_paths))
    for file in file_paths:
        title, artist, album = extract_audio_metadata(file)
        audio_files.append((file, MusicFile(title, artist, album)))
    return audio_files
