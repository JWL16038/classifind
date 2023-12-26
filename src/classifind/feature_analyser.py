"""
Feature extractor
"""
import logging
from pathlib import Path
import librosa
import torch
from torchaudio import transforms

N_FFT = 2048
N_MFCC = 13
N_MELS = 23
HOP_LENGTH = 512

ABSOLUTE_PATH = Path().resolve().parent.parent
RAW_PATH = Path("data/raw/classical_music_files")
PROCESSED_PATH = Path("data/processed/classical_music_files")
FULL_RAW_PATH = ABSOLUTE_PATH.joinpath(RAW_PATH)
FULL_PROCESSED_PATH = ABSOLUTE_PATH.joinpath(PROCESSED_PATH)


class FeatureExtractor:
    """
    Class that contains all feature analyser tools
    """

    def __init__(self, musicdata):
        self.waveform = musicdata.waveform
        self.sample_rate = musicdata.sample_rate

    def calculate_mfccs(self):
        """
        Calculates the Mel-frequency cepstral coefficients (MFCC) of the waveform for the music data
        """
        transform = transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,
                "center": False,
            },
        )
        return transform(self.waveform)

    def calculate_db(self):
        """
        Calculates the DB from amplitude for the waveform
        """
        amplitudetodb = transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        return amplitudetodb(self.waveform)

    def calculate_zero_crossing_rate(self):
        """
        Calculates the zero crossing rate for the waveform

        Indicates the number of times that a signal crosses the horizontal axis, i.e. the number of times that the amplitude reaches 0.
        """
        return librosa.feature.zero_crossing_rate(self.waveform.numpy())

    def calculate_chromagram(self):
        """
        Calculates the chromagram for the waveform
        """
        return librosa.feature.chroma_stft(y=self.waveform.numpy(), sr=self.sample_rate)

    def get_tempo(self):
        """
        Gets the tempo of the waveform
        """
        return librosa.feature.tempo(y=self.waveform.numpy(), sr=self.sample_rate)

    def calculate_tempogram(self):
        """
        Calculates the tempogram of the waveform
        """
        oenv = librosa.onset.onset_strength(
            y=self.waveform.numpy(), sr=self.sample_rate, hop_length=HOP_LENGTH
        )
        return librosa.feature.tempogram(
            onset_envelope=oenv, sr=self.sample_rate, hop_length=HOP_LENGTH
        )


class Normaliser:
    """
    A normaliser that normalises the waveform to a specific range and back to the original min and max range.
    """

    def __init__(self, musicdata):
        self.orig_min = musicdata.waveform.numpy().min()
        self.orig_max = musicdata.waveform.numpy().max()
        self.normalised = False

    def normalise(self, waveform, new_min, new_max):
        """
        https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/4e71d22683edb9bd56aa46de3f022f4e1dec1cf1/12%20Preprocessing%20pipeline/preprocess.py#L70
        """
        if self.normalised:
            logging.debug("Waveform has already been normalised. Skipping.")
            return waveform
        array = waveform.numpy()
        wave_norm = (array - array.min()) / (array.max() - array.min())
        wave_norm = wave_norm * (new_max - new_min) + new_min
        self.normalised = True
        logging.debug(
            "Before normalisation: %s, After normalisation: %s",
            array,
            torch.from_numpy(wave_norm),
        )
        return torch.from_numpy(wave_norm)

    def denormalise(self, waveform):
        """
        https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/4e71d22683edb9bd56aa46de3f022f4e1dec1cf1/12%20Preprocessing%20pipeline/preprocess.py#L70
        """
        if not self.normalised:
            logging.debug("Waveform has not normalised. Skipping.")
            return waveform
        array = waveform.numpy()
        denormalised = (array - array.min()) / (array.max() - array.min())
        denormalised = denormalised * (self.orig_max - self.orig_min) + self.orig_min
        self.normalised = False
        logging.debug(
            "Before denormalisation: %s, After denormalisation: %s",
            array,
            torch.from_numpy(denormalised),
        )
        return torch.from_numpy(denormalised)
