"""
Feature extractor
"""
import logging
from pathlib import Path
import librosa
import torch
from torchaudio import transforms, functional

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

    def extract_mfccs(self):
        """
        Extracts the Mel-frequency cepstral coefficients (MFCC) of the waveform
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

    def extract_spectrogram(self, win_length=None, hop_length=None):
        """
        Extracts the Spectrogram of the waveform
        """
        spectrogram = transforms.Spectrogram(
            n_fft=N_FFT,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        return spectrogram(self.waveform)

    def extract_melspectrogram(self, win_length=None, hop_length=None):
        """
        Extracts the MelSpectrogram of the waveform
        """
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=N_FFT,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=N_MELS,
            mel_scale="htk",
        )

        melspec = mel_spectrogram(self.waveform)
        return melspec

    def extract_pitch(self):
        """
        Extracts the pitch frequency of the waveform
        """
        pitch = functional.detect_pitch_frequency(self.waveform, self.sample_rate)
        return pitch

    def get_tempo(self):
        """
        Gets the tempo of the waveform
        """
        return librosa.feature.tempo(y=self.waveform.numpy(), sr=self.sample_rate)

    def print_stats(self, src=None):
        """
        Prints the stats of the waveform
        """
        if src:
            print("-" * 10)
            print("Source:", src)
            print("-" * 10)
        print("Sample Rate:", self.sample_rate)
        print("Shape:", tuple(self.waveform.shape))
        print("Dtype:", self.waveform.dtype)
        print(f" - Max:     {self.waveform.max().item():6.3f}")
        print(f" - Min:     {self.waveform.min().item():6.3f}")
        print(f" - Mean:    {self.waveform.mean().item():6.3f}")
        print(f" - Std Dev: {self.waveform.std().item():6.3f}")


class FeatureAugmentator:
    """
    Class that contains all feature augmentation tools
    """

    def __init__(self, musicdata):
        self.waveform = musicdata.waveform
        self.sample_rate = musicdata.sample_rate

    def apply_timestretch(self, rate=1.2):
        """
        Applies timestretch to the waveform
        """
        strech = transforms.TimeStretch()
        spec = strech(self.waveform, rate)
        return spec

    def apply_timemasking(self, time_mask=80):
        """
        Applies time masking to the waveform
        """
        masking = transforms.TimeMasking(time_mask_param=time_mask)
        spec = masking(self.waveform)
        return spec

    def apply_frequencymasking(self, time_mask=80):
        """
        Applies frequency masking to the waveform
        """
        masking = transforms.FrequencyMasking(freq_mask_param=time_mask)
        spec = masking(self.waveform)
        return spec


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
