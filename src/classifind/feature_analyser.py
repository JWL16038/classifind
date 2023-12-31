"""
Feature extractor
"""
from pathlib import Path
import librosa
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
