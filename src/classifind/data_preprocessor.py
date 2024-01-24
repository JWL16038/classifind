"""
The data augmentator used to create new audio files based on existing audio data.
"""
import logging
import math
from pathlib import Path
import random
from glob import glob
import pandas as pd
from torchaudio import transforms
import torch
import torchaudio

ABSOLUTE_PATH = Path().resolve().parent
NOISE_PATH = Path("data/raw/noise")
ARCA23K_PATH = Path("data/raw/ARCA23K")
SAMPLES_PATH = Path("data/processed/samples")
FULL_NOISE_PATH = ABSOLUTE_PATH / ARCA23K_PATH  # NOISE_PATH
FULL_SAMPLES_PATH = ABSOLUTE_PATH / SAMPLES_PATH


class RandomPitch:
    """
    Apply a pitch change to the waveform

    Function taken from
    https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.pitch_step = None

    def __call__(self, musicdata):
        n_step = random.choice([-3, -2, -1, 0, 1, 2, 3])
        if n_step == 0:  # no change
            return musicdata
        transform = transforms.PitchShift(self.sample_rate, n_step)
        self.pitch_step = n_step
        musicdata.waveform = transform(musicdata.waveform)  # (channel, time)
        logging.debug(
            "Pitch steps: %s, Duration %s",
            n_step,
            musicdata.waveform.size(1) / self.sample_rate,
        )
        return musicdata

    def get_current_pitch_step(self):
        """
        Gets the current pitch step
        """
        if self.pitch_step is None:
            return -1
        return self.pitch_step


class RandomSpeed:
    """
    Apply random speed change to the waveform

    Function taken from
    https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.speed_factor = None

    def __call__(self, musicdata):
        speed_factor = random.choice([0.8, 0.9, 1.0, 1.1, 1.2])
        if speed_factor == 1.0:  # no change
            return musicdata
        # Change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            musicdata.waveform, self.sample_rate, sox_effects
        )
        self.speed_factor = speed_factor
        musicdata.waveform = transformed_audio
        logging.debug(
            "Speed: %s, Duration: %s",
            speed_factor,
            transformed_audio.size(1) / self.sample_rate,
        )
        return musicdata

    def get_current_speed_factor(self):
        """
        Gets the current speed factor
        """
        if self.speed_factor is None:
            return -1
        return self.speed_factor


class RandomBackgroundNoise:
    """
    Applys a set of random background noise to the waveform.

    Function taken from
    https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, sample_rate, save_sample=False, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.save_sample = save_sample
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        metadata = pd.read_csv(
            FULL_NOISE_PATH.joinpath("metadata.csv"), dtype={"fname": "str"}
        )
        exclude_list = [
            "Acoustic_guitar",
            "Bass_guitar",
            "Bowed_string_instrument",
            "Crash_cymbal",
            "Electric_guitar",
            "Female_singing",
            "Gong",
            "Harp",
            "Organ",
            "Piano",
            "Rattle_(instrument)",
            "Snare_drum",
            "Train",
            "Trumpet",
            "Wind_instrument_and_woodwind_instrument",
        ]
        metadata = metadata[~metadata["label"].isin(exclude_list)]
        noise_files = glob(str(FULL_NOISE_PATH.joinpath("**/*.wav")), recursive=True)
        self.noise_files = [
            fpath
            for fpath in noise_files
            if Path(fpath).stem in metadata["fname"].values
        ]

    def __call__(self, musicdata, prob_threshold=0.35):
        audio_length = musicdata.waveform.shape[-1]
        if random.random() <= prob_threshold:
            noise, _ = self.get_random_noise(musicdata.waveform)
        else:
            # Fill the noise waveform with empty silence anywhere between 3 to 10 seconds
            noise = torch.zeros(1, random.randrange(3000, 10000))
        # Continue adding random noise files until the entire waveform is filled
        while noise.shape[-1] <= audio_length:
            if random.random() <= prob_threshold:
                new_noise, _ = self.get_random_noise(musicdata.waveform)
                noise = torch.cat([noise, new_noise], dim=-1)
            else:
                # Fill the noise waveform with empty silence anywhere between 3 to 10 seconds
                noise = torch.cat(
                    [noise, torch.zeros(1, random.randrange(3000, 10000))], dim=-1
                )

        # Trim the noise if it's longer than the audio
        if noise.shape[-1] > audio_length:
            noise = noise[..., :audio_length]

        snr = math.exp(random.randint(self.min_snr_db, self.max_snr_db) / 10)
        audio_power = musicdata.waveform.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * (noise_power / audio_power)
        assert (
            noise.shape[-1] == audio_length
        ), "Length of noise doesn't align with the audio length"
        musicdata.waveform = (scale * musicdata.waveform + noise) / 2
        if self.save_sample:
            torchaudio.save(
                FULL_SAMPLES_PATH.joinpath("test.wav"),
                musicdata.waveform,
                musicdata.sample_rate,
            )
        return musicdata

    def get_random_noise(self, waveform):
        """
        Gets a random noise audio file from the noise directory
        """
        random_noise_file = random.choice(self.noise_files)
        effects = [
            ["remix", "1"],  # convert to mono
            ["rate", str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(
            random_noise_file, effects, normalize=True
        )
        input_peak = torch.amax(noise.abs())
        target_peak = torch.amax(waveform.abs())
        gain_db = target_peak.item() - input_peak.item()
        noise = torchaudio.functional.gain(noise, gain_db=gain_db)
        logging.debug("Gain DB for noise: %s", gain_db)
        length = noise.shape[-1]
        return noise, length
