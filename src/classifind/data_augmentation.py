"""
The data augmentator used to create new audio files based on existing audio data.
"""
import logging
import os
from pathlib import Path
import random
from glob import glob
import pandas as pd
import torchaudio
from torchaudio import transforms
import torch

ABSOLUTE_PATH = Path().resolve().parent
NOISE_PATH = Path("data/raw/noise")
ARCA23K_PATH = Path("data/raw/ARCA23K")
SAMPLES_PATH = Path("data/processed/samples")
FULL_NOISE_PATH = ABSOLUTE_PATH / ARCA23K_PATH  # NOISE_PATH
FULL_SAMPLES_PATH = ABSOLUTE_PATH / SAMPLES_PATH
RELATIVE_PROCESSED_PATH = Path("data/processed/classical_music_files")
FULL_PROCESSED_PATH = ABSOLUTE_PATH / RELATIVE_PROCESSED_PATH


def save_processed_mp3(inst, save_path):
    """
    Saves the given audio instance as a .mp3 file, incrementing the file name if one or more exist in the directory.

    Parameters:
    inst: The audio instance to be saved.
    directory: The directory where the .mp3 file should be saved.
    base_filename: The base name for the .mp3 file (default is 'file').
    """
    Path(os.path.join(FULL_PROCESSED_PATH, save_path)).parent.mkdir(
        parents=True, exist_ok=True
    )

    torchaudio.save(
        os.path.join(FULL_PROCESSED_PATH, save_path),
        inst.waveform,
        inst.sample_rate,
    )
    logging.info("Saved instance as %s", save_path)


def apply_random_effect(inst, probability=0.5, use_compose=False):
    """
    Randomly applies one of RandomBackgroundNoise, WhiteNoise, RandomPitch, or RandomSpeed
    to an audio instance with a certain probability.

    Parameters:
    inst: The audio instance to be processed.
    p: The probability with which to apply an effect (default is 0.5).

    Returns:
    The audio instance after applying the random effect.
    """
    if random.random() < probability:
        if use_compose:
            c_transform = ComposeTransform(
                [
                    RandomPitch(inst.sample_rate),
                    RandomBackgroundNoise(inst.sample_rate),
                    RandomTempo(inst.sample_rate),
                ]
            )
            return c_transform(inst)
        effects = [
            RandomPitch(inst.sample_rate),
            RandomBackgroundNoise(inst.sample_rate),
            RandomTempo(inst.sample_rate),
            WhiteNoise(inst.sample_rate),
            ReverseAudio(),
            RandomCrop(inst.sample_rate),
            DistortionAudio(),
        ]
        effect = random.choice(effects)
        return effect(inst)
    return inst


class ComposeTransform:
    """
    Compose a list of functions to augment the instance

    Function was taken from
    https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, functions):
        self.functions = functions

    def __call__(self, audio_data):
        for func in self.functions:
            audio_data = func(audio_data)
        return audio_data

    def get_transforms(self):
        """
        Gets all composed transform functions
        """
        return [type(func).__name__ for func in self.functions]


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


class RandomTempo:
    """
    Applies random tempo change to the waveform.

    The code to build this class was originally taken from
    https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.tempo_factor = None

    def __call__(self, musicdata):
        tempo_factor = random.choice([0.8, 0.9, 1.0, 1.1, 1.2])
        if tempo_factor == 1.0:  # no change
            return musicdata
        # Change the tempo and resample to original rate:
        sox_effects = [
            ["tempo", str(tempo_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            musicdata.waveform, self.sample_rate, sox_effects
        )
        self.tempo_factor = tempo_factor
        musicdata.waveform = transformed_audio
        logging.debug(
            "Tempo: %s, Duration: %s",
            tempo_factor,
            transformed_audio.size(1) / self.sample_rate,
        )
        return musicdata

    def get_current_tempo(self):
        """
        Gets the current tempo factor
        """
        if self.tempo_factor is None:
            return -1
        return self.tempo_factor


class WhiteNoise:
    """
    Applies white background noise to the waveform.

    Function taken from
    https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    """

    def __init__(self, sample_rate, min_snr_db=15, max_snr_db=30):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def __call__(self, musicdata):
        # Calculate RMS of original signal
        original_rms = torch.sqrt(torch.mean(musicdata.waveform**2))

        # Generate white noise
        noise = torch.randn_like(musicdata.waveform)
        noise_rms = torch.sqrt(torch.mean(noise**2))

        # Calculate desired noise level based on random SNR
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        snr_linear = 10 ** (snr_db / 20)

        # Scale noise to achieve desired SNR while preserving signal volume
        scaling_factor = original_rms / (noise_rms * snr_linear)
        scaled_noise = noise * scaling_factor

        # Add scaled noise to original signal
        musicdata.waveform = musicdata.waveform + scaled_noise

        # Normalize to prevent clipping while preserving relative volume
        max_val = torch.max(torch.abs(musicdata.waveform))
        if max_val > 1.0:
            musicdata.waveform = musicdata.waveform / max_val

        return musicdata

    def get_white_noise(self, musicdata):
        """
        Gets the white noise from the musicdata waveform.
        """
        # Calculate RMS of original signal
        original_rms = torch.sqrt(torch.mean(musicdata.waveform**2))

        # Generate white noise
        noise = torch.randn_like(musicdata.waveform)
        noise_rms = torch.sqrt(torch.mean(noise**2))

        # Calculate desired noise level based on random SNR
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        snr_linear = 10 ** (snr_db / 20)

        # Scale noise to achieve desired SNR while preserving signal volume
        scaling_factor = original_rms / (noise_rms * snr_linear)
        scaled_noise = noise * scaling_factor
        return scaled_noise


class RandomBackgroundNoise:
    """
    Applies random background noise samples to the audio while preserving the original signal's volume.

    The noise is selected from a collection of pre-recorded environmental sounds, excluding musical
    instruments and similar sounds that could interfere with the classification task. The noise
    is applied with gaps of silence and controlled SNR to create realistic background ambience.

    Function taken from
    https://jonathanbgn.com/2021/08/30/audio-augmentation.html

    Attributes:
        sample_rate (int): The sample rate of the audio
        min_snr_db (float): Minimum signal-to-noise ratio in decibels
        max_snr_db (float): Maximum signal-to-noise ratio in decibels
    """

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

    def __init__(self, sample_rate, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        # Load available noise files, excluding musical instruments
        metadata = pd.read_csv(
            FULL_NOISE_PATH.joinpath("metadata.csv"), dtype={"fname": "str"}
        )
        metadata = metadata[~metadata["label"].isin(self.exclude_list)]
        noise_files = glob(str(FULL_NOISE_PATH.joinpath("**/*.wav")), recursive=True)
        self.noise_files = [
            fpath
            for fpath in noise_files
            if Path(fpath).stem in metadata["fname"].values
        ]

    def __call__(self, musicdata, prob_threshold=0.35):
        """
        Applies random background noise to the audio while maintaining original signal volume.

        Args:
            musicdata: MusicData instance containing the waveform to be processed
            prob_threshold (float): Probability of adding noise vs. silence at each segment

        Returns:
            MusicData: The processed audio with background noise added
        """
        audio_length = musicdata.waveform.shape[-1]
        original_waveform = musicdata.waveform.clone()

        # Generate noise sequence with silence gaps
        noise = self._generate_noise_sequence(audio_length, prob_threshold)

        # Calculate and apply SNR-based scaling
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        snr_linear = 10 ** (snr_db / 20)

        signal_power = torch.mean(original_waveform**2)
        noise_power = torch.mean(noise**2)
        scaling_factor = torch.sqrt(signal_power / (noise_power * (snr_linear**2)))

        scaled_noise = (
            noise * scaling_factor * 0.3
        )  # Additional reduction factor for subtlety

        # Mix original audio with scaled noise
        musicdata.waveform = original_waveform + scaled_noise

        # Normalize to prevent clipping while preserving relative volume
        max_val = torch.max(torch.abs(musicdata.waveform))
        if max_val > 1.0:
            musicdata.waveform = musicdata.waveform / max_val

        return musicdata

    def _generate_noise_sequence(self, target_length, prob_threshold):
        """
        Generates a sequence of background noise mixed with silence periods.

        Args:
            target_length (int): Desired length of the noise sequence in samples
            prob_threshold (float): Probability of adding noise vs. silence at each segment

        Returns:
            torch.Tensor: Generated noise sequence
        """
        noise = torch.zeros(1, random.randrange(3000, 10000))

        while noise.shape[-1] <= target_length:
            if random.random() <= prob_threshold:
                new_noise, _ = self.get_random_noise(noise)
                noise = torch.cat([noise, new_noise], dim=-1)
            else:
                silence = torch.zeros(1, random.randrange(3000, 10000))
                noise = torch.cat([noise, silence], dim=-1)

        return noise[..., :target_length]

    def get_random_noise(self, waveform):
        """
        Loads and preprocesses a random noise file from the available collection.

        Args:
            reference_waveform (torch.Tensor): Reference waveform for volume matching

        Returns:
            tuple: (preprocessed noise tensor, length of noise)
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
        logging.debug("Gain DB for noise file %s: %s", random_noise_file, gain_db)
        return noise, noise.shape[-1]


class ReverseAudio:
    """
    Flips/reverses the waveform.

    The code was originally taken from:
    https://github.com/Spijkervet/torchaudio-augmentations/blob/master/torchaudio_augmentations/augmentations/reverse.py
    """

    def __call__(self, musicdata):
        musicdata.waveform = torch.flip(musicdata.waveform, dims=[-1])
        return musicdata

    def get_reversed_waveform(self, musicdata):
        """
        Gets the reversed waveform

        Parameters:
            musicdata: MusicData instance containing the waveform to be processed

        Returns:
            MusicData: The processed audio with the reversed waveform
        """
        return torch.flip(musicdata.waveform, dims=[-1])


class DistortionAudio:
    """
    Applies random distortion effect to the waveform using waveshaping.

    Parameters:
        min_drive (float): Minimum amount of distortion
        max_drive (float): Maximum amount of distortion
        mix (float): Mix between dry and wet signal (0.0 to 1.0)
    """

    def __init__(self, min_drive=3, max_drive=10, mix=0.5):
        super().__init__()
        self.min_drive = min_drive
        self.max_drive = max_drive
        self.mix = mix
        self.current_drive = None

    def get_random_drive(self):
        """
        Gets a random drive.
        """
        return random.uniform(self.min_drive, self.max_drive)

    def waveshape(self, wave):
        """
        Gets the waveshape.

        Parameters:
            wave: The waveform to be processed
        """
        # Get new random drive value for each call
        self.current_drive = self.get_random_drive()
        # Apply non-linear distortion using tanh
        return torch.tanh(wave * self.current_drive)

    def __call__(self, musicdata):
        """
        Applies random distortion to the waveform.

        Parameters:
            musicdata: MusicData instance containing the waveform to be processed

        Returns:
            MusicData: The processed audio with the distorted waveform
        """
        # Normalize input
        waveform = musicdata.waveform
        max_val = torch.max(torch.abs(waveform))
        normalized = waveform / max_val

        # Apply distortion
        distorted = self.waveshape(normalized)

        # Mix dry and wet signals
        mixed = (1 - self.mix) * normalized + self.mix * distorted

        # Restore original scale
        musicdata.waveform = mixed * max_val
        return musicdata

    def get_distorted_waveform(self, musicdata):
        """
        Gets the distorted waveform.

        Parameters:
            musicdata: MusicData instance containing the waveform to be processed

        Returns:
            MusicData: The processed audio with the distorted waveform
        """
        temp_data = musicdata
        return self(temp_data).waveform


class RandomCrop:
    """
    Randomly crops the audio waveform to a specified duration while preserving
    the meaningful parts of the audio (avoiding silent sections).

    The class uses an energy-based approach to identify non-silent sections and
    ensures the crop includes meaningful audio content.

    Attributes:
        sample_rate (int): The sample rate of the audio
        max_crop_seconds (float): Maximum duration to crop in seconds
        min_crop_seconds (float): Minimum duration to crop in seconds
        min_energy_threshold (float): Minimum energy threshold to consider a section non-silent
        frame_length (int): Length of frames for energy calculation in samples
    """

    def __init__(
        self,
        sample_rate,
        max_crop_seconds=30.0,
        min_crop_seconds=15.0,
        min_energy_threshold=0.01,
        frame_length=2048,
    ):
        self.sample_rate = sample_rate
        self.max_crop_samples = int(max_crop_seconds * sample_rate)
        self.min_crop_samples = int(min_crop_seconds * sample_rate)
        self.min_energy_threshold = min_energy_threshold
        self.frame_length = frame_length

    def __call__(self, musicdata):
        """
        Applies random cropping to the audio, avoiding silent sections.

        Args:
            musicdata: MusicData instance containing the waveform to be processed

        Returns:
            MusicData: The processed audio with cropped waveform
        """
        waveform = musicdata.waveform
        total_samples = waveform.shape[-1]

        # Ensure minimum length requirements
        if total_samples <= self.min_crop_samples:
            return musicdata

        # Calculate energy per frame
        frames = waveform.unfold(-1, self.frame_length, self.frame_length // 2)
        frame_energies = torch.mean(frames**2, dim=1)

        # Find frames with sufficient energy
        valid_frames = torch.where(frame_energies > self.min_energy_threshold)[0]

        if len(valid_frames) == 0:
            # If no valid frames found, fall back to random crop
            crop_length = random.randint(self.min_crop_samples, self.max_crop_samples)
            start_idx = random.randint(0, total_samples - crop_length)
        else:
            # Random crop length between min and max
            crop_length = random.randint(
                self.min_crop_samples, min(self.max_crop_samples, total_samples)
            )

            # Convert frame indices to sample indices
            valid_starts = valid_frames * (self.frame_length // 2)

            # Filter valid start positions that allow for full crop length
            valid_starts = valid_starts[valid_starts <= (total_samples - crop_length)]

            if len(valid_starts) == 0:
                # If no valid start positions, fall back to random crop
                start_idx = random.randint(0, total_samples - crop_length)
            else:
                # Choose random start position from valid positions
                start_idx = valid_starts[random.randint(0, len(valid_starts) - 1)]

        # Apply the crop
        musicdata.waveform = waveform[..., start_idx : start_idx + crop_length]

        # Update start and end sample positions
        musicdata.start_sample = start_idx
        musicdata.end_sample = start_idx + crop_length

        return musicdata

    def get_current_crop_duration(self, musicdata):
        """
        Gets the duration of the current crop in seconds.

        Args:
            musicdata: MusicData instance containing the cropped waveform

        Returns:
            float: Duration of the crop in seconds
        """
        return musicdata.waveform.shape[-1] / self.sample_rate
