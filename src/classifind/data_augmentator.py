"""
The data augmentator used to create new audio files based on existing audio data.
"""
import logging
import random
from torchaudio import transforms
import torch
import torchaudio


class FeatureAugmentator:
    """
    The processor used to augment audio files
    """

    def __init__(self, musicdata):
        self.orig_min = musicdata.waveform.numpy().min()
        self.orig_max = musicdata.waveform.numpy().max()
        self.sample_rate = musicdata.sample_rate
        self.normalised = False

    def normalise(self, waveform, new_min, new_max):
        """
        A normaliser that converts the waveform to a specific range.

        Code was taken from:
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
        A denormaliser that converts the normalised waveform back to the original min and max range.

        Code was taken from:
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

    def apply_timestretch(self, waveform, rate=1.2):
        """
        Applies timestretch to the waveform
        """
        strech = transforms.TimeStretch()
        spec = strech(waveform, rate)
        return spec

    def apply_timemasking(self, waveform, time_mask=80):
        """
        Applies time masking to the waveform
        """
        masking = transforms.TimeMasking(time_mask_param=time_mask)
        spec = masking(waveform)
        return spec

    def apply_frequencymasking(self, waveform, time_mask=80):
        """
        Applies frequency masking to the waveform
        """
        masking = transforms.FrequencyMasking(freq_mask_param=time_mask)
        spec = masking(waveform)
        return spec

    def change_speed(self, waveform):
        """
        Apply random speed change to the waveform
        """
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0:  # no change
            return waveform

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, sox_effects
        )
        return transformed_audio
