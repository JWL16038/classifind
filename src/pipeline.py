"""
Main script to run the ClassiFind Pipeline
"""
import logging
import random
from classifind import data_parser
from classifind.feature_analyser import FeatureExtractor
from classifind.data_preprocessor import (
    WhiteNoise,
    RandomPitch,
    RandomSpeed,
    RandomBackgroundNoise,
    save_sample,
)


def apply_random_effect(inst, probability=0.5):
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
        effects = [
            RandomBackgroundNoise(inst.sample_rate, True),
            WhiteNoise(inst.sample_rate, True),
            RandomPitch(inst.sample_rate),
            RandomSpeed(inst.sample_rate),
        ]
        effect = random.choice(effects)
        return effect(inst)
    return inst


def run_pipeline():
    """
    Executes the pipeline
    """
    df = data_parser.read_metadata(sample_amount=0.01)
    df = df.head(1)
    data = data_parser.process_audiofiles(df)
    logging.info("Number of instances in dataset: %s", data.num_instances())
    for i in range(data.num_instances()):
        inst = data.get_instance(i)
        proc_inst = apply_random_effect(inst, 0.75)
        save_sample(proc_inst, "../data/processed/samples", "sample")
        extractor = FeatureExtractor(proc_inst)
        mfcc = extractor.extract_mfccs()
        spectrogram = extractor.extract_spectrogram()
        melspectrogram = extractor.extract_melspectrogram()
        pitch = extractor.extract_pitch()
        logging.debug("Mfcc: %s", mfcc)
        logging.debug("Spectrogram: %s", spectrogram)
        logging.debug("Melspectrogram: %s", melspectrogram)
        logging.debug("Pitch: %s", pitch)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.DEBUG
    )
    logging.info("Running ClassiFind pipeline.")
    run_pipeline()
