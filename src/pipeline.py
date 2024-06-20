"""
Main script to run the ClassiFind Pipeline
"""
import logging
from classifind import data_parser
from classifind.feature_analyser import FeatureExtractor
from classifind.data_preprocessor import (
    apply_random_effect,
    save_sample,
    ComposeTransform,
    RandomPitch,
    RandomSpeed,
    WhiteNoise,
    RandomBackgroundNoise,
)


def run_pipeline():
    """
    Executes the pipeline
    """
    df = data_parser.read_metadata(sample_amount=0.01)
    df = df.head(1)
    data = data_parser.process_audiofiles(df)
    logging.info("Number of instances in dataset: %s", data.num_instances())
    compose = False
    for i in range(data.num_instances()):
        inst = data.get_instance(i)
        if compose:
            c_transform = ComposeTransform(
                [
                    RandomPitch(inst.sample_rate),
                    RandomSpeed(inst.sample_rate),
                    WhiteNoise(inst.sample_rate),
                    RandomBackgroundNoise(inst.sample_rate),
                ]
            )
            proc_inst = c_transform(inst)
        else:
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
