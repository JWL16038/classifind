"""
Main script to run the ClassiFind Pipeline
"""
import logging
from classifind import data_parser
from classifind.feature_analyser import FeatureExtractor
from classifind.data_preprocessor import RandomPitch, RandomSpeed, RandomBackgroundNoise


def run_pipeline():
    """
    Executes the pipeline
    """
    df = data_parser.read_metadata(sample_amount=0.01)
    df = df.head(1)
    data = data_parser.process_audiofiles(df)
    logging.info("Number of instances in dataset: %s", data.num_instances())
    for i in range(1):  # (data.num_instances()):
        inst = data.get_instance(i)
        background_noise = RandomBackgroundNoise(inst.sample_rate, True)
        out = background_noise(inst)
        random_pitch = RandomPitch(inst.sample_rate)
        out = random_pitch(inst)
        random_speed = RandomSpeed(inst.sample_rate)
        out = random_speed(inst)
        print(out)
        extractor = FeatureExtractor(data.get_instance(i))
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
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Running ClassiFind pipeline.")
    run_pipeline()
