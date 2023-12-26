"""
Main script to run the ClassiFind Pipeline
"""
import logging
from classifind import data_parser
from classifind.feature_analyser import FeatureExtractor, Normaliser


def run_pipeline():
    """
    Executes the pipeline
    """
    df = data_parser.read_metadata(sample_amount=0.01)
    data = data_parser.process_audiofiles(df)
    logging.info("Number of instances in dataset: %s", data.num_instances())
    for i in data.dataset:
        extractor = FeatureExtractor(data.get_instance(i))
        mfcc = extractor.calculate_mfccs()
        print(mfcc)
        normaliser = Normaliser(data.get_instance(i))
        norm = normaliser.normalise(data.get_instance(i).waveform, 0, 1)
        print(normaliser.denormalise(norm))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Running ClassiFind pipeline.")
    run_pipeline()
