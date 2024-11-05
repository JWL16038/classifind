"""
Main script to run the ClassiFind Pipeline
"""
import logging
from classifind import data_parser
from classifind.feature_analyser import FeatureExtractor


def run_pipeline():
    """
    Executes the pipeline
    """
    df = data_parser.read_metadata(sample_amount=0.01)
    df = df.head(1)
    data = data_parser.process_audiofiles(df)
    logging.info("Number of instances in dataset: %s", data.num_instances())
    extractor = FeatureExtractor(data.get_instance(0))
    logging.debug(extractor.extract_melspectrogram())


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.DEBUG
    )
    logging.info("Running ClassiFind pipeline.")
    run_pipeline()
