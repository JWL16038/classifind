"""
Main script to run the ClassiFind Pipeline
"""
import logging
from classifind import data_parser, feature_analyser


def run_pipeline():
    """
    Executes the pipeline
    """
    df = data_parser.read_metadata(sample_amount=0.01)
    data = data_parser.process_audiofiles(df)
    logging.info("Number of instances in dataset: %s", data.num_instances())
    for i in data.dataset:
        feature_analyser.calculate_mfccs(data.get_instance(i))
        feature_analyser.calculate_chromagram(data.get_instance(i))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Running ClassiFind pipeline.")
    run_pipeline()
