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
    for i in range(data.num_instances()):
        extractor = FeatureExtractor(data.get_instance(i))
        mfcc = extractor.extract_mfccs()
        spectrogram = extractor.extract_spectrogram()
        melspectrogram = extractor.extract_melspectrogram()
        logging.debug("Mfcc: %s", mfcc)
        logging.debug("Spectrogram: %s", spectrogram)
        logging.debug("Melspectrogram: %s", melspectrogram)
        normaliser = Normaliser(data.get_instance(i))
        waveform = data.get_instance(i).waveform
        logging.debug("Norm: %s", waveform)
        norm = normaliser.normalise(waveform, 0, 1)
        logging.debug("Denorm: %s", normaliser.denormalise(norm))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Running ClassiFind pipeline.")
    run_pipeline()
