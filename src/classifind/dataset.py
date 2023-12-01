"""
Required datasets and data structures for ClassiFind
"""
import logging
import librosa


class ClassicalMusicDataset:
    """
    Represents the whole music dataset
    """

    def __init__(self):
        self.dataset = {}

    def num_instances(self):
        """
        Gets the number of instances in this dataset
        """
        return len(self.dataset)

    def add_instance(self, musicdata):
        """
        Adds an instance
        """
        index = len(self.dataset)
        self.dataset[index] = musicdata

    def add_segments(self, segments):
        """
        Adds segments of an audio file (via split_audiofile)
        """
        for i in segments:
            index = len(self.dataset)
            self.dataset[index] = i

    def get_instance(self, index):
        """
        Gets an instance by index
        """
        return self.dataset.get(index)


class MusicData:
    """
    Defines the basic structure of the music data that includes the timeseries data, sample rate,
    duration (in ms) and tempo (in BPMs)
    """

    def __init__(
        self,
        title,
        seg_no,
        composer,
        composer_enc,
        timeseries,
        sample_rate,
        start_time,
        end_time,
    ):
        self.title = title
        self.segment_no = seg_no
        self.composer = composer
        self.composer_encoded = composer_enc
        self.timeseries = timeseries
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.end_time = end_time

    def get_tempo(self):
        """
        Calculates the tempo (in BPMs) for the track
        """
        tempo, _ = librosa.beat.beat_track(y=self.timeseries, sr=self.sample_rate)
        logging.info("Estimated tempo: %.2f beats per minute", tempo)
        return tempo

    def get_duration(self):
        """
        Gets the duration of the song
        """
        return librosa.get_duration(y=self.timeseries)

    def get_encoded_label(self):
        """
        Gets the encoded label of the composer's name
        """
        return self.composer_encoded
