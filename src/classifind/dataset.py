"""
Required datasets and data structures for ClassiFind
"""
import logging
import librosa


class MusicData:
    """
    Defines the basic structure of the music data that includes the waveform data, sample rate,
    duration (in ms) and tempo (in BPMs)
    """

    def __init__(
        self,
        title,
        seg_no,
        composer,
        composer_enc,
        waveform,
        sample_rate,
        start_sample,
        end_sample,
    ):
        self.title = title
        self.segment_no = seg_no
        self.composer = composer
        self.composer_encoded = composer_enc
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.start_sample = start_sample
        self.end_sample = end_sample

    def get_tempo(self):
        """
        Calculates the tempo (in BPMs) for the track
        """
        tempo, _ = librosa.beat.beat_track(y=self.waveform, sr=self.sample_rate)
        logging.info("Estimated tempo: %.2f beats per minute", tempo)
        return tempo

    def get_duration(self):
        """
        Gets the duration of the song in seconds
        """
        return self.waveform.size(1) // self.sample_rate

    def get_encoded_label(self):
        """
        Gets the encoded label of the composer's name
        """
        return self.composer_encoded


class ClassicalMusicDataset:
    """
    Represents the whole music dataset
    """

    def __init__(self):
        self.dataset = []

    def num_instances(self) -> int:
        """
        Gets the number of instances in this dataset
        """
        return len(self.dataset)

    def add_instance(self, musicdata):
        """
        Adds an instance
        """
        self.dataset.append(musicdata)

    def add_segments(self, segments):
        """
        Adds segments of an audio file (via split_audiofile)
        """
        for seg in segments:
            self.dataset.append(seg)

    def get_instance(self, index):
        """
        Gets an instance by index
        """
        return self.dataset[index]
