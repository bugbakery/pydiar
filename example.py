from pydiar.models import BinaryKeyDiarizationModel, Segment
from pydiar.util.misc import optimize_segments
from pydub import AudioSegment
import numpy as np

if __name__ == "__main__":
    INPUT_FILE = "./"
    OUTLIER_FILE = "./"

    sample_rate = 32000
    audio = AudioSegment.from_wav(INPUT_FILE)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)

    outlier = AudioSegment.from_wav(OUTLIER_FILE)
    outlier = outlier.set_frame_rate(sample_rate)
    outlier = outlier.set_channels(1)

    # combine audio and outlier
    audio = audio + outlier

    diarization_model = BinaryKeyDiarizationModel(clustering_selection_max_speakers=2)

    segments = diarization_model.diarize(
        sample_rate, np.array(audio.get_array_of_samples())
    )
    optimized_segments = optimize_segments(segments)

    unique_speakers = set()
    for segment in optimized_segments:
        unique_speakers.add(segment.speaker_id)

    print(f"Number of speakers: {len(unique_speakers)}")
