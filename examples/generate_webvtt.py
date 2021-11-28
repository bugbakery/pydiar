import argparse
import json
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from vosk import KaldiRecognizer, Model, SetLogLevel

from pydiar.models import BinaryKeyDiarizationModel
from pydiar.util.misc import optimize_segments


def format_time(time):
    secs = time % 60
    mins = time // 60
    hours = int(mins // 60)
    mins = int(mins % 60)
    return f"{hours:02d}:{mins:02d}:{secs:.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help=(
            "The path to a vosk model, which you can "
            " download at https://alphacephei.com/vosk/models"
        ),
    )
    args = parser.parse_args()

    SAMPLE_RATE = 32000
    audio = AudioSegment.from_wav(args.input)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(1)

    diarization_model = BinaryKeyDiarizationModel()
    segments = diarization_model.diarize(
        SAMPLE_RATE, np.array(audio.get_array_of_samples())
    )

    segments = optimize_segments(segments)

    SetLogLevel(-1)
    model = Model(str(args.model))

    for segment in segments:
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.SetWords(False)
        st = format_time(segment.start)
        end = format_time(segment.start + segment.length)
        print(f"{st} --> {end}")
        data_start = int(segment.start * 1000)
        data_end = int((segment.start + segment.length) * 1000)
        data = audio[data_start:data_end]
        rec.AcceptWaveform(data.get_array_of_samples().tobytes())
        vosk_result = json.loads(rec.FinalResult())
        print(f"<v speaker{int(segment.speaker_id)}>{vosk_result['text']}\n")
