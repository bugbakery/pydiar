# PyDiar

This repo contains simple to use, pretrained/training-less models for speaker diarization.

## Supported Models

- [x] Binary Key Speaker Modeling

  Based on [pyBK](https://github.com/josepatino/pyBK) by Jose Patino which implements the diarization system from "The EURECOM submission to the first DIHARD Challenge" by Patino, Jose and Delgado, Héctor and Evans, Nicholas

If you have any other models you would like to see added, please open an issue.

## Usage

This library seeks to provide a very basic interface. To use the Binary Key model on a file, do something like this:

```python
import numpy as np
from pydiar.models import BinaryKeyDiarizationModel, Segment
from pydiar.util.misc import optimize_segments
from pydub import AudioSegment

INPUT_FILE = "test.wav"

sample_rate = 32000
audio = AudioSegment.from_wav("test.wav")
audio = audio.set_frame_rate(sample_rate)
audio = audio.set_channels(1)

diarization_model = BinaryKeyDiarizationModel()
segments = diarization_model.diarize(
    sample_rate, np.array(audio.get_array_of_samples())
)
optimized_segments = optimize_segments(segments)
```

Now `optimized_segments` contains a list of segments with their start, length and speaker id

## Example

A simple script which reads an audio file, diarizes it and transcribes it into the WebVTT format can be found in `examples/generate_webvtt.py`.
To use it, download a vosk model from https://alphacephei.com/vosk/models and then run the script using

```shell
poetry install
poetry run python -m examples.generate_webvtt -i PATH/TO/INPUT.wav -m PATH/TO/VOSK_MODEL
```
