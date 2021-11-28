import abc
from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    start: float
    length: float
    speaker_id: int


class DiarizationModel(abc.ABC):
    @abc.abstractmethod
    def diarize(self, sample_rate, signal) -> List[Segment]:
        pass
