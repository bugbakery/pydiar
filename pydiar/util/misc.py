from typing import List

from pydiar.models import Segment


def optimize_segments(
    segments: List[Segment], keep_gaps=False, skip_short_limit=0.5
) -> List[Segment]:
    """
    Optimize segments returned by the diarization models

    Some model like the binary key model generates segments which
    a) only contain speech, leaving gaps of non-speech audio between them
    b) sometime switch to a different speaker for a very short time

    This function fixes these by removing very short segments (less than
    skip_short_limit seconds) and reassinging the gaps to the previous speaker.

    Args:
        segments (List[Segment]): A list of segments to optimize
        keep_gaps (bool, optional): If `True` gaps between segments will be kept
        skip_short_limit (float, optional): The minimum length of a segment

    Returns:
        List[Segment]: A list of optimized segments
    """
    new_segments = []
    cur_start: float = 0
    cur_end: float = 0
    cur_speaker = None
    for segment in segments:
        # If this is a very short segment, assign it to the previous segments
        if segment.length < skip_short_limit:
            continue

        if cur_speaker is None:
            cur_speaker = segment.speaker_id

        # If there is a gap in the segments, add it to the previous speaker
        if cur_end < segment.start:
            if keep_gaps:
                new_segments.append(
                    Segment(cur_start, cur_end - cur_start, cur_speaker)
                )
                cur_start = segment.start
                cur_end = segment.start + segment.length
                cur_speaker = segment.speaker_id
            else:
                cur_end = segment.start

        if segment.speaker_id != cur_speaker:
            new_segments.append(Segment(cur_start, cur_end - cur_start, cur_speaker))
            cur_start = segment.start
            cur_end = segment.start + segment.length
            cur_speaker = segment.speaker_id

    if cur_speaker is not None:
        new_segments.append(Segment(cur_start, cur_end - cur_start, cur_speaker))
    return new_segments
