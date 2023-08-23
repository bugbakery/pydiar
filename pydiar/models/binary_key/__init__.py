import logging
import warnings
from typing import List

import numpy as np
from python_speech_features import mfcc

from pydiar.models.util import DiarizationModel, Segment

from .diarizationFunctions import (
    detect_speech,
    getBestClustering,
    getSegmentBKs,
    getSegments,
    getSegmentTable,
    getVgMatrix,
    performClustering,
    performResegmentation,
    trainKBM,
)


class BinaryKeyDiarizationModel(DiarizationModel):
    """
    Speaker diarization using binary key speaker modelling

    This implementation is heavily based on https://github.com/josepatino/pyBK
    """

    def __init__(
        self,
        framelength=0.025,
        frameshift=0.01,
        nfilters=30,
        ncoeff=30,
        segment_length=100,
        segment_shift=100,
        segment_overlap=100,
        kbm_max_window_shift=50,
        kbm_window_length=200,
        kbm_min_gaussians=1024,
        kbm_size_rel=0.1,
        top_gaussians_per_frame=5,
        initial_clusters=16,
        bk_one_percentage=0.2,
        clustering_metric="cosine",
        clustering_selection_metric="cosine",
        clustering_selection_max_speakers=16,
        resegmentation_model_size=6,
        resegmentation_nb_iter=10,
        resegmentation_smooth_win=100,
    ):
        self.FRAMELENGTH = framelength
        self.FRAMESHIFT = frameshift
        self.NFILTERS = nfilters
        self.NCOEFF = ncoeff

        self.SEGMENT_LENGTH = segment_length
        self.SEGMENT_SHIFT = segment_shift
        self.SEGMENT_OVERLAP = segment_overlap

        self.KBM_MAX_WINDOW_SHIFT = kbm_max_window_shift
        self.KBM_WINDOW_LENGTH = kbm_window_length
        self.KBM_MIN_GAUSSIANS = kbm_min_gaussians

        self.KBM_SIZE_REL = kbm_size_rel

        self.TOP_GAUSSIANS_PER_FRAME = top_gaussians_per_frame

        self.INITIAL_CLUSTERS = initial_clusters
        self.BK_ONE_PERCENTAGE = bk_one_percentage

        if clustering_metric != "cosine":
            raise ValueError(
                "Only `cosine` distance is supported for clustering metric."
            )

        self.CLUSTERING_METRIC = clustering_metric

        if clustering_selection_metric != "cosine":
            raise ValueError(
                "Only `cosine` distance is supported for clustering selection metric."
            )

        self.CLUSTERING_SELECTION_METRIC = "cosine"
        self.CLUSTERING_SELECTION_MAX_SPEAKERS = clustering_selection_max_speakers

        self.RESEGMENTATION_MODEL_SIZE = resegmentation_model_size
        self.RESEGMENTATION_NB_ITER = resegmentation_nb_iter
        self.RESEGMENTATION_SMOOTH_WIN = resegmentation_smooth_win

    def _extract_features(self, sample_rate, signal):
        """
        Extract the MFCC Features from an audio signal.

        Args:
            sample_rate (int): Sample rate of the input audio, in Hz
            signal (np.ndarray [shape=(n,)]): Input audio
        """

        framelength_in_samples = self.FRAMELENGTH * sample_rate
        n_fft = int(2 ** np.ceil(np.log2(framelength_in_samples)))

        additional_kwargs = {}
        if sample_rate >= 16000:
            additional_kwargs.update({"lowfreq": 20, "highfreq": 7600})

        # TODO: This is slow :(
        return mfcc(
            signal=signal,
            samplerate=sample_rate,
            numcep=self.NCOEFF,
            nfilt=self.NFILTERS,
            nfft=n_fft,
            winlen=self.FRAMELENGTH,
            winstep=self.FRAMESHIFT,
            **additional_kwargs,
        )

    def _preprocessing(self, sample_rate, signal):
        logging.info("Extracting features")
        features = self._extract_features(
            sample_rate,
            signal,
        )
        logging.info("Feature extraction done. Got", len(features), "features")

        mask = detect_speech(
            signal, sample_rate, len(features), self.FRAMESHIFT, self.FRAMELENGTH
        )

        segment_table = getSegmentTable(
            mask, self.SEGMENT_LENGTH, self.SEGMENT_SHIFT, self.SEGMENT_LENGTH
        )

        masked_features = features[np.where(mask == 1)]

        return segment_table, masked_features, len(features), mask

    def _acousting_processing(self, masked_features, segment_table, speech_mapping):
        # Calculate window shift: calculate enough to get KBM_MIN_GAUSSIANS windows,
        # but at most KBM_MAX_WINDOW_SHIFT
        kbm_window_shift = np.floor(
            (len(masked_features) - self.KBM_WINDOW_LENGTH) / self.KBM_MIN_GAUSSIANS
        )
        if kbm_window_shift > self.KBM_MAX_WINDOW_SHIFT:
            kbm_window_shift = self.KBM_MAX_WINDOW_SHIFT

        elif kbm_window_shift == 0:
            kbm_window_shift = 1

        kbm_window_count = np.floor(
            (len(masked_features) - self.KBM_WINDOW_LENGTH) / kbm_window_shift
        )
        kbm_size = int(np.floor(kbm_window_count * self.KBM_SIZE_REL))

        kbm, gmPool = trainKBM(
            masked_features, self.KBM_WINDOW_LENGTH, kbm_window_shift, kbm_size
        )
        logging.info("Trained {kbm_size} gaussians")
        Vg = getVgMatrix(masked_features, gmPool, kbm, self.TOP_GAUSSIANS_PER_FRAME)

        logging.info("Calculating binary keys")
        segmentBKTable, segmentCVTable = getSegmentBKs(
            segment_table, kbm_size, Vg, self.BK_ONE_PERCENTAGE, speech_mapping
        )
        logging.info("Initializing clustering")
        initialClustering = np.digitize(
            np.arange(len(segment_table)),
            np.arange(
                0, len(segment_table), len(segment_table) / self.INITIAL_CLUSTERS
            ),
        )

        return segmentBKTable, segmentCVTable, kbm_size, initialClustering, Vg

    def _binary_processing(
        self,
        speech_mapping,
        segment_table,
        segmentBKTable,
        segmentCVTable,
        Vg,
        kbm_size,
        initialClustering,
    ):
        logging.info("Running clustering")
        finalClusteringTable, k = performClustering(
            speech_mapping,
            segment_table,
            segmentBKTable,
            segmentCVTable,
            Vg,
            self.BK_ONE_PERCENTAGE,
            kbm_size,
            self.INITIAL_CLUSTERS,
            initialClustering,
            self.CLUSTERING_METRIC,
        )
        logging.info("Finding best clustering")

        bestClusteringID = getBestClustering(
            self.CLUSTERING_SELECTION_METRIC,
            segmentBKTable,
            segmentCVTable,
            finalClusteringTable,
            k,
            self.CLUSTERING_SELECTION_MAX_SPEAKERS,
        ).astype(int)

        best_clustering = finalClusteringTable[:, bestClusteringID]

        logging.info(
            f"Best: {bestClusteringID} with "
            f"{np.size(np.unique(best_clustering), 0)} clusters"
        )

        return best_clustering

    def _resegmentation(
        self, masked_features, speech_mapping, mask, best_clustering, segment_table
    ):
        return performResegmentation(
            masked_features,
            speech_mapping,
            mask,
            best_clustering,
            segment_table,
            self.RESEGMENTATION_MODEL_SIZE,
            self.RESEGMENTATION_NB_ITER,
            self.RESEGMENTATION_SMOOTH_WIN,
            len(masked_features),
        )

    def diarize(self, sample_rate, signal) -> List[Segment]:
        """
        The diarization process of this model consists of four parts[^1]:

        A. Preprocessing

        This performs feature extraction on the input audio, checks which parts of the
        audio are speech and returns a list of segments.

        B. Acoustic Processing (Convert Samples to Binary Keys)

        C. Binary Processing (Clustering)

        D. Resegmentation

        [^1]: This description is heavily based on
        https://ieeexplore.ieee.org/document/7268861
        """

        # A. Preprocessing
        (
            segment_table,
            masked_features,
            feature_count,
            speech_mask,
        ) = self._preprocessing(sample_rate, signal)

        if len(masked_features) == 0:
            warnings.warn(
                "No speech detected, returning empty segment list", RuntimeWarning
            )
            return []

        speech_mapping = np.zeros(feature_count)
        # you need to start the mapping from 1 and end it in the actual feature count
        # independently of the indexing style so that we don't lose features on the way
        speech_mapping[np.nonzero(speech_mask)] = np.arange(1, len(masked_features) + 1)

        # B. Acoustic Processing
        (
            segmentBKTable,
            segmentCVTable,
            kbm_size,
            initialClustering,
            Vg,
        ) = self._acousting_processing(masked_features, segment_table, speech_mapping)

        # C. AHC
        best_clustering = self._binary_processing(
            speech_mapping,
            segment_table,
            segmentBKTable,
            segmentCVTable,
            Vg,
            kbm_size,
            initialClustering,
        )

        # D. Resegmentation
        best_clustering, segment_table = self._resegmentation(
            masked_features, speech_mapping, speech_mask, best_clustering, segment_table
        )

        segments = getSegments(
            self.FRAMESHIFT,
            segment_table,
            np.squeeze(best_clustering, 1),
        )

        # TODO: Move this conversion to getSegments
        return [Segment(*x) for x in segments]
