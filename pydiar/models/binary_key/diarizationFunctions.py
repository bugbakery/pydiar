"""
This file was taken from https://github.com/josepatino/pyBK/blob/master/diarizationFunctions.py
and is licensed under the MIT license: https://github.com/josepatino/pyBK/blob/master/LICENSE
"""
# flake8: noqa

# AUTHORS
# Jose PATINO, EURECOM, Sophia-Antipolis, France, 2019
# http://www.eurecom.fr/en/people/patino-jose
# Contact: patino[at]eurecom[dot]fr, josempatinovillar[at]gmail[dot]com

# DIARIZATION FUNCTIONS

import logging

import numpy as np
import webrtcvad
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn import mixture

import pydiar.util.sad


def unravelMask(mask):
    """
    Unravel a mask to yield continous segments.
    Example: If your mask is

    1 0 0 0 0 1 1 1

    this will return two segments: 0 and 5...7
    However this returns the changing points, the beggings of the segments,
    the ends of the segments and the number of segments seperately, so in this case:

    (array([-1,  0,  0,  0,  1,  0,  0]), array([0, 5]), array([0, 7]), 2)


    """
    changePoints = np.diff(1 * mask)
    segBeg = np.where(changePoints == 1)[0] + 1
    segEnd = np.where(changePoints == -1)[0]
    if mask[0] == 1:
        segBeg = np.insert(segBeg, 0, 0)
    if mask[-1] == 1:
        segEnd = np.append(segEnd, np.size(mask) - 1)
    nSegs = np.size(segBeg)
    return changePoints, segBeg, segEnd, nSegs


def getSegmentTable(
    mask: np.ndarray, segment_length: int, segment_shift: int, segment_overlap: int
):
    """
    Generate a list of segments from mask which are at most segment_length long.
    """
    changePoints, segBeg, segEnd, nSegs = unravelMask(mask)
    segmentTable = np.empty([0, 4])
    for i in range(nSegs):
        begs = np.arange(segBeg[i], segEnd[i], segment_shift)
        bbegs = np.maximum(segBeg[i], begs - segment_overlap)
        ends = np.minimum(begs + segment_length - 1, segEnd[i])
        eends = np.minimum(ends + segment_overlap, segEnd[i])
        segmentTable = np.vstack(
            (segmentTable, np.vstack((bbegs, begs, ends, eends)).T)
        )
    return segmentTable


def trainKBM(data, windowLength, windowRate, kbmSize):
    # Calculate number of gaussian components in the whole gaussian pool
    numberOfComponents = int(np.floor((np.size(data, 0) - windowLength) / windowRate))
    # Add new array for storing the mvn objects
    gmPool = []
    likelihoodVector = np.zeros((numberOfComponents, 1))
    muVector = np.zeros((numberOfComponents, np.size(data, 1)))
    sigmaVector = np.zeros((numberOfComponents, np.size(data, 1)))
    for i in range(numberOfComponents):
        mu = np.mean(
            data[np.arange((i * windowRate), (i * windowRate + windowLength), 1, int)],
            axis=0,
        )
        std = np.std(
            data[np.arange((i * windowRate), (i * windowRate + windowLength), 1, int)],
            axis=0,
        )
        muVector[i], sigmaVector[i] = mu, std
        mvn = multivariate_normal(mu, std)
        gmPool.append(mvn)
        likelihoodVector[i] = -np.sum(
            mvn.logpdf(
                data[
                    np.arange((i * windowRate), (i * windowRate + windowLength), 1, int)
                ]
            )
        )
    # Define the global dissimilarity vector
    v_dist = np.inf * np.ones((numberOfComponents, 1))
    # Create the kbm itself, which is a vector of kbmSize size, and contains the gaussian IDs of the components
    kbm = np.zeros((kbmSize, 1))
    # As the stored likelihoods are negative, get the minimum likelihood
    bestGaussianID = np.where(likelihoodVector == np.min(likelihoodVector))[0]
    currentGaussianID = bestGaussianID
    kbm[0] = currentGaussianID
    v_dist[currentGaussianID] = -np.inf
    # Compare the current gaussian with the remaining ones
    dpairsAll = cdist(muVector, muVector, metric="cosine")
    np.fill_diagonal(dpairsAll, -np.inf)
    for j in range(1, kbmSize):
        dpairs = dpairsAll[currentGaussianID]
        v_dist = np.minimum(v_dist, dpairs.T)
        # Once all distances are computed, get the position with highest value
        # set this position to 1 in the binary KBM ponemos a 1 en el vector kbm
        # store the gaussian ID in the KBM
        currentGaussianID = np.where(v_dist == np.max(v_dist))[0]
        kbm[j] = currentGaussianID
        v_dist[currentGaussianID] = -np.inf
    return [kbm, gmPool]


def getVgMatrix(data, gmPool, kbm, topGaussiansPerFrame):
    logging.info("Calculating log-likelihood table... ", end="")
    logLikelihoodTable = getLikelihoodTable(data, gmPool, kbm)
    logging.info("done")
    logging.info("Calculating Vg matrix... ", end="")
    # The original code was:
    #     Vg = np.argsort(-logLikelihoodTable)[:, 0:topGaussiansPerFrame]
    #     return Vg
    # However this sorts the entire likelihood table, but we only need the top five,
    # thich argpartition does in linear time
    partition_args = np.argpartition(-logLikelihoodTable, 5, axis=1)[:, :5]
    partition = np.take_along_axis(-logLikelihoodTable, partition_args, axis=1)
    vg = np.take_along_axis(partition_args, np.argsort(partition), axis=1)
    logging.info("done")
    return vg


def getLikelihoodTable(data, gmPool, kbm):
    # GETLIKELIHOODTABLE computes the log-likelihood of each feature in DATA
    # against all the Gaussians of GMPOOL specified by KBM vector
    # Inputs:
    #   DATA = matrix of feature vectors
    #   GMPOOL = pool of Gaussians of the kbm model
    #   KBM = vector of the IDs of the actual Gaussians of the KBM
    # Output:
    #   LOGLIKELIHOODTABLE = NxM matrix storing the log-likelihood of each of
    #   the N features given each of th M Gaussians in the KBM
    kbmSize = np.size(kbm, 0)
    logLikelihoodTable = np.zeros([np.size(data, 0), kbmSize])
    for i in range(kbmSize):
        # logging.info("pdf", i, gmPool[int(kbm[i])].logpdf(data).shape)
        logLikelihoodTable[:, i] = gmPool[int(kbm[i])].logpdf(data)
    return logLikelihoodTable


def getSegmentBKs(segmentTable, kbmSize, Vg, bitsPerSegmentFactor, speechMapping):
    # GETSEGMENTBKS converts each of the segments in SEGMENTTABLE into a binary key
    # and/or cumulative vector.

    # Inputs:
    #   SEGMENTTABLE = matrix containing temporal segments returned by 'getSegmentTable' function
    #   KBMSIZE = number of components in the kbm model
    #   VG = matrix of the top components per frame returned by 'getVgMatrix' function
    #   BITSPERSEGMENTFACTOR = proportion of bits that will be set to 1 in the binary keys
    # Output:
    #   SEGMENTBKTABLE = NxKBMSIZE matrix containing N binary keys for each N segments in SEGMENTTABLE
    #   SEGMENTCVTABLE = NxKBMSIZE matrix containing N cumulative vectors for each N segments in SEGMENTTABLE

    numberOfSegments = np.size(segmentTable, 0)
    segmentBKTable = np.zeros([numberOfSegments, kbmSize])
    segmentCVTable = np.zeros([numberOfSegments, kbmSize])
    for i in range(numberOfSegments):
        # Conform the segment according to the segmentTable matrix
        beginningIndex = int(segmentTable[i, 0])
        endIndex = int(segmentTable[i, 3])
        # Store indices of features of the segment
        # speechMapping is substracted one because 1-indexing is used for this variable
        A = np.arange(
            speechMapping[beginningIndex] - 1, speechMapping[endIndex], dtype=int
        )
        segmentBKTable[i], segmentCVTable[i] = binarizeFeatures(
            kbmSize, Vg[A, :], bitsPerSegmentFactor
        )
    logging.info("done")
    return segmentBKTable, segmentCVTable


def binarizeFeatures(binaryKeySize, topComponentIndicesMatrix, bitsPerSegmentFactor):
    # BINARIZEMATRIX Extracts a binary key and a cumulative vector from the the
    # rows of VG specified by vector A

    # Inputs:
    #   BINARYKEYSIZE = binary key size
    #   TOPCOMPONENTINDICESMATRIX = matrix of top Gaussians per frame
    #   BITSPERSEGMENTFACTOR = Proportion of positions of the binary key which will be set to 1
    # Output:
    #   BINARYKEY = 1xBINARYKEYSIZE binary key
    #   V_F = 1xBINARYKEYSIZE cumulative vector
    numberOfElementsBinaryKey = np.floor(binaryKeySize * bitsPerSegmentFactor)
    # Declare binaryKey
    binaryKey = np.zeros([1, binaryKeySize])
    # Declare cumulative vector v_f
    v_f = np.zeros([1, binaryKeySize])
    unique, counts = np.unique(topComponentIndicesMatrix, return_counts=True)
    # Fill CV
    v_f[:, unique] = counts
    # Fill BK
    binaryKey[0, np.argsort(-v_f)[0][0 : int(numberOfElementsBinaryKey)]] = 1
    # CV normalization
    vf_sum = np.sum(v_f)
    if vf_sum != 0:
        v_f = v_f / vf_sum
    return binaryKey, v_f


def performClustering(
    speechMapping,
    segmentTable,
    segmentBKTable,
    segmentCVTable,
    Vg,
    bitsPerSegmentFactor,
    kbmSize,
    N_init,
    initialClustering,
    clusteringMetric,
):
    numberOfSegments = np.size(segmentTable, 0)
    clusteringTable = np.zeros([numberOfSegments, N_init])
    finalClusteringTable = np.zeros([numberOfSegments, N_init])
    activeClusters = np.ones([N_init, 1])
    clustersBKTable = np.zeros([N_init, kbmSize])
    clustersCVTable = np.zeros([N_init, kbmSize])
    clustersBKTable, clustersCVTable = calcClusters(
        clustersCVTable,
        clustersBKTable,
        activeClusters,
        initialClustering,
        N_init,
        segmentTable,
        kbmSize,
        speechMapping,
        Vg,
        bitsPerSegmentFactor,
    )
    ####### Here the clustering algorithm begins. Steps are:
    ####### 1. Reassign all data among all existing signatures and retrain them
    ####### using the new clustering
    ####### 2. Save the resulting clustering solution
    ####### 3. Compare all signatures with each other and merge those two with
    ####### highest similarity, creating a new signature for the resulting
    ####### cluster
    ####### 4. Back to 1 if #clusters > 1
    for k in range(N_init):
        ####### 1. Data reassignment. Calculate the similarity between the current segment with all clusters and assign it to the one which maximizes
        ####### the similarity. Finally re-calculate binaryKeys for all cluster
        # before doing anything, check if there are remaining clusters
        # if there is only one active cluster, break
        if np.sum(activeClusters) == 1:
            break
        clustersStillActive = np.zeros([1, N_init])
        segmentToClustersSimilarityMatrix = binaryKeySimilarity_cdist(
            clusteringMetric,
            segmentBKTable,
            segmentCVTable,
            clustersBKTable,
            clustersCVTable,
        )
        # clusteringTable[:,k] = finalClusteringTable[:,k] = np.argmax(segmentToClustersSimilarityMatrix,axis=1)+1
        clusteringTable[:, k] = finalClusteringTable[:, k] = (
            np.nanargmax(segmentToClustersSimilarityMatrix, axis=1) + 1
        )
        # clustersStillActive[:,np.unique(clusteringTable[:,k]).astype(int)-1] = 1
        clustersStillActive[:, np.unique(clusteringTable[:, k]).astype(int) - 1] = 1
        ####### update all binaryKeys for all new clusters
        activeClusters = clustersStillActive
        clustersBKTable, clustersCVTable = calcClusters(
            clustersCVTable,
            clustersBKTable,
            activeClusters.T,
            clusteringTable[:, k].astype(int),
            N_init,
            segmentTable,
            kbmSize,
            speechMapping,
            Vg,
            bitsPerSegmentFactor,
        )
        ####### 2. Compare all signatures with each other and merge those two with highest similarity, creating a new signature for the resulting
        clusterSimilarityMatrix = binaryKeySimilarity_cdist(
            clusteringMetric,
            clustersBKTable,
            clustersCVTable,
            clustersBKTable,
            clustersCVTable,
        )
        np.fill_diagonal(clusterSimilarityMatrix, np.nan)
        value = np.nanmax(clusterSimilarityMatrix)
        location = np.nanargmax(clusterSimilarityMatrix)
        R, C = np.unravel_index(location, (N_init, N_init))
        ### Then we merge clusters R and C
        # logging.info('Merging clusters',R+1,'and',C+1,'with a similarity score of',np.around(value,decimals=4))
        logging.info(
            "Merging clusters",
            "%3s" % str(R + 1),
            "and",
            "%3s" % str(C + 1),
            "with a similarity score of",
            np.around(value, decimals=4),
        )
        activeClusters[0, C] = 0
        ### 3. Save the resulting clustering and go back to 1 if the number of clusters >1
        mergingClusteringIndices = np.where(clusteringTable[:, k] == C + 1)
        # update clustering table
        clusteringTable[mergingClusteringIndices[0], k] = R + 1
        # remove binarykey for removed cluster
        clustersBKTable[C, :] = np.zeros([1, kbmSize])
        clustersCVTable[C, :] = np.nan
        # prepare the vector with the indices of the features of thew new cluster and then binarize
        segmentsToBinarize = np.where(clusteringTable[:, k] == R + 1)[0]
        M = []
        for l in np.arange(np.size(segmentsToBinarize, 0)):
            M = np.append(
                M,
                np.arange(
                    int(segmentTable[segmentsToBinarize][:][l, 1]),
                    int(segmentTable[segmentsToBinarize][:][l, 2]) + 1,
                ),
            )
        clustersBKTable[R, :], clustersCVTable[R, :] = binarizeFeatures(
            kbmSize,
            Vg[np.array(speechMapping[np.array(M, dtype="int")], dtype="int") - 1].T,
            bitsPerSegmentFactor,
        )
    logging.info("done")
    return clusteringTable, k


def calcClusters(
    clustersCVTable,
    clustersBKTable,
    activeClusters,
    clusteringTable,
    N_init,
    segmentTable,
    kbmSize,
    speechMapping,
    Vg,
    bitsPerSegmentFactor,
):
    for i in np.arange(N_init):
        if activeClusters[i] == 1:
            segmentsToBinarize = np.where(clusteringTable == i + 1)[0]
            M = []
            for l in np.arange(np.size(segmentsToBinarize, 0)):
                M = np.append(
                    M,
                    np.arange(
                        int(segmentTable[segmentsToBinarize][:][l, 1]),
                        int(segmentTable[segmentsToBinarize][:][l, 2]) + 1,
                    ),
                )
            clustersBKTable[i], clustersCVTable[i] = binarizeFeatures(
                kbmSize,
                Vg[
                    np.array(speechMapping[np.array(M, dtype="int")], dtype="int") - 1
                ].T,
                bitsPerSegmentFactor,
            )
        else:
            clustersBKTable[i] = np.zeros([1, kbmSize])
            clustersCVTable[i] = np.nan
    return clustersBKTable, clustersCVTable


def binaryKeySimilarity_cdist(clusteringMetric, bkT1, cvT1, bkT2, cvT2):
    if clusteringMetric == "cosine":
        S = 1 - cdist(cvT1, cvT2, metric=clusteringMetric)
    elif clusteringMetric == "jaccard":
        S = 1 - cdist(bkT1, bkT2, metric=clusteringMetric)
    else:
        logging.info("Clustering metric must be cosine or jaccard")
    return S


def getBestClustering(
    bestClusteringMetric, bkT, cvT, clusteringTable, n, maxNrSpeakers
):

    wss = np.zeros([1, n])
    overallMean = np.mean(cvT, 0)
    if bestClusteringMetric == "cosine":
        distances = cdist(
            np.expand_dims(overallMean, axis=0), cvT, bestClusteringMetric
        )
    elif bestClusteringMetric == "jaccard":
        nBitsTol = np.sum(bkT[0, :])
        indices = np.argsort(-overallMean)
        overallMean = np.zeros([1, np.size(bkT, 1)])
        overallMean[0, indices[np.arange(nBitsTol).astype(int)]] = 1
        distances = cdist(overallMean, bkT, bestClusteringMetric)
    distances2 = np.square(distances)
    wss[0, n - 1] = np.sum(distances2)
    for i in np.arange(n - 1):
        T = clusteringTable[:, i]
        clusterIDs = np.unique(T)
        variances = np.zeros([np.size(clusterIDs, 0), 1])
        for j in np.arange(np.size(clusterIDs, 0)):
            clusterIDsIndex = np.where(T == clusterIDs[j])
            meanVector = np.mean(cvT[clusterIDsIndex, :], axis=1)
            if bestClusteringMetric == "cosine":
                distances = cdist(
                    meanVector, cvT[clusterIDsIndex, :][0], bestClusteringMetric
                )
            elif bestClusteringMetric == "jaccard":
                indices = np.argsort(-meanVector[0, :])
                meanVector = np.zeros([1, np.size(bkT, 1)])
                meanVector[0, indices[np.arange(nBitsTol).astype(int)]] = 1
                distances = cdist(
                    meanVector, bkT[clusterIDsIndex, :][0], bestClusteringMetric
                )
            distances2 = np.square(distances)
            variances[j] = np.sum(distances2)
        wss[0, i] = np.sum(variances)
    nPoints = np.size(wss, 1)
    allCoord = np.vstack((np.arange(1, nPoints + 1), wss)).T
    firstPoint = allCoord[0, :]
    allCoord = allCoord[
        np.arange(np.where(allCoord[:, 1] == np.min(allCoord[:, 1]))[0], nPoints), :
    ]
    lineVec = allCoord[-1, :] - firstPoint
    rssLineVec = np.sqrt(np.sum(np.square(lineVec)))
    if rssLineVec != 0:
        lineVecN = lineVec / rssLineVec
    else:
        lineVecN = lineVec
    vecFromFirst = np.subtract(allCoord, firstPoint)
    scalarProduct = vecFromFirst * lineVecN
    scalarProduct = scalarProduct[:, 0] + scalarProduct[:, 1]
    vecFromFirstParallel = np.expand_dims(scalarProduct, axis=1) * np.expand_dims(
        lineVecN, 0
    )
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(np.square(vecToLine), axis=1))
    bestClusteringID = allCoord[np.argmax(distToLine)][0]

    # Select best clustering that matches max speaker limit
    nrSpeakersPerSolution = np.zeros((clusteringTable.shape[1]))
    for k in np.arange(clusteringTable.shape[1]):
        nrSpeakersPerSolution[k] = np.size(np.unique(clusteringTable[:, k]))

    firstAllowedClustering = np.min(np.where(nrSpeakersPerSolution <= maxNrSpeakers))
    # Note: clusters are ordered from most clusters to least, so this selects the bestClusteringID
    # unless it has more than maxNrSpeakers nodes, in which case it selects firstAllowedClustering
    bestClusteringID = np.maximum(
        firstAllowedClustering,
        bestClusteringID,
    )
    return bestClusteringID


def performResegmentation(
    data,
    speechMapping,
    mask,
    finalClusteringTable,
    segmentTable,
    modelSize,
    nbIter,
    smoothWin,
    numberOfSpeechFeatures,
):

    np.random.seed(0)

    changePoints, segBeg, segEnd, nSegs = unravelMask(mask)
    speakerIDs = np.unique(finalClusteringTable)
    trainingData = np.empty([2, 0])
    for i in np.arange(np.size(speakerIDs, 0)):
        spkID = speakerIDs[i]
        speakerFeaturesIndxs = []
        idxs = np.where(finalClusteringTable == spkID)[0]
        for l in np.arange(np.size(idxs, 0)):
            speakerFeaturesIndxs = np.append(
                speakerFeaturesIndxs,
                np.arange(
                    int(segmentTable[idxs][:][l, 1]),
                    int(segmentTable[idxs][:][l, 2]) + 1,
                ),
            )
        formattedData = np.vstack(
            (
                np.tile(spkID, (1, np.size(speakerFeaturesIndxs, 0))),
                speakerFeaturesIndxs,
            )
        )
        trainingData = np.hstack((trainingData, formattedData))

    llkMatrix = np.zeros([np.size(speakerIDs, 0), numberOfSpeechFeatures])
    for i in np.arange(np.size(speakerIDs, 0)):
        spkIdxs = np.where(trainingData[0, :] == speakerIDs[i])[0]
        spkIdxs = speechMapping[trainingData[1, spkIdxs].astype(int)].astype(int) - 1
        msize = np.minimum(modelSize, np.size(spkIdxs, 0))
        w_init = np.ones([msize]) / msize
        m_init = data[
            spkIdxs[np.random.randint(np.size(spkIdxs, 0), size=(1, msize))[0]], :
        ]
        gmm = mixture.GaussianMixture(
            n_components=msize,
            covariance_type="diag",
            weights_init=w_init,
            means_init=m_init,
            verbose=0,
        )
        gmm.fit(data[spkIdxs, :])
        llkSpk = gmm.score_samples(data)
        llkSpkSmoothed = np.zeros([1, numberOfSpeechFeatures])
        for jx in np.arange(nSegs):
            sectionIdx = np.arange(
                speechMapping[segBeg[jx]] - 1, speechMapping[segEnd[jx]]
            ).astype(int)
            sectionWin = np.minimum(smoothWin, np.size(sectionIdx))
            if sectionWin % 2 == 0:
                sectionWin = sectionWin - 1
            if sectionWin >= 2:
                llkSpkSmoothed[0, sectionIdx] = smooth(llkSpk[sectionIdx], sectionWin)
            else:
                llkSpkSmoothed[0, sectionIdx] = llkSpk[sectionIdx]
        llkMatrix[i, :] = llkSpkSmoothed[0].T
    segOut = np.argmax(llkMatrix, axis=0) + 1
    segChangePoints = np.diff(segOut)
    changes = np.where(segChangePoints != 0)[0]
    relSegEnds = speechMapping[segEnd]
    relSegEnds = relSegEnds[0:-1]
    changes = np.sort(np.unique(np.hstack((changes, relSegEnds))))

    # Create the new segment and clustering tables
    currentPoint = 0
    finalSegmentTable = np.empty([0, 4])
    finalClusteringTableResegmentation = np.empty([0, 1])

    for i in np.arange(np.size(changes, 0)):
        addedRow = np.hstack(
            (
                np.tile(
                    np.where(speechMapping == np.maximum(currentPoint, 1))[0], (1, 2)
                ),
                np.tile(
                    np.where(speechMapping == np.maximum(1, changes[i].astype(int)))[0],
                    (1, 2),
                ),
            )
        )
        finalSegmentTable = np.vstack((finalSegmentTable, addedRow[0]))
        finalClusteringTableResegmentation = np.vstack(
            (finalClusteringTableResegmentation, segOut[(changes[i]).astype(int)])
        )
        currentPoint = (changes[i] + 1).astype(int)
    addedRow = np.hstack(
        (
            np.tile(np.where(speechMapping == np.maximum(currentPoint, 1))[0], (1, 2)),
            np.tile(np.where(speechMapping == numberOfSpeechFeatures)[0], (1, 2)),
        )
    )

    finalSegmentTable = np.vstack((finalSegmentTable, addedRow[0]))
    finalClusteringTableResegmentation = np.vstack(
        (finalClusteringTableResegmentation, segOut[currentPoint])
    )

    return finalClusteringTableResegmentation, finalSegmentTable


def getSegments(frameshift, finalSegmentTable, finalClusteringTable):
    numberOfSpeechFeatures = finalSegmentTable[-1, 2].astype(int) + 1
    solutionVector = np.zeros([1, numberOfSpeechFeatures])
    for segmentRow, clusterRow in zip(finalSegmentTable, finalClusteringTable):
        solutionVector[
            0,
            np.arange(segmentRow[1], segmentRow[2] + 1).astype(int),
        ] = clusterRow
    seg = np.empty([0, 3])
    solutionDiff = np.diff(solutionVector)[0]
    first = 0
    for i in np.arange(0, np.size(solutionDiff, 0)):
        if solutionDiff[i]:
            last = i + 1
            seg1 = (first) * frameshift
            seg2 = (last - first) * frameshift
            seg3 = solutionVector[0, last - 1]
            if seg3:
                seg = np.vstack((seg, [seg1, seg2, seg3]))
            first = i + 1
    last = np.size(solutionVector, 1)
    seg1 = (first - 1) * frameshift
    seg2 = (last - first + 1) * frameshift
    seg3 = solutionVector[0, last - 1]
    logging.info(seg, seg1, seg2, seg3)
    seg = np.vstack((seg, [seg1, seg2, seg3]))
    return seg


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    # From https://stackoverflow.com/a/40443565
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), "valid") / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[: WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def detect_speech(data, sr, nFeatures, frameshift, framelength):
    """
    The following VAD applying procedure is adapted from the repository
    provided as part of the DIHARD II challenge speech enhancement system:
    https://github.com/staplesinLA/denoising_DIHARD18/blob/master/main_get_vad.py
    """

    va_framed = pydiar.util.sad.py_webrtcvad(
        data, fs=sr, fs_vad=sr, hoplength=30, vad_mode=0
    )
    logging.info("Speech segments:", unravelMask(va_framed))
    segments = pydiar.util.sad.get_py_webrtcvad_segments(va_framed, sr)

    mask = np.zeros(nFeatures)
    for i in range(segments.shape[0]):
        start_time = segments[i][0]
        end_time = segments[i][1]
        start_frame = np.floor(start_time / frameshift)
        end_frame = np.ceil(end_time / frameshift)
        mask[int(start_frame) : int(end_frame)] = 1
    return mask
