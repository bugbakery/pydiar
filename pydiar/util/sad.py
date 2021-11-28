# flake8: noqa

import warnings

import numpy as np
import webrtcvad
from numpy.lib.stride_tricks import as_strided


def py_webrtcvad(data, fs, fs_vad, hoplength=30, vad_mode=0):

    # from librosa.core import resample
    # from librosa.util import frame
    """Voice activity detection.
    This was implementioned for easier use of py-webrtcvad.
    Thanks to: https://github.com/wiseman/py-webrtcvad.git
    Parameters
    ----------
    data : ndarray
        numpy array of mono (1 ch) speech data.
        1-d or 2-d, if 2-d, shape must be (1, time_length) or (time_length, 1).
        if data type is int, -32768 < data < 32767.
        if data type is float, -1 < data < 1.
    fs : int
        Sampling frequency of data.
    fs_vad : int, optional
        Sampling frequency for webrtcvad.
        fs_vad must be 8000, 16000, 32000 or 48000.
        Default is 16000.
    hoplength : int, optional
        Step size[milli second].
        hoplength must be 10, 20, or 30.
        Default is 0.1.
    vad_mode : int, optional
        set vad aggressiveness.
        As vad_mode increases, it becomes more aggressive.
        vad_mode must be 0, 1, 2 or 3.
        Default is 0.
    Returns
    -------
    vact : ndarray
        voice activity. time length of vact is same as input data.
        If 0, it is unvoiced, 1 is voiced.
    """

    # check argument
    if fs_vad not in [8000, 16000, 32000, 48000]:
        raise ValueError("fs_vad must be 8000, 16000, 32000 or 48000.")

    if hoplength not in [10, 20, 30]:
        raise ValueError("hoplength must be 10, 20, or 30.")

    if vad_mode not in [0, 1, 2, 3]:
        raise ValueError("vad_mode must be 0, 1, 2 or 3.")

    # check data
    if data.dtype.kind == "i":
        if data.max() > 2 ** 15 - 1 or data.min() < -(2 ** 15):
            raise ValueError(
                "when data type is int, data must be -32768 < data < 32767."
            )
        data = data.astype("f")

    elif data.dtype.kind == "f":
        if np.abs(data).max() >= 1:
            data = data / np.abs(data).max() * 0.9
            warnings.warn("input data was rescaled.")
        data = (data * 2 ** 15).astype("f")
    else:
        raise ValueError("data dtype must be int or float.")

    data = data.squeeze()
    if not data.ndim == 1:
        raise ValueError("data must be mono (1 ch).")

    # resampling
    if fs != fs_vad:
        raise ValueError("fs and fs_vad must be the same")

    resampled = data.astype("int16")

    hop = fs_vad * hoplength // 1000
    framelen = resampled.size // hop + 1
    padlen = framelen * hop - resampled.size
    paded = np.lib.pad(resampled, (0, padlen), "constant", constant_values=0)
    framed = frame(paded, frame_length=hop, hop_length=hop).T

    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    valist = [vad.is_speech(tmp.tobytes(), fs_vad) for tmp in framed]

    hop_origin = fs * hoplength // 1000
    va_framed = np.zeros([len(valist), hop_origin])
    va_framed[valist] = 1

    return va_framed.reshape(-1)[: data.size]


def get_py_webrtcvad_segments(vad_info, fs):
    vad_index = np.where(vad_info == 1.0)  # find the speech index
    vad_diff = np.diff(vad_index)

    vad_temp = np.zeros_like(vad_diff)
    vad_temp[np.where(vad_diff == 1)] = 1
    vad_temp = np.column_stack((np.array([0]), vad_temp, np.array([0])))
    final_index = np.diff(vad_temp)

    starts = np.where(final_index == 1)
    ends = np.where(final_index == -1)

    sad_info = np.column_stack((starts[1], ends[1]))
    vad_index = vad_index[0]

    segments = np.zeros_like(sad_info, dtype=np.float)
    for i in range(sad_info.shape[0]):
        segments[i][0] = float(vad_index[sad_info[i][0]]) / fs
        segments[i][1] = float(vad_index[sad_info[i][1]] + 1) / fs

    return segments  # present in seconds


def frame(x, frame_length, hop_length, axis=-1):
    """
    This function was taken from librosa and is licensed under the ISC License
    https://github.com/librosa/librosa/blob/main/LICENSE.md

    Slice a data array into (overlapping) frames.

    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.

    However, if the input data is not contiguous in memory, a warning
    will be issued and the output will be a full copy, rather than
    a view of the input data.

    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::

        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]

    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.

    The second way (``axis=0``) results in the array ``x_frames``::

        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]

    where each row ``x_frames[i]`` contains a contiguous slice of the input.

    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either to the end of the array (``axis=-1``)
    or the beginning of the array (``axis=0``).


    Parameters
    ----------
    x : np.ndarray
        Array to frame

    frame_length : int > 0 [scalar]
        Length of the frame

    hop_length : int > 0 [scalar]
        Number of steps to advance between frames

    axis : 0 or -1
        The axis along which to frame.

        If ``axis=-1`` (the default), then ``x`` is framed along its last dimension.
        ``x`` must be "F-contiguous" in this case.

        If ``axis=0``, then ``x`` is framed along its first dimension.
        ``x`` must be "C-contiguous" in this case.

    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES) or (N_FRAMES, frame_length, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::

            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]

        If ``axis=0`` (framing on the first dimension), then::

            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]

    Raises
    ------
    ParameterError
        If ``x`` is not an `np.ndarray`.

        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.

        If ``hop_length < 1``, frames cannot advance.

        If ``axis`` is not 0 or -1.  Framing is only supported along the first or last axis.


    See Also
    --------
    numpy.asfortranarray : Convert data to F-contiguous representation
    numpy.ascontiguousarray : Convert data to C-contiguous representation
    numpy.ndarray.flags : information about the memory layout of a numpy `ndarray`.

    Examples
    --------
    Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
           [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
           ...,
           [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
           [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
          dtype=float32)
    >>> y.shape
    (117601,)

    >>> frames.shape
    (2048, 1806)

    Or frame along the first axis instead of the last:

    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (1806, 2048)

    Frame a stereo signal:

    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 1806)

    Carve an STFT into fixed-length patches of 32 frames with 50% overlap

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 230)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 13)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    """

    if not isinstance(x, np.ndarray):
        raise ParameterError(
            "Input must be of type numpy.ndarray, " "given type(x)={}".format(type(x))
        )

    if x.shape[axis] < frame_length:
        raise ParameterError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ParameterError("Invalid hop_length: {:d}".format(hop_length))

    if axis == -1 and not x.flags["F_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise ParameterError("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)
