import numpy as np
import pandas as pd
from typing import List, Literal, Optional
from scipy.fft import rfft, irfft, rfftfreq
from scipy.linalg import toeplitz
from scipy.signal import butter, filtfilt, find_peaks, firwin2, iirnotch
from scipy.stats import zscore
from sklearn.cluster import KMeans
#from sklearn.mixture import GaussianMixture
from ..evaluation.evaluate import *


def bandpass_signals(
        data: np.ndarray, # (n_channels x n_samples)
        fsamp: float, 
        high_pass: float = 20, 
        low_pass: float = 500, 
        method: Literal["butter", "firwin2"] = "butter",
        order: int | None = 2, 
        numtabs: int | None = 101,
) -> np.ndarray:
    """
    Bandpass filter timeseries data using a digital infinite 
    impulse  response filter (``butter``) or finite impulse 
    response filter (``firwin2``). To obtain a zero-phase response,
    the filter is applied in forward and backward direction.

    Parameters
    ----------

    data : np.ndarray
        Input data of shape ``(n_channels x n_samples)``
    fsamp : float 
        Sampling frequency in Hz
    high_pass : float, default 20 
        Cut-off frequency for the high-pass filter in Hz    
    low_pass : float, default 500 
        Cut-off frequency for the low-pass filter in Hz
    method :  {"butter", "firwin2"}, default "butter"
        Filter type
    order : int | None, default 2 
        Order of the filter (required if ``method==butter``) 
    numtabs : int | None, default 101 
        Number of filter tabs (required if ``method==firwin2``)

    Returns
    -------
    data : np.ndarray  
        filtered data of shape ``(n_channels, n_samples)``

    Examples
    --------
    Generate 5 seconds of white Gaussian noise and obtain 
    colored noise using a second order butterworth filter

    >>> from muniverse.algorithms.core import bandpass_signals
    >>> fsamp = 2048
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((3, int(fsamp*5))) 
    >>> data_filt = bandpass_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     high_pass=10,
    ...     low_pass=450,
    ...     method="butter",
    ...     order=2
    ... )

    Filter the same data using a finite impulse response filter   

    >>> data_filt = bandpass_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     high_pass=10,
    ...     low_pass=450,
    ...     method="firwin2",
    ...     numtabs=101
    ... )

    """

    if high_pass >= low_pass:
        raise ValueError(
            "The value of low_pass must be larger than your high_pass value."
        )

    if method == "butter":
        if order is None:
            raise ValueError(
                "If method is *butter*, order must be an integer."
            )
        b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype="band")
        data = filtfilt(b, a, data, axis=1)
    elif method == "firwin2":
        if numtabs is None:
            raise ValueError(
                "If method is *firwin2*, numtabs must be an integer."
            )
        # Normalize frequencies to Nyquist (0..1)
        nyq = fsamp / 2
        f = [0, high_pass*0.9, high_pass, low_pass, low_pass*1.1, nyq]  # small transition bands
        m = [0, 0, 1, 1, 0, 0]  # 0 outside band, 1 inside
        # Design FIR filter
        fir_coeff = firwin2(numtabs, f, m, fs=fsamp)
        data = filtfilt(fir_coeff, [1.0], data, axis=1)
    else:
        raise ValueError(
            f"The specified filter type option *{method}* is invalid"
            "Must be one of *butter* or *firwin2*"
        )

    return data


def highpass_signals( 
        data: np.ndarray,  # (n_channels x n_samples)
        fsamp: float,
        high_pass: float = 20,
        method: Literal["butter", "firwin2"] = "butter",
        order: int | None = 2,
        numtabs: int | None = 101,
) -> np.ndarray:
    """
    High-pass filter timeseries data using a digital infinite 
    impulse  response filter ("butter") or finite impulse 
    response filter ("firwin2").

    Parameters
    ----------
    data : np.ndarray
        Input data of shape ``(n_channels, n_samples)``
    fsamp : float 
        Sampling frequency in Hz
    high_pass : float, default 20 
        Cut-off frequency for the high-pass filter in Hz    
    method :  {"butter", "firwin2"}, default "butter"
        Filter type
    order : int | None, default 2 
        Order of the filter (required if ``method==butter``) 
    numtabs : int | None, default 101 
        Number of filter tabs (required if ``method==firwin2``)

    Returns
    -------
    data : np.ndarray  
        filtered data of shape ``(n_channels x n_samples)``

    Examples
    --------
    Generate 5 seconds of white Gaussian noise and obtain 
    colored noise using a second order butterworth filter

    >>> from muniverse.algorithms.core import highpass_signals
    >>> fsamp = 2048
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((3, int(fsamp*5))) 
    >>> data_filt = highpass_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     high_pass=10,
    ...     method="butter",
    ...     order=2
    ... )

    Filter the same data using a finite impulse response filter   

    >>> data_filt = bandpass_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     high_pass=10,
    ...     method="firwin2",
    ...     numtabs=101
    ... )


    """

    if high_pass <= 0:
        raise ValueError("high_pass must be > 0")

    if method == "butter":
        if order is None:
            raise ValueError("order must be provided for butter")

        b, a = butter(order, high_pass, fs=fsamp, btype="highpass")
        data = filtfilt(b, a, data, axis=1)

    elif method == "firwin2":
        if numtabs is None:
            raise ValueError("numtabs must be provided for firwin2")

        nyq = fsamp / 2

        f = [0, high_pass * 0.9, high_pass, nyq]
        m = [0, 0, 1, 1]

        fir_coeff = firwin2(numtabs, f, m, fs=fsamp)
        data = filtfilt(fir_coeff, [1.0], data, axis=1)

    else:
        raise ValueError("method must be 'butter' or 'firwin2'")

    return data

def lowpass_signals(
        data: np.ndarray,  # (n_channels x n_samples)
        fsamp: float,
        low_pass: float = 500,
        method: Literal["butter", "firwin2"] = "butter",
        order: int | None = 2,
        numtabs: int | None = 101,
) -> np.ndarray:
    """
    Low-pass filter timeseries data using a digital infinite 
    impulse  response filter (``butter``) or finite impulse 
    response filter (``firwin2``).

    Parameters
    ----------
    data : np.ndarray
        Input data of shape ``(n_channels, n_samples)``
    fsamp : float 
        Sampling frequency in Hz  
    low_pass : float, default 500 
        Cut-off frequency for the low-pass filter in Hz
    method :  {"butter", "firwin2"}, default "butter"
        Filter type
    order : int | None, default 2 
        Order of the filter (required if ``method==butter``) 
    numtabs : int | None, default 101 
        Number of filter tabs (required if ``method==firwin2``)

    Returns
    -------
    data : np.ndarray  
        filtered data of shape ``(n_channels, n_samples)``

    Examples
    --------
    Generate 5 seconds of white Gaussian noise and obtain 
    colored noise using a second order butterworth filter

    >>> from muniverse.algorithms.core import lowpass_signals
    >>> fsamp = 2048
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((3, int(fsamp*5))) 
    >>> data_filt = lowpass_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     low_pass=450,
    ...     method="butter",
    ...     order=2
    ... )

    Filter the same data using a finite impulse response filter   

    >>> data_filt = lowpass_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     low_pass=450,
    ...     method="firwin2",
    ...     numtabs=101
    ... )    

    """

    nyq = fsamp / 2

    if low_pass <= 0 or low_pass >= nyq:
        raise ValueError("low_pass must be between 0 and Nyquist frequency")

    if method == "butter":
        if order is None:
            raise ValueError("order must be provided for butter")

        b, a = butter(order, low_pass, fs=fsamp, btype="lowpass")
        data = filtfilt(b, a, data, axis=1)

    elif method == "firwin2":
        if numtabs is None:
            raise ValueError("numtabs must be provided for firwin2")

        f = [0, low_pass, low_pass * 1.1, nyq]
        m = [1, 1, 0, 0]

        fir_coeff = firwin2(numtabs, f, m, fs=fsamp)
        data = filtfilt(fir_coeff, [1.0], data, axis=1)

    else:
        raise ValueError("method must be 'butter' or 'firwin2'")

    return data

def notch_signals(
        data: np.ndarray, 
        fsamp: float, 
        freqs: List[float] = [50, 100, 150], 
        method: Literal[
            "butter", "iirnotch", "fft_nulling", "fft_interpolation"
        ] = "butter", 
        order: int | None = 2, 
        dfreq: int | None = 1,
    ) -> np.ndarray:
    """
    Notch filter (stop band) time series data using either a infinite impulse 
    response filter ("butter"), a finite impulse response filter ("iirnotch") or 
    performing filtering in the frequency domain ("fft_nulling" and "fft_interpolation"). 
    For "fft_nulling" the spectrum in the specified frequency band is set to zero, 
    for "fft_interpolation" the spectral amplitude is interpolated through by the 
    neighbourhood. Time series data is then recovered through an inverse fft.

    Parameters
    ----------

    data : np.ndarray
        Input data of shape ``(n_channels, n_samples)``
    fsamp : float 
        Sampling frequency in Hz
    freqs : list of float, default [50, 100, 150] 
        List of frequencies to be notch filtered
    method : {"butter", "iirnotch", "fft_nulling", "fft_interpolation"}, default "butter"
        Filter type 
    order : int or None, default 2
        Order of the filter (if method is ``butter``)
    dfreq : float or None, default 1 
        Width of the notch filter (in both directions) in Hz 
        (if method is iirnotch, fft_nulling, fft_interpolation).

    Returns
    -------

    data : np.ndarray 
        Filtered data of shape ``(n_channels, n_samples)``

    Examples
    --------

    Notch filter 50 Hz power line noise and two harmonics
    using a second order butterworth filter

    >>> from muniverse.algorithms.core import notch_signals
    >>> fsamp = 2048
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((3, int(fsamp * 5))) 
    >>> data_filt = notch_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     freqs=[50, 100, 150],
    ...     method="butter",
    ...     order=2
    ... )

    Filter the same data by setting the fft to zero in the 
    desired frequency band  

    >>> data_filt = notch_signals(
    ...     data=data,
    ...     fsamp=fsamp,
    ...     low_pass=[50, 100, 150],
    ...     method="fft_nulling",
    ...     dfreq=1
    ... ) 


    """

    if isinstance(freqs, float) or isinstance(freqs, int):
        freqs = [freqs]

    if method == "butter":

        for f0 in freqs:
            b, a = butter(
                order,
                [f0 - dfreq, f0 + dfreq],
                fs=fsamp,
                btype="bandstop",
            )
            data = filtfilt(b, a, data, axis=1)

    elif method == "iirnotch":
        for f0 in freqs:
            b, a = iirnotch(f0, f0/(2*dfreq), fsamp)
            data = filtfilt(b, a, data, axis=1)        

    elif method == "fft_nulling":
        N = data.shape[1]

        spectrum = rfft(data, axis=1)
        fft_freqs = rfftfreq(N, d=1/fsamp)

        for f0 in freqs:

            # Create notch mask (1D)
            mask = np.abs(fft_freqs - f0) <= dfreq
    
            # Broadcast mask across channels
            spectrum[:, mask] = 0

        data = irfft(spectrum, n=N, axis=1)

    elif method == "fft_interpolation":
        N = data.shape[1]

        spectrum = rfft(data, axis=1)
        fft_freqs = rfftfreq(N, d=1/fsamp)

        eps = 1e-12  # avoid log(0)

        for f0 in freqs:

            mask = np.abs(fft_freqs - f0) <= dfreq
            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            left = idx[0] - 1
            right = idx[-1] + 1

            # Handle edge cases
            if left < 0 or right >= len(fft_freqs):
                continue

            # magnitude and phase ---
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)

            # log-magnitude
            log_mag = np.log(mag + eps)

            # values at boundaries
            left_log = log_mag[:, left]   
            right_log = log_mag[:, right]

            # interpolate in log space
            interp_log = np.linspace(
                left_log,
                right_log,
                len(idx),
                axis=1
            )
            # transform to back to magnitude space
            interp_mag = np.exp(interp_log)

            # use average phase from edges (avoids phase jumps)
            left_phase = phase[:, left]
            right_phase = phase[:, right]

            # unwrap phase to avoid discontinuities
            phase_pair = np.stack([left_phase, right_phase], axis=1)
            phase_pair = np.unwrap(phase_pair, axis=1)

            interp_phase = np.linspace(
                phase_pair[:, 0],
                phase_pair[:, 1],
                len(idx),
                axis=1
            )

            # reconstruct spectrum
            spectrum[:, idx] = interp_mag * np.exp(1j * interp_phase)

        data = irfft(spectrum, n=N, axis=1)

    else:
        raise ValueError(
            f"The specified filter type option {method} is invalid"
            "Valid options are *butter*, *iirnotch*, *fft_nulling* or *fft_interpolation*"                      
        )

    return data

def find_outliers(
        x: np.ndarray, # (n_features, )
        threshold: float = 3, 
        max_iter: int = 3, 
        mode: Literal["above", "below", "two-sided"] = "two-sided",
        mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Detect ouliers by comparing the z-score of variable x against
    some threshold. This is repeaded until there are no outliers or
    the maximum number of iterations is reached. 

    Parameters
    ----------
    x : np.ndarray 
        Variable to test for outliers with shape ``(n_features, )``
    threshold : float, default 3 
        Threshold for outlier detection
    max_iter: int , default 3
            Maximum number of iterations
    mode : {"above", "below", "two-sided"} , default "two-sided"
        Specify weather to serach for outliers   
        on both ends ("two-sided"), just on the positive ("above") 
        or just the negative side ("below").
    mask : np.ndarray | None , default None
        Boolean mask to exclude channels from outlier detection
        (True: outlier, False: no outlier)    

    Returns
    -------
    mask : np.ndarray (n_features, )
        Boolean mask (True: outlier, False: no outlier)

    Examples
    --------

    Generate random numbers from a standard normal distribution
    and add one outlier. 

    >>> import numpy as np
    >>> from muniverse.algorithms.core import find_outliers
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal(20)
    >>> X[0] = 100
    >>> find_outliers(X, threshold=3, max_iter=1, mode="two_sided")
    array([True, False, False, False, False, False, False, 
        False, False, False, False, False, False, False, False, 
        False, False, False, False, False])

    Now consider only ousiders on the left side of the distribution  

    >>> find_outliers(X, threshold=3, max_iter=1, mode="below") 
    array([False, False, False, False, False, False, False, 
        False, False, False, False, False, False, False, False, 
        False, False, False, False, False]) 


    """

    if mask is None:
        mask = np.zeros(len(x), dtype=bool)

    iter = 0
    while iter < max_iter:
        xm = np.ma.masked_array(x, mask=mask)
        if mode == "above":
            idx = zscore(xm) > threshold
        elif mode == "below":
            idx = -zscore(xm) > threshold
        else:
            idx = np.abs(zscore(xm)) > threshold 
        mask = mask | idx.data
        if not np.any(idx):
            break
        else:
            iter = iter + 1 

    return mask  

def extension(
        Y: np.ndarray, # (n_channels x n_samples)
        R: int
) -> np.ndarray:
    """
    Extend a multi-channel signal Y by an extension factor R
    using Toeplitz matrices.

    Parameters
    ----------
    Y : np.ndarray 
        Input signal with shape ``(n_features x n_samples)``
    R : int 
        Extension factor (number of lags)

    Returns
    -------
    eY : np.ndarray 
        Extended signal with shape ``(n_channels x (R * n_samples))``

    Examples
    --------

    Make a simple test signal and extend it by a factor of 2:

    >>> import numpy as np
    >>> from muniverse.algorithms.core import extension
    >>> Y = np.arange(8).reshape(2, 4)
    >>> extension(Y, 2)
    array([
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [4, 5, 6, 7],
        [0, 4, 5, 6]
    ])

    """
    n_channels, n_samples = Y.shape
    eY = np.zeros((n_channels * R, n_samples)).astype(Y.dtype)

    for i in range(n_channels):
        col = np.concatenate(([Y[i, 0]], np.zeros(R - 1)))
        row = Y[i, :]
        T = toeplitz(col, row)
        eY[i * R : (i + 1) * R, :] = T

    return eY


def whitening(
        Y: np.ndarray, # (n_channels x n_samples)
        method: Literal["ZCA", "PCA", "Cholesky"] = "ZCA", 
        backend: Literal["ed", "svd"] = "ed", 
        regularization: str | float | None = "auto", 
        eps: Optional[float] = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Whiten data using the ZCA, PCA, or Cholesky method.

    Parameters
    ----------
    Y : np.ndarray 
        Input signal with shape ``(n_features x n_samples)``

    method : {"ZCA", "PCA", "Cholesky"}, default "ZCA" 
        Whitening method 

    backend : {"ed", "svd"}, default "ed" 
        Method used to calculate eigenvalues and eigenvectors. Can be
        either based on singular value decomposition ("svd") or an
        eigendecomposition ("ed"). Only needed if method is "ZCA" or "PCA".

    regularization : {"auto", float, None}, default "auto" 
        Adds a small value to the eigenvalues for regularization.
        If "auto", the mean of the second half of the eigenvalues is used.

    eps : float 
        Small epsilon added to the eigenvalues for numerical stability

    Returns
    -------
    wY : np.ndarray)
        Whitened signal with shape ``(n_features, n_samples)``

    Z : np.ndarray 
        Whitening matrix ``(n_features, n_features)``

    Z_inv : np.ndarray 
        Inverse of the whitening matrix ``(n_features, n_features)``    

    Examples
    --------

    Perform ZCA whitening on a simple test signal
    with analytical solution:
    
    >>> import numpy as np
    >>> from muniverse.algorithms.core import whitening
    >>> X = np.array([
    ...     [1, -1, 0, 0], 
    ...     [0, 0, 2, -2]
    ... ])
    >>> Xw_ref = np.array([
    ...     [np.sqrt(3/2), -np.sqrt(3/2), 0, 0], 
    ...     [0, 0, np.sqrt(3/2), -np.sqrt(3/2)]
    ... ])
    >>> Z_ref = np.array([
    ...     [np.sqrt(3/2), 0],
    ...     [0, np.sqrt(3/8)]
    ... ])
    >>> Xw, Z, Z_inv = whitening(X, 
    ...     method="ZCA", backend="ed",
    ...     regularization=None, eps=0
    ... )
    >>> np.allclose(Xw, Xw_ref, atol=1e-8, rtol=1e-5)  
    True
    >>> np.allclose(Z, Z_ref, atol=1e-8, rtol=1e-5)  
    True

    """
    n_channels, n_samples = Y.shape
    use_svd = backend == "svd"

    if method == "Cholesky":
        covariance = Y @ Y.T / (n_samples - 1)
        L = np.linalg.cholesky(covariance)
        Z = np.linalg.inv(L)
        Z_inv = L
        wY = Z @ Y
        return wY, Z, Z_inv

    # Use SVD
    if use_svd:
        covariance = Y @ Y.T / (n_samples - 1)
        # covariance = np.cov(Y)
        U, S, _ = np.linalg.svd(covariance, full_matrices=False)
        if regularization == "auto":
            reg = np.mean(S[len(S) // 2 :] ** 2)
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        #S_reg = np.sqrt(S + reg + eps)   
        S_reg = np.sqrt(np.maximum(S + reg + eps, eps))
        S_inv = 1.0 / S_reg

        if method == "ZCA":
            Z = U @ np.diag(S_inv) @ U.T
            Z_inv = U @ np.diag(S_reg) @ U.T
        elif method == "PCA":
            Z = np.diag(S_inv) @ U.T
            Z_inv = np.diag(S_reg) @ U.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    # Use EIG
    else:
        covariance = Y @ Y.T / (n_samples - 1)
        S, V = np.linalg.eigh(covariance)

        if regularization == "auto":
            reg = np.mean(S[: len(S) // 2])
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        #S_reg = np.sqrt(S + reg + eps)
        S_reg = np.sqrt(np.maximum(S + reg + eps, eps))    
        S_inv = 1.0 / S_reg

        if method == "ZCA":
            Z = V @ np.diag(S_inv) @ V.T
            Z_inv = V @ np.diag(S_reg) @ V.T
        elif method == "PCA":
            Z = np.diag(S_inv) @ V.T
            Z_inv = np.diag(S_reg) @ V.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    return wY, Z, Z_inv


def est_spike_times(
        source: np.ndarray, # (n_samples, ) 
        fsamp: float, 
        a: float = 2, 
        min_delay: float = 0.01
) -> tuple[np.ndarray, float]:
    """
    Estimate spike indices given a spiky source signal and compute
    a silhouette-like metric for source quality quantification.
    To do so, (i) a asymetric power law is applied to the signal,
    (ii) a peak detection method identifies spike candidates that
    are (iii) sorted with kmeans++ to estimate true and false spikes.

    Parameters
    ----------
    source : np.ndarray 
        Spike-like source signal with shape ``(n_samples, )``
    fsamp : float 
        Sampling rate in Hz
    cluster : {"kmeans"}, default "kmeans"
        Clustering method used to identify the spike indices. Currently,
        only kmeans++ is implemented
    a : float , default 2
        Exponent of assymetric power law used for contrast enhancement
    min_delay : float , default 0.01 
        Mimium distance between two spikes (used for peak detection)   

    Returns
    -------
    est_spikes : np.ndarray 
        Array of the Estimated spike indices
    sil : float 
        Silhouette-like score (0 = poor, 1 = strong separation)

    Examples
    --------

    Generate a spiky signals as the superposition of random noise
    and spikes. Then estimate the spike times and calculate a 
    silhouette-like score to quantify the separation of the detected 
    peaks from the noise

    >>> import numpy as np
    >>> from muniverse.algorithms.core import est_spike_times
    >>> fsamp = 2000
    >>> rng = np.random.default_rng(42)
    >>> noise = rng.standard_normal(int(fsamp * 5))  
    >>> spikes = np.arange(int(fsamp * 1), int(fsamp * 4), 400)
    >>> noise[spikes] += 10
    >>> est_spikes, sil = est_spike_times(noise, fsamp)
    >>> np.all(spikes == est_spikes)
    np.True_
    >>> sil
    np.float64(0.9825454841122653)           

    """

    # Assymetric power law that can be useful for contrast enhancement
    sig = np.sign(source) * source**a

    # Apply a peak detection method
    min_peak_dist = int(round(fsamp * min_delay))
    peaks, _ = find_peaks(sig, distance=min_peak_dist)

    if len(peaks) == 0:
        return np.array([])

    # Get peak values
    peak_vals = sig[peaks].reshape(-1, 1)

    # K-means clustering to separate signal vs. noise
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(peak_vals)
    centroids = kmeans.cluster_centers_.flatten()

    # Spikes are those in the cluster with the higher mean
    spike_cluster = np.argmax(centroids)
    est_spikes = peaks[labels == spike_cluster]

    # Compute within- and between-cluster distances
    D = kmeans.transform(peak_vals)  # Distances to both centroids
    sumd = np.sum(
        D[labels == spike_cluster, spike_cluster] ** 2
    )  # Exponent 2 for obtaining the squared Euclidian distance
    between = np.sum(
        D[labels == spike_cluster, 1 - spike_cluster] ** 2
    )  # Exponent 2 for obtaining the squared Euclidian distance

    # Silhouette-inspired score
    denom = max(sumd, between)
    sil = (between - sumd) / denom if denom > 0 else 0.0   

    return est_spikes, sil


def gram_schmidt(
        w: np.ndarray, # (n, )
        B: np.ndarray  # (n, k)
) -> np.ndarray:
    """
    Stabilized Gram-Schmidt orthogonalization.

    Parameters
    ---------
    w : np.ndarray 
        Vector to be orthogonalized with shape ``(n_features, )``
    B : np.ndarray 
        Matrix of basis vectors with shape ``(n_features, n_columns)``.
        Columns containing zero vectors are ignored

    Returns
    -------
    u : np.ndarray 
        Orthogonalized vector with shape ``(n_features, )``

    Examples
    --------

    Orthogonalize vector ``v`` given the basis vecors stored 
    in matrix ``B``

    >>> import numpy as np
    >>> from muniverse.algorithms.core import gram_schmidt
    >>> B = np.array([
    ...     [1, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 0, 0]    
    ... ])
    >>> v = np.array([2, 3, 4])
    >>> gram_schmidt(v, B)
    array([0., 0., 4.])

    """
    #w = np.asarray(w, dtype=float)
    B = np.asarray(B, dtype=w.dtype)

    # Remove zero columns from B
    non_zero_cols = ~np.all(B == 0, axis=0)
    B = B[:, non_zero_cols]

    u = w.copy()
    for i in range(B.shape[1]):
        a = B[:, i]
        projection = (np.dot(u, a) / np.dot(a, a)) * a
        u = u - projection

    return u

def spike_triggered_average(
        sig: np.ndarray, # (n_channels, n_samples) 
        spikes: np.ndarray, # (n_spikes, ) 
        win: float = 0.02, 
        fsamp: float = 2048
) -> np.ndarray:
    """
    Estimate the impulse response of a finite impulse response filters 
    given the time samples of the events

    Parameters
    ----------
    sig : np.ndarray 
        Input signal with shape ``(n_features, n_samples)``
    spikes : np.ndarray 
        Array of spike indices with shape ``(n_spikes, )``
    win : float , default 0.02
        Window size (in both directions) in seconds used for 
        impulse response extraction     
    fsamp : float , default 2048
        Sampling frequency in Hz

    Returns
    -------
    waveform : np.ndarray 
        Estimated impulse response of a given source 
        with shape ``(n_features, n_window)``

    """

    width = int(win * fsamp)
    waveform = np.zeros((sig.shape[0], 2 * width + 1))

    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]

    for i in np.arange(len(spikes)):
        waveform = waveform + sig[:, (spikes[i] - width) : (spikes[i] + width + 1)]

    waveform = waveform / len(spikes)

    return waveform


def peel_off(
        sig: np.ndarray, # (n_channels, n_samples) 
        spikes: np.ndarray, # (n_spikes, ) 
        win: float = 0.02, 
        fsamp: float = 2048,
        method: Literal["sparse", "fft_conv"] = "sparse"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Peel off the signal contribution of a source with finite impulse
    response filter given the time stamps of the impulses (spikes) 
    using spike triggered averaging. The reconstruction of the 
    component signal is either achieved in the frequency domain (fft/ifft)
    or through sparse template subtraction.

    Parameters
    ----------
    sig : np.ndarray 
        Input signal with shape ``(n_features, n_samples)``
    spikes : np.ndarray 
        Array of spike indices with shape ``(n_spikes, ) ``
    win : float , default 0.02
        Window size in seconds for MUAP template     
    fsamp : float , default 2048
        Sampling frequency in Hz
    method : {"sparse", "fft_conv"}  
        Method used for peel-of. If "sparse" (default), 
        the function loops over all spikes and inserts a 
        the waveform reconstructed from spike-triggered averaging. 
        If "fft_conv" the signal is reconstructed by convolving
        (fft-based) the waveform with the spike train.  

    Returns
    -------
    residual_sig : np.ndarray 
        Residual signal after removing component with shape ``(n_features, n_samples)``
    comp_sig : np.ndarray 
        Estimated contribution of the given source with shape ``(n_features, n_samples)``
    waveform : np.ndarray 
        Impulse response of the given component with shape ``(n_channels, n_waveform)``    
    """

    waveform = spike_triggered_average(sig, spikes, win, fsamp)

    width = int(win * fsamp)
    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]

    if method == "sparse":
        comp_sig = np.zeros_like(sig)
        for s in spikes:
            start = s - width
            end = s + width + 1
            comp_sig[:, start:end] = waveform

    elif method == "fft_conv":
        firings = np.zeros(sig.shape[1])
        firings[spikes] = 1

        # Zero-pad waveform to match signal shape
        L = sig.shape[1]
        pad_len = L - waveform.shape[1]
        waveform_padded = np.pad(
            waveform, ((0, 0), (0, pad_len)), mode="constant"
        )

        # FFT of firings (same for all channels)
        fft_firings = rfft(firings)

        # FFT of waveform for each channel
        fft_waveform = rfft(waveform_padded, axis=1)

        # Multiply in frequency domain (broadcasting firings to each channel)
        fft_product = fft_waveform * fft_firings

        # IFFT to get time domain component signal
        comp_sig = irfft(fft_product, n=L, axis=1)

        # Correct time shift due to FFT convolution (center of kernel)
        shift = (waveform.shape[1] - 1) // 2
        comp_sig = np.roll(comp_sig, -shift, axis=1)

    residual_sig = sig - comp_sig

    return residual_sig, comp_sig, waveform


def spike_dict_to_long_df(
        spike_dict: dict, 
        fsamp: float = 2048
) -> pd.DataFrame:
    """
    Convert a dictionary of spike instances into a long-formatted 
    spike table. The table follows BIDS-events files and has
    the columns: 
    - "onset": Time of the event in seconds
    - "duration": Duration of the event (0 for neural spikes) 
    - "sample": Sample indice of the event  
    - "unit_id": Unique unit ID (integer)
    - "event_type": Event classifier (here: "motor-unit-spike")

    Parameters
    ----------
    spike_dict : dict 
        Dictonary of spike times {unit_id (int): list(int)}
    fsamp : float, default 2048
        Sampling frequency in Hz

    Returns
    -------
    df : pd.DataFrame 
        Table of motor unit spikes
    """

    columns = ["onset", "duration", "sample", "unit_id", "event_type"]

    rows = []
    for unit_id, spikes in spike_dict.items():
        for t in spikes:
            rows.append({
                "onset": t / fsamp,
                "duration": 0,
                "sample": t,
                "unit_id": unit_id, 
                "event_type": "motor-unit-spike"
            })

    # If no spikes were found, create an empty DataFrame 
    if not rows:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)

    # Drop dublicates and sort by onset
    df = df.drop_duplicates(subset=["onset", "unit_id", "sample"])
    df = df.sort_values(by=["onset"])
    df.reset_index(drop=True, inplace=True)

    return df


def get_duplicates_mask(
        spikes: pd.DataFrame,
        scores: np.ndarray, 
        fsamp: float, 
        mode: Literal["max", "min", "first"] = "max",
        t_start: float = 0, 
        t_end: float = -1,
        duplicate_theshold: float = 0.3,
        max_shift: float = 0.01,
        tol: float = 0.001,
        mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Idendify duplicates spike trains and only keep for each 
    unique unit label the source with the best quality score

    Parameters
    ----------
    spikes : pd.DataFrame
        Long-table dictonary of spikes

    scores : np.ndarray 
        Array of quality metrics

    fsamp : float 
        Sampling rate in Hz

    mode : {"max", "min", "first"} , default "max"         
        Whether to keep the source with the maximal score ("max"),
        the minimal score ("min"), or the first copy ("first").

    duplicate_theshold : float , default 0.3
        Minimum fraction of common spikes to classify 
        two units identical       

    max_shift : float , default 0.01
        Maximal delay between two spike trains in seconds

    tol : float , default 0.001
        All spikes with a delay lower than the given tolerance 
        (in seconds) are classified identical

    Returns
    -------
    keep_mask : np.ndarray of bool 
        Boolean mask with shape ``(n_units, )``;
        True: keep source, False: reject source

    new_labels : np.ndarray of int (n_units,)
        New label for each source   


    """

    if mask is not None:
        idx = np.where(~mask)
        scores[idx] = -1

    units = sorted(spikes["unit_id"].unique())
    n_source = len(units)
    keep_mask = np.zeros(n_source, dtype=bool)

    new_labels, _ = label_sources(
        spikes, fsamp=fsamp, t_start=t_start, t_end=t_end, 
        threshold=duplicate_theshold, max_shift=max_shift, tol=tol
    )

    unique_labels = np.unique(new_labels)

    for label in unique_labels:
        idx = np.where(new_labels == label)[0]

        # pick best according to score (or keep the first element)
        if mode == "max":
            best_idx = idx[np.argmax(scores[idx])]
        elif mode == "min":
            best_idx = idx[np.argmin(scores[idx])]
        else:
            best_idx = idx[0] # mode == "first"

        keep_mask[best_idx] = True 

    return keep_mask, new_labels

def get_bad_source_mask(
        spikes: pd.DataFrame,
        score: np.ndarray,
        threshold: float = 0.9,
        mode: Literal["below", "above"] = "below",
        min_num_spikes: int = 10
) -> np.ndarray:
    """
    Generate a boolean mask that filters out bad sources 
    based on a quality score and minimum number of spikes. 

    Parameters
    ----------
    spikes : pd.DataFrame
        Spike data frame
    score : np.ndarray
        Vector of scores
    threshold : float
        Threshold used classify bad and good sources
    mode : {"below", "above"} , default "below"         
        Weather values below or above the threshold are
        considered bad
    min_num_spikes : int
        Minimum number of spikes required for a good source

    Returns
    -------
    keep_mask : np.ndarray 
        Boolean mask with shape ``(n_units ,)``;
        True: keep source; False: reject source 

    Examples
    --------

    Make 10 spikes correspodning to two units with quality 
    score of 0.89 and 0.93. We want to reject all units
    with a score below 0.9 and less than 3 spikes    

    >>> import numpy as np
    >>> import pandas as pd
    >>> from muniverse.algorithms.core import get_bad_source_mask
    >>> rng = np.random.default_rng(42)
    >>> spikes = {
    ...     "onset": np.arange(10),
    ...     "duration": np.zeros(10),
    ...     "unit_id": rng.integers(0, 2, 10)
    ... }   
    >>> spikes = pd.DataFrame(spikes)
    >>> spikes["unit_id"].to_numpy()
    array([0, 1, 1, 0, 0, 1, 0, 1, 0, 0])
    >>> scores = np.array([0.89, 0.93])
    >>> get_bad_source_mask(
    ... spikes, scores, threshold=0.9, mode="below", min_num_spikes=3
    ... )
    array([False,  True])

    Now we want to reject units with less than 5 spikes and a score 
    below 0.87

    >>> get_bad_source_mask(
    ... spikes, scores, threshold=0.87, mode="below", min_num_spikes=5
    ... )
    array([True, False])

    """

    units = sorted(spikes["unit_id"].unique())
    n_source = len(score)

    # Initialize mask 
    keep_mask = np.ones(n_source, dtype=bool)

    # Reject bad sources
    for i in range(n_source):

        local_spikes = spikes[spikes["unit_id"] == units[i]]["onset"].values
        n_spikes = len(local_spikes)  
        if n_spikes < min_num_spikes:
            keep_mask[i] = False 

        if mode == "below" and score[i] < threshold:
            keep_mask[i] = False
        elif mode == "above" and score[i] > threshold:
            keep_mask[i] = False

    return keep_mask

def filter_spikes(    
    spikes: pd.DataFrame, 
    keep_mask: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """
    Filter units in an events table using a boolean mask.

    Parameters
    ----------
    spikes : pd.DataFrame
        BIDS events table. Must contain a column ``unit_id`` 
    keep_mask : np.ndarray (bool)
        Boolean mask of shape ``(n_units, )`` indicating sources
        that should be kept (True) or rejected (False)

    Returns
    -------
    spikes : pd.DataFrame
        Filtered spikes with new ``unit_id`` labels 
    label_mapping : dict
        Mapping of from old to new ``unit_id`` labels
    """

    spikes = spikes.copy()

    # --- keep only valid units ---
    valid_units = np.where(keep_mask)[0]
    spikes = spikes[spikes["unit_id"].isin(valid_units)]

    # --- remap labels to 0..N-1 ---
    unique_units = np.sort(spikes["unit_id"].unique())

    label_map = {old: new for new, old in enumerate(unique_units)}

    spikes["unit_id"] = spikes["unit_id"].map(label_map)

    return spikes, label_map 

def map_spikes(    
    spikes: pd.DataFrame, 
    fsamp: float,
    t_start: float
) -> tuple[pd.DataFrame, dict]:
    """
    Apply a temporal shift to all events in a BIDS-events table

    Parameters
    ----------
    spikes : pd.DataFrame
        BIDS events table. Must contain the columns ``onset`` and ``sample``
    fsamp : float
        Sampling rate in Hz
    t_start : float
        Global reference time frame (in seconds) 

    Returns
    -------
    spikes : pd.DataFrame
        Temporally mapped events
    """

    spikes = spikes.copy()

    spikes["onset"] = spikes["onset"] + t_start
    spikes["sample"] = spikes["sample"] + int(t_start * fsamp)

    return spikes      

