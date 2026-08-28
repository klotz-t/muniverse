import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
from .core import (
    bandpass_signals, 
    notch_signals, 
    highpass_signals, 
    lowpass_signals,
    find_outliers,
    peel_off
)


class PreProcessEMG:
    """
    Class to preprocess HD-EMG data. Availible steps include 
    (for details see below)

    - Bandpass, highpass, lowpass and notch filtering
    - Automatically detect bad channels
    - Segment data and mask bad channels
    - Downsample data
    - Peel off contributions from already decomposed MUs
    - Compute basic signal metrics 

    Parameters
    ----------
    steps : list of dict
        List of preprocessing steps. Each step is a dictionary describing
        the processing operation.

    Examples
    --------
    Pre process multi-channel high density EMG data of 
    shape ``(n_channels, n_samples)``

    1.) Apply a second order butterworth-type bandpass filter

    2.) Reject the power line interference (50 Hz) and two 
        harmonics using a fisrt order butterworth filter

    >>> from muniverse.algorithms.pre_processing import PreProcessEMG
    >>> model = PreProcessEMG(steps = [
    ...     {
    ...         "step": "bandpass",
    ...         "high_pass": 20,
    ...         "low_pass": 500,
    ...         "method": "butter",
    ...         "order": 2
    ...     },
    ...     {
    ...         "step": "notch",
    ...         "freqs": [50, 100, 150],
    ...         "method": "butter",
    ...         "order": 1
    ...     },
    ... ])
    >>> preprocessed_data, metadata = model.pre_process(
    ...     data=emg_data, fsamp=2048
    ... )     

                        
    """

    def __init__(
            self, 
            steps: list[dict] = []          
    ):

        #self.pre_process_steps = pre_process_steps
        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]

    class Bandpass(BaseModel):
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

        Examples
        --------

        Preprocess multi-channel EMG data using a fisrt order butterwoth 
        bandpass filter

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "bandpass",
        ...     "high_pass": 20,
        ...     "low_pass": 500,
        ...     "method": "butter",
        ...     "order": 1
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """    
        step: Literal["bandpass"]
        high_pass: float = 20
        low_pass: float = 500
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101

    class Highpass(BaseModel):
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
        

        Examples
        --------

        Apply a 10 Hz highpass filter using a finite impulse response
        ``firwin2``filter

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "highpass",
        ...     "high_pass": 10,
        ...     "method": "firwin2",
        ...     "numtabs": 101
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """  
        step: Literal["highpass"]
        high_pass: float = 20
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101 

    class Lowpass(BaseModel):
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
                
        
        Examples
        --------

        Apply a low pass filter at 450 Hz using a 4-th order 
        ``butterworth``filter

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "lowpass",
        ...     "low_pass": 450,
        ...     "method": "butter",
        ...     "order": 4
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """ 
        step: Literal["lowpass"]
        low_pass: float = 500
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101       

    class Notch(BaseModel):
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
                
        
        Examples
        --------

        Apply a notch filter at the 50 Hz power line frequency and the 
        100 and 150 Hz harmonics using ``fft_nulling``

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "notch",
        ...     "freqs": [50, 100, 150],
        ...     "method": "fft_nulling",
        ...     "dfreq": 1 
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """ 
        step: Literal["notch"]
        freqs: List[float] = [50, 100, 150]
        method: Literal[
            "butter", "iirnotch", "fft_nulling", "fft_interpolation"
            ] = "butter" 
        order: int = 2
        dfreq: float = 1
   
    class BadChannelDetection(BaseModel):
        """
        Automatically detect bad channels based on some per channel
        metric computed in a given time window (given in seconds). 
        If method is ``zscore`` the score distribution is normalized (zero mean, 
        unit variance). All scores are compared to a "threshold_value". If mode 
        is "above", all values above the threshold are rejected, if mode is "below", 
        all values below the theshold are rejected. For mode=="two-sided", the 
        absolute value of the score is computed and all values above the threshold 
        are rejected (only availible if the selected method is ``zscore``)  

        metric : {"std", "rms", "medfreq", "medpower"}
            Metric used to detect outlier channels. Availible options
            are standard deviation "std" the root mean square ("rms"),
            the median frequency content ("medfreq") or the median power
            of the signal ("medpower")
        window : {tuple, None} , default None
            Time window (in seconds) in which the specified metric is 
            calculated. If ``None``the full length of the signal is considered 
        method : {"zscore", "threshold"} , default "zscore"
            Method used for outlier detection. If ``zscore``, all values 
            are zscore normalized and outlier thresholds are specified by assuming
            a standard normal distribution. When using ``threshold``, a fixed 
            theshold for outlier detection is used. 
        threshold_value : float , default 3 
            Values above/below this threshold are considered bad channels
        mode : {"above", "below", "two-sided"} , default "two-sided"
            Specify weather to serach for outliers above the threshold ("above") 
            or below the thershold ("below"). If the slected method is ``zscore``
            you can also select "two-sided" to serach for outliers on
            both ends of the distribution.
        max_iter : int , default 3
            Needed if the slected method is ``zscore``. Specifies the number
            of iterations that are used for outlier detection  
        bandwidth : {tuple, None} , default None
            For the spectral metrics "medfreq" and "medpow" you can specify a 
            tuple indicating a bandwidth of interest     
        description : str
            Short free-text description why the channel was rejected
            and that appears in the processing metadata     
             
                        
        Examples
        --------

        Automatically reject flat channels based on the zscore normalized 
        signal amplitude 

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "bad_channel_detection",
        ...     "metric": "rms",
        ...     "method": "zscore"
        ...     "threshold_value": 3,
        ...     "mode": "below",
        ...     "description": "Flat channel detected"
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """ 
        step: Literal["bad_channel_detection"]
        metric: Literal["std", "rms", "medfreq", "medpower", "cumpower"]
        method: Literal["zscore", "threshold"] = "zscore"
        threshold_value: float = 3
        max_iter: int | None = 3
        mode: Literal["above", "below", "two-sided"] = "two-sided"
        window: tuple[float, float] | None = None
        bandwidth: tuple[float, float] | None = None
        description: str = "Automatical bad channel detection"   

    class MaskChannels(BaseModel):
        """
        Mask all channels given in ``channel_list`` to be excluded in the following. 
        Can be either used to reject known bad channels or limit the analysis to a 
        subset of your data (e.g., only one EMG array)

        channel_list : list of int , default []
            List of ignored channels
        description : str   
            Short free-text description why the channel was rejected
            and that appears in the processing metadata  
            
        Examples
        --------

        Remove channels 60 and 61 that have been manually classified 
        to be bad channels

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "mask_channels",
        ...     "channel_list": [60, 61],
        ...     "description": "Manually detected bad channels"
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """ 
        step: Literal["mask_channels"]
        channel_list: list[int] = []  
        description: str = "Manually masked channel"  

    class Downsample(BaseModel):
        """
        Reduces the sampling frequency by the specified value 

        Parameters
        ----------
        
        factor : int 
            The factor by which the signal is downsamples
                
        Examples
        --------

        Downsample a signal sampled at 10240 Hz to 2048 Hz

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "downsample",
        ...     "factor": 5 
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """ 
        step: Literal["downsample"]
        factor: int

    class TimeWindow(BaseModel):
        """
        Restric your analysis to a selected time window.
            
        t_start : float
            Starting point of the region of interest
        t_end : float
            End point of the region of interest. If ``t_end = -1``,
            the time window ends with the last sample
                    
        Examples
        --------

        Select a time window from 5 to 25 seconds for your signal 
        analysis

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "time_window",
        ...     "t_start": 5.0,
        ...     "t_end": 25.0 
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """ 
        step: Literal["time_window"]
        t_start: float = 0
        t_end: float = -1

    class GetMetric(BaseModel):
        """
        Calculate for each channel the specified metric 
        and which will be reported in the processing metadata:

        Parameters
        ----------
        
        metric : {"std", "rms", "medfreq", "medpower"}
            Metric to be computed. Can be the standard deviation ("std"),
            the root mean square ("rms"), the median frequency ("medfreq")
            or the median power ("medpower") of the chanels 
        window : {tuple, None}
            Time window (in seconds) used to compute the specified metric.
            If ``None``, the full data is considered.
            
        
        Examples
        --------

        Calculate the median frequency of each channel in the time window
        from 7.5 to 22.5 seconds

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "get_metric",
        ...     "metric":"medfreq",
        ...     "window": (7.5, 22.5)
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048
        ... )

        """ 
        step: Literal["get_metric"]
        metric: Literal["std", "rms", "medfreq", "medpower", "cumpower"]
        window: tuple[float, float] | None = None
        bandwidth: tuple[float, float] | None = None 
        description: str = ""

    class PeelOff(BaseModel):
        """
        Peel of contributions of known MU activity to obtain
        a residual multi-channel EMG signal for further processing
    
        Parameters
        ----------
        window_size : float
            Half-length of the peel off window in seconds
    
        
        Examples
        --------

        Peel of the contribution of all motor units that are already
        decomposed with spike times stored in ``known_spikes" and using
        a peel off window of 0.025 seconds

        >>> from muniverse.algorithms.pre_processing import PreProcessEMG
        >>> steps = [{
        ...     "step": "peel_off",
        ...     "window_size": 0.025 
        ... }]
        >>> model = PreProcessEMG(steps=steps)
        >>> preprocessed_data, metadata = model.pre_process(
        ...     data=emg_data, fsamp=2048, spikes=known_spikes
        ... )

        """ 
        step: Literal["peel_off"]
        window_size: float = 0.02          

    PreprocessStep = Annotated[
        Union[Bandpass, 
            Lowpass,
            Highpass,  
            Notch, 
            BadChannelDetection, 
            MaskChannels, 
            Downsample,
            TimeWindow,
            GetMetric,
            PeelOff
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PreprocessStep)

    def add_step(self, step):
        """ 
        Add an additional post processing step
        
        Parameters
        ----------
        
        step : dict
            Dictonary with the parameters of the added processing step
        
        """
        
        self.steps.append(
            self._adapter.validate_python(step)
        )

    def _get_scores(
            self, 
            data: np.ndarray, # (n_channels, n_samples) 
            metric: Literal[
                "rms", "std", "medfreq", "medpower", "cumpower"
            ], 
            fsamp: float | None = 2048, 
            bw: tuple | None = (20, 500)
    ):
        """
        Calculate channel specific scores

        Args
        ----
            data : np.ndarray (n_channels, n_samples)
                time series data 
            metric : {"rms", "std", "medfreq", "medpower", "cumpower"}
                Specify the computed metric. Can be the root-mean-square ("rms"),
                the standard deviation ("std"), the median frequency ("medfreq"),
                the median power ("medpower") or the cumulative power ("cumpower")
            fsamp : float
                If your metric is "medfreq", "medpower", "cumpower" you need
                to specify the sampling rate in Hz
            bw : tuple of float
                If your metric is "medfreq", "medpower", "cumpower" you need
                to specify the considered bandwidth in Hz     

        Returns
        -------
            score : np.ndarray
                Array of score values (n_channels, )
        
        """

        METRICS = ["rms", "std", "medfreq", "medpower", "cumpower"]
                    
        if metric == "rms":
            score = np.mean(data**2, axis=1)**0.5
        elif metric == "std":
            score = np.std(data, axis=1)
        elif metric == "medfreq":
            freqs, psd = welch(data, fs=fsamp, nperseg=fsamp, noverlap=fsamp/2)
            idx = np.where((freqs > bw[0]) & (freqs < bw[1]))[0]
            cumulative = np.cumsum(psd[:, idx], axis=1)
            total = cumulative[:, -1][:, None]
            med_idx = np.argmax(cumulative >= total / 2, axis=1)
            score = freqs[med_idx]
        elif metric == "medpower":
            freqs, psd = welch(data, fs=fsamp, nperseg=fsamp, noverlap=fsamp/2)
            idx = np.where((freqs > bw[0]) & (freqs < bw[1]))[0]
            score = np.median(psd[:,idx], axis=1)
        elif metric == "cumpower":
            freqs, psd = welch(data, fs=fsamp, nperseg=fsamp, noverlap=fsamp/2)
            idx = np.where((freqs > bw[0]) & (freqs < bw[1]))[0]
            score = np.sum(psd[:,idx], axis=1)
        else:
            raise ValueError(
                f"Invalid metric {metric}"
                f"Must be one of {METRICS}"
            )
        
        return score

    def _get_bad_channels(
            self, 
            score: np.ndarray, # (n_channels, ) 
            mask: np.ndarray[bool], # (n_channels, )  
            method: Literal["zscore", "threshold"], 
            threshold_value: float, 
            max_iter: int | None = 3, 
            mode: Literal["above", "below", "two-sided"] = "two-sided"
    ):
        """
        Automatically detect bad channels based on the scores of
        channel-specific metrics and a given threshold value. 
        Either using a fixed threshold  (method = "threshold") or 
        z-score normalized scores (method = "zscore").

        Args
        ----
            score : np.ndarray
                Channel-specific scores
            mask : np.ndarry
                Boolean mask of considered channels (True: not used, False: used)
            method : {"zscore", "threshold"}
                Method used for bad channel detection. If "zscore" 
                scores are z-score normalized prior to thresholding.
            threshold_value : float
                Treshold for bad channel detection
            mode : {"above", "below", "two-sided"} , default "two-sided"
                If "above" flag all values above the threshold;
                If "below" flag all values below the threshold;
                If "two-sided" (only availible if method = "zscore"), flag all 
                channels with an absolute score above the threshold                    

        Returns
        -------
            mask : np.ndarray of bool
                Boolean mask (True: bad channel, False: good channel)
        
        """

        if method == "zscore":
            mask = find_outliers(
                score, threshold_value, max_iter=max_iter, mode=mode, mask=mask
            )
        elif method == "threshold":
            if mode == "above":
                mask = score > threshold_value
            elif mode == "below":
                mask = score < threshold_value
            else:
                raise ValueError(
                    f"For method '{method}' mode must be 'above' or 'below'."
                )
        else:
            raise ValueError(
                "Invalid bad channel detection method"
                "Must be one of *zscore* or *threshold*"
            )

        return mask             

    def pre_process(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            fsamp: float = 2048,
            spikes: pd.DataFrame | None = None
    ):
        
        """
        Pre process multi-channel time EMG data using the 
        specified list of steps.

        Parameters
        ----------
        data : np.ndarray 
            Raw time series data of shape ``(n_channels, n_samples)``
        fsamp : float 
            Sampling rate in Hz
        spikes : pd.DataFrame
            Prior knowledge motor unit spike times    

        Returns
        -------
        data : np.ndarray (n_channels x n_samples)
            Pre-prcessed time series data 
        metadata : dict
            Dictonary of process metadata 
            - fsamp (float): Sampling rate in Hz
            - ch_mask (np.ndarray): Boolean channel selection mask
            - sample_mask (np.ndarray): Boolean sample selection mask
            - ch_status (pd.DataFrame): Channel status 
        steps : list
            List of the applied processing steps       
        
        """

        # Initalize the pocess metadata
        fsamp_new = fsamp
        ch_mask = np.ones(data.shape[0], dtype=bool)
        sample_mask = np.ones(data.shape[1], dtype=bool)
        ch_status = pd.DataFrame({
            "name": [f"Ch{i:03d}" for i in range(1, data.shape[0] + 1)],
            "status": ["on"] * data.shape[0],
            "status_description": ["n/a"] * data.shape[0],
        })
        t_start = 0

        # Loop over all steps
        if self.steps is not None:
            for step in self.steps:
                
                if isinstance(step, self.Bandpass):
                    data = bandpass_signals(
                        data,
                        fsamp_new,
                        high_pass=step.high_pass,
                        low_pass=step.low_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.Notch):
                    data = notch_signals(
                        data,
                        fsamp_new,
                        freqs=step.freqs,
                        method=step.method,
                        order=step.order,
                        dfreq=step.dfreq,
                    )  
                elif isinstance(step, self.Highpass):
                    data = highpass_signals(
                        data,
                        fsamp_new,
                        high_pass=step.high_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.Lowpass):
                    data = lowpass_signals(
                        data,
                        fsamp_new,
                        low_pass=step.low_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.MaskChannels):
                    local_mask = np.ones(data.shape[0], dtype=bool)
                    local_mask[step.channel_list] = False
                    ch_mask = ch_mask & local_mask
                    ch_status.loc[~local_mask, "status"] = "off"
                    ch_status.loc[
                        ~local_mask, "status_description"
                    ] = step.description
                elif isinstance(step, self.BadChannelDetection):
                    if step.window is not None:
                        idx0 = int(step.window[0] * fsamp_new)
                        idx1 = int(step.window[1] * fsamp_new)
                    else:
                        idx0 = 0
                        idx1 = data.shape[1]
                    scores = self._get_scores(data[:, idx0:idx1], step.metric)
                    bad_mask = self._get_bad_channels(
                        scores,
                        ~ch_mask,
                        method=step.method,
                        threshold_value=step.threshold_value,
                        max_iter=step.max_iter,
                        mode=step.mode
                    )
                    new_true = bad_mask & ch_mask
                    ch_mask = ch_mask & ~bad_mask
                    ch_status.loc[bad_mask, "status"] = "off"
                    ch_status.loc[
                        new_true, "status_description"
                    ] = step.description
                    ch_status[step.metric] = scores
                elif isinstance(step, self.Downsample):
                    data = data[:, ::step.factor]
                    sample_mask = sample_mask[::step.factor]
                    fsamp_new = fsamp_new / step.factor
                elif isinstance(step, self.TimeWindow):
                    n_samples = data.shape[1]
                    t = np.linspace(0, (n_samples-1) / fsamp_new, n_samples)
                    if step.t_end == -1:
                        t_end = (n_samples - 1) / fsamp
                    else:
                        t_end = step.t_end
                    sample_mask = (t >= step.t_start) & (t <= t_end)
                    t_start = step.t_start
                elif isinstance(step, self.GetMetric):
                    if step.window is not None:
                        idx0 = int(step.window[0] / fsamp_new)
                        idx1 = int(step.window[1] / fsamp_new)
                    else:
                        idx0 = 0
                        idx1 = data.shape[1]
                    scores = self._get_scores(data[:, idx0:idx1], step.metric)
                    col_name = f"{step.metric}{step.description}"
                    ch_status[col_name] = scores
                elif isinstance(step, self.PeelOff):

                    units = sorted(spikes["unit_id"].unique())

                    for unit in units:

                        mu_spikes = spikes[
                            spikes["unit_id"] == unit
                        ]["sample"].to_numpy(dtype=np.int64)
                        
                        data, _, _ = peel_off(
                            sig=data,
                            spikes=mu_spikes, 
                            win=step.window_size, 
                            fsamp=fsamp_new,
                            method="sparse"
                        )    
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
                
        # Package the applied processing steps
        steps = [step.model_dump() for step in self.steps]

        # Package process metadata in a dictonary        
        metadata = {
            "fsamp": fsamp,
            "ch_mask": ch_mask,
            "sample_mask": sample_mask,
            "t_start": t_start,
            "ch_status": ch_status,
            "steps": steps
        }

        return data, metadata

