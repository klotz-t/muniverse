import warnings
import numpy as np
from scipy.signal import welch
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
from .core import (bandpass_signals, 
                   notch_signals, 
                   highpass_signals, 
                   lowpass_signals,
                   find_outliers
                   )


class pre_processing:
    """
    Class to preprocess HD-EMG data.

    Parameters
    ----------
    steps : list of dict
        List of preprocessing steps. Each step is a dictionary describing
        the processing operation.

        Supported step types are:
        "bandpass", "highpass", "lowpass", "notch",
        "bad_channel_detection", "mask_channels", "downsample":

        **Bandpass filter**: Bandpass filter time series data using a digital
        infinite impulse response filter ("butter") or finite impulse response
        filter ("firwin2")::

            {
                "step": "bandpass",
                "high_pass": float,
                "low_pass": float,
                "method": "butter" | "firwin2",
                "order": int,      # required if method == "butter"
                "numtabs": int,    # required if method == "firwin2"
            }

        **Highpass filter**: Highpass filter time series data using a digital
        infinite impulse response filter ("butter") or finite impulse response
        filter ("firwin2")::

            {
                "step": "highpass",
                "high_pass": float,
                "method": "butter" | "firwin2",
                "order": int, # required if method == "butter"
                "numtabs": int, # required if method == "firwin2"
            }

        **Lowpass filter**: Lowpass filter time series data using a digital
        infinite impulse response filter ("butter") or finite impulse response
        filter ("firwin2")::

            {
                "step": "lowpass",
                "low_pass": float,
                "method": "butter" | "firwin2",
                "order": int, # required if method == "butter"
                "numtabs": int, # required if method == "firwin2"
            }

        **Notch filter**:: Apply a digital notch (stop band) filter using either a
        infinite impulse response filter ("butter"), a finite impulse response 
        filter ("iirnotch") or performing filtering in the frequency domain 
        ("fft_nulling" and "fft_interpolation"). For "fft_nulling" the spectrum in
        the specified frequency band is set to zero, for "fft_interpolation" the 
        spectral amplitude is interpolated through by the neighbourhood. Time series
        data is recovered through an inverse fft.

            {
                "step": "notch",
                "freqs": list[float],
                "method": "butter" | "iirnotch" | "fft_nulling" | "fft_interpolation",
                "order": int,   # if "butter"
                "dfreq": float  # if "iirnotch", "fft_nulling" or "fft_interpolation"
            }

        **Bad Channel Detection**: Automatically detect bad channels based on some 
        metric ("std" or "rms") computed in a given time window (given in seconds). 
        If method is "zscore" the score distribution is normalized (zero mean, 
        unit standard deviation). All scores are compared to a "threshold_value". 
        If tail=1 all values above the threshold are rejected, if tail=-1 all values 
        below the theshold are rejected. For tail=0 all the absolute value of the score
        is computed and all values above the threshold are rejected (only availible 
        if "method" == "zscore")::  

            {
                "step": "bad_channel_detection",
                "metric": Literal["std", "rms"],
                "window": (t0, t1) | None, # Given in seconds, if None consider the full data
                "method": "zscore" | "threshold",
                "threshold_value": float,
                "tail": -1 | 0 | 1
            } 

        **Mask Channels**: Mask all channels given in "channel_list" to be excluded in the following. 
        Can be either used to reject known bad channels or limit the analysis to a subset of your data::  

            {
                "step": "mask_channels",
                "channel_list": list[int]
            }

        **Downsample**: Reduces the sampling frequency by the specified value::  

            {
                "step": "downsample",
                "factor": int 
            }

        **Time window**: Truncate your signal to only consider a selected time window.
        If t_end = -1 the time window ends with the last sample::  

            {
                "step": "time_window",
                "t_start": float,
                "t_end": float 
            }    

    Examples:
    ---------
    Pre process HD-EMG data using a bandpass and a notch filter.
    >>> model = pre_processing(steps = [
    >>>     {
    >>>         "step": "bandpass",
    >>>         "high_pass": 20,
    >>>         "low_pass": 500,
    >>>         "method": "butter",
    >>>         "order": 2
    >>>     },
    >>>     {
    >>>         "step": "notch",
    >>>         "freqs": [50, 100, 150],
    >>>         "method": "butter",
    >>>         "order": 2
    >>>     },
    >>> ])
    >>> preprocessed_data, metadata = model.pre_process(data=emg_data, fsamp=2048)                     

    """

    def __init__(
            self, 
            steps: list[dict] = [
                {  
                    "step": "bandpass",
                    "high_pass": 20,
                    "low_pass": 500,
                    "method": "butter",
                    "order": 2,
                },
                {
                    "step": "notch",
                    "freqs": [50, 100, 150],
                    "method": "butter",
                    "order": 2
                }    
            ]          
    ):

        #self.pre_process_steps = pre_process_steps
        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]

    class Bandpass(BaseModel):
        step: Literal["bandpass"]
        high_pass: float = 20
        low_pass: float = 500
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101

    class Highpass(BaseModel):
        step: Literal["highpass"]
        high_pass: float = 20
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101 

    class Lowpass(BaseModel):
        step: Literal["lowpass"]
        low_pass: float = 500
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101       

    class Notch(BaseModel):
        step: Literal["notch"]
        freqs: List[float] = [50, 100, 150]
        method: Literal[
            "butter", "iirnotch", "fft_nulling", "fft_interpolation"
            ] = "butter" 
        order: int = 2
        dfreq: float = 1

    class BadChannelDetection(BaseModel):
        step: Literal["bad_channel_detection"]
        metric: Literal["std", "rms"]
        window: tuple[float, float] | None = None
        method: Literal["zscore", "threshold"] = "zscore"
        threshold_value: float = 3
        max_iter: int = 3
        tail: Literal[-1, 0, 1] = 0    

    class MaskChannels(BaseModel):
        step: Literal["mask_channels"]
        channel_list: list[int] = []    

    class Downsample(BaseModel):
        step: Literal["downsample"]
        factor: int = 2

    class TimeWindow(BaseModel):
        step: Literal["time_window"]
        t_start: float = 0
        t_end: float = -1    

    PreprocessStep = Annotated[
        Union[Bandpass, 
            Lowpass,
            Highpass,  
            Notch, 
            BadChannelDetection, 
            MaskChannels, 
            Downsample,
            TimeWindow
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PreprocessStep)

    def add_step(self, step):
        """ Add an additional post processing step"""
        
        self.steps.append(
            self._adapter.validate_python(step)
        )

    def _get_scores(
            self, 
            data: np.ndarray, # (n_channels, n_samples) 
            metric: Literal["rms", "std", "medfreq", "medpower"], 
            fsamp: float | None = 2048, 
            bw: tuple | None = [20, 500]
    ):
        """
        Calculate score for bad channel detection

        
        """

        METRICS = ["rms", "std", "medfreq", "medpower"]
                    
        if metric == "rms":
            score = np.mean(data**2, axis=1)**0.5
        elif metric == "std":
            score = np.std(data, axis=1)
        elif metric == "medfreq":
            freqs, psd = welch(data, fs=self.fsamp, nperseg=self.fsamp/2)
            cumulative = np.cumsum(psd)
            total = cumulative[-1]
            idx = np.where(cumulative >= total / 2)[0][0]
            score = freqs[idx]
        elif metric == "medpower":
            freqs, psd = welch(data, fs=self.fsamp, nperseg=self.fsamp/2)
            total = cumulative[-1]
            score = np.median(psd, axis=1)
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
            tail: Literal[-1, 0, 1] = 0
    ):
        """
        Automatically detect bad channels

        TODO
        
        """

        if method == "zscore":
            mask = find_outliers(score, threshold_value, max_iter=max_iter, tail=tail)
        elif method == "threshold":
            if tail == 1:
                mask = score > threshold_value
            elif tail == -1:
                mask = score < threshold_value
            else:
                raise ValueError(
                    f"For method *{method}* tail must be -1 or 1."
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
    ):
        
        """
        Pre process multi-channel time series data using the 
        specified list of steps.

        Args
        ----
            data (np.ndarray): Raw time series data (n_channels x n_samples)
            fsamp (float): Sampling rate in Hz

        Returns
        -------
            data (np.ndarray): Pre-prcessed time series data (n_channels x n_samples)
            metadata (dict): TODO    
        
        """


        metadata = {}
        metadata["fsamp_out"] = fsamp

        # Mask bad channels
        mask = np.zeros(data.shape[0], dtype=bool)

        if self.steps is not None:
            for step in self.steps:
                
                if isinstance(step, self.Bandpass):
                    data = bandpass_signals(
                        data,
                        metadata["fsamp_out"],
                        high_pass=step.high_pass,
                        low_pass=step.low_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.Notch):
                    data = notch_signals(
                        data,
                        metadata["fsamp_out"],
                        freqs=step.freqs,
                        method=step.method,
                        order=step.order,
                        dfreq=step.dfreq,
                    )  
                elif isinstance(step, self.Highpass):
                    data = highpass_signals(
                        data,
                        metadata["fsamp_out"],
                        high_pass=step.high_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.Lowpass):
                    data = lowpass_signals(
                        data,
                        metadata["fsamp_out"],
                        low_pass=step.low_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.MaskChannels):
                    local_mask = np.zeros(data.shape[0], dtype=bool)
                    local_mask[step.channel_list] = True
                    mask += local_mask
                elif isinstance(step, self.BadChannelDetection):
                    if step.window is not None:
                        idx0 = int(step.window[0] / metadata["fsamp_out"])
                        idx1 = int(step.window[1] / metadata["fsamp_out"])
                    else:
                        idx0 = 0
                        idx1 = data.shape[1]
                    scores = self._get_scores(data[:, idx0:idx1], step.metric)
                    local_mask = self._get_bad_channels(
                        scores,
                        mask,
                        method=step.method,
                        threshold_value=step.threshold_value,
                        max_iter=step.max_iter,
                        tail=step.tail
                    )
                    mask += local_mask
                elif isinstance(step, self.Downsample):
                    data = data[:, ::step.factor]
                    metadata["fsamp_out"] = metadata["fsamp_out"] / step.factor
                elif isinstance(step, self.TimeWindow):
                    start_idx = step.t_start / metadata["fsamp_out"]
                    if step.t_end == -1:
                        end_idx = data.shape[1]
                    else:
                        end_idx = step.t_end / metadata["fsamp_out"]
                    data = data[:, start_idx:end_idx]
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
        
        selected_idx = np.where(mask == False)[0]    
            
        data = data[selected_idx, :]
        metadata["mask"] = mask

 
        return data, metadata

