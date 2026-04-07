import warnings
import numpy as np
from scipy.stats import zscore
from scipy.signal import welch
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
from .core import bandpass_signals, notch_signals, highpass_signals, lowpass_signals


class pre_processing:
    """
    Class to preproces HD-EMG data

    """

    def __init__(self, 
                 pre_process_steps: list = [
                    {
                        "step": "bad_channel_detection",
                        "metric": "std",
                        "method": "zscore",
                        "threshold_value": 3,
                        "tail": 0,
                    },
                    {
                        "step": "bad_channel_detection",
                        "metric": "rms",
                        "method": "threshold",
                        "threshold_value": 1e-6, 
                        "tail": -1,
                    },
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
        self.pre_process_steps = [
            self._adapter.validate_python(step)
            for step in pre_process_steps
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
        method: Literal["zscore", "threshold"] = "zscore"
        threshold_value: float = 3
        max_iter: int = 3
        tail: Literal[-1, 0, 1] = 0    

    class MaskChannels(BaseModel):
        step: Literal["mask_channels"]
        channel_list: list = []    

    class Downsample(BaseModel):
        step: Literal["downsample"]
        factor: int = 2

    PreprocessStep = Annotated[
        Union[Bandpass, 
            Lowpass,
            Highpass,  
            Notch, 
            BadChannelDetection, 
            MaskChannels, 
            Downsample
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PreprocessStep)

    def add_step(self, step):
        
        self.pre_process_steps.append(
            self._adapter.validate_python(step)
        )

    def _get_scores(self, data, metric, bw=[20, 500]):

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
    
    def _find_outliers(self, 
                       x: np.ndarray, 
                       threshold_value: float = 3, 
                       max_iter: int = 3, 
                       tail: Literal[-1,0,1] = 0
        ):
        """
        Detect ouliers by comparing the z-score of variable x against
        some threshold. This is repeaded until there are no outliers or
        the maximum number of iterations is reached. 

        Args:
            x (np.array): Variable to test for outliers
            threshold (float): Threshold for outlier detection
            max_iter (int): Maximum number of iterations
            tail (-1,0,1): Specify weather to serach for outliers   
                on both ends (0), just on the positive side (1) 
                or just the negative side (-1).

        Return:
            mask (np.array): Boolean mask (True: bad_channel, False: good_channel) 
            
        """

        mask = np.zeros(len(x), dtype=bool)

        iter = 0
        while iter < max_iter:
            xm = np.ma.masked_array(x, mask=mask)
            if tail == 1:
                idx = zscore(xm) > threshold_value
            elif tail == -1:
                idx = -zscore(xm) > threshold_value
            else:
                idx = np.abs(zscore(xm)) > threshold_value 
            mask += idx
            if not np.any(idx):
                break
            else:
                iter = iter + 1  

        return mask 

    def _get_bad_channels(self, score, mask, method, threshold_value, max_iter=3, tail=0):

        if method == "zscore":
            mask = self._find_outliers(score, threshold_value, max_iter=max_iter, tail=tail)
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

    def pre_process(self, 
                    data: np.ndarray, # (n_channels x n_samples)
                    fsamp: float = 2048,
    ) -> tuple[np.ndarray, dict]:
        
        """
        Pre process multi-channel time series data

        Args:
            data (np.ndarray): Raw time series data (n_channels x n_samples)
            fsamp (float): Sampling rate in Hz

        Returns:
            tuple:
                - data (np.ndarray): Pre-prcessed time series data (n_channels x n_samples)
                - metadata (dict): TODO    
        
        """


        metadata = {}
        metadata["fsamp_out"] = fsamp

        # Mask bad channels
        mask = np.zeros(data.shape[0], dtype=bool)

        if self.pre_process_steps is not None:
            for step in self.pre_process_steps:

                cfg = step.dict()
                
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
                    scores = self._get_scores(data, step.metric)
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
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
        
        selected_idx = np.where(mask == False)[0]    
            
        data = data[selected_idx, :]
        metadata["mask"] = mask

 
        return data, metadata

