import numpy as np
from scipy.stats import zscore
from .core import bandpass_signals, notch_signals

class pre_processing:
    """
    Class for performing convolutive blind source separation to identify the
    spiking activity of motor neurons using the fastICA algorithm.

    """

    def __init__(self, cfg=None, **kwargs):

        # Default parameters
        self.fsamp = 2048
        self.bandpass = [20, 500]
        self.bandpass_type = "butter"
        self.bandpass_order = 2
        self.bandpass_numtabs = 101
        self.notch_frequency = [50]
        self.notch_n_harmonics = 3
        self.notch_type = "butter"
        self.notch_order = 2
        self.notch_width = 1
        self.downsample_factor = None
        self.bad_channel_list = []
        self.bad_channel_metrics = ["RMS", "med_freq"]
        self.bad_channel_thresholds = [3, 3] 
        self.selected_channels = None

        # Convert config object (if provided) to a dictionary
        config_dict = vars(cfg) if cfg is not None else {}

        # Merge with directly passed keyword arguments (overwrites config)
        params = {**config_dict, **kwargs}

        valid_keys = self.__dict__.keys()

        # Assign all parameters as attributes
        for key, value in params.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

    def set_param(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter: {key}")

    def _find_outliers(x, threshold=3, max_iter=3, tail=0):
        """
        Detect ouliers by comparing the z-score of variable x against
        some threshold. This is repeaded until there are no outliers or
        the maximum number of iterations is reached. 

        Args:
            x (np.array): Variable to test for outliers
            threshold (float): Threshold for outlier detection
            max_iter (int): Maximum number of iterations
            tail {-1,0,1}: Specify weather to serach for outliers   
                on both ends (0), just on the positive (1) or just 
                the negative side (-1).
        
        """

        mask = np.zeros(len(x), dtype=bool)

        iter = 0
        while iter < max_iter:
            xm = np.ma.masked_array(x, mask=mask)
            if tail == 1:
                idx = zscore(xm) > threshold
            elif tail == -1:
                idx = -zscore(xm) > threshold
            else:
                idx = np.abs(zscore(xm)) > threshold 
            mask = mask + idx
            if not np.any(idx):
                break
            else:
                iter = iter + 1

        bad_idx = np.where(mask)[0]    

        return bad_idx        
            
    def bandpass_signals(self, data):

        if self.bandpass is not None:
            data = bandpass_signals(
                data,
                self.fsamp,
                high_pass=self.bandpass[0],
                low_pass=self.bandpass[1],
                ftype=self.bandpass_type,
                order=self.bandpass_order,
                numtabs=self.bandpass_numtabs,
            )

        return data

    def notch_signals(self, data):  

        # Notch filter signals
        if self.notch_frequency is not None:
            sig = notch_signals(
                sig,
                self.fsamp,
                nfreq=self.notch_frequency,
                dfreq=self.notch_width,
                n_harmonics=self.notch_n_harmonics,
                ftype=self.notch_type,
                order=self.notch_order,
            )    

    def downsample(self, data):

        if self.downsample_factor is not None:
            data = data[:, ::self.downsample_factor]
            fsamp_new = self.fsamp / self.downsample_factor
        else:
            fsamp_new = self.fsamp

        return data, fsamp_new  

    def detect_bad_channels(self, data):

        pass 

    def select_channels(self, data):

        if self.select_channels is not None:
            data = data[self.select_channels, :]
            channel_idx = self.select_channels
        else:
            channel_idx = np.arange(data.shape[0])

        return data, channel_idx     

    def pre_process(self, data):

        data = self.bandpass_signals(data)
        data = self.notch_signals(data)
        data, channel_idx = self.select_channels(data)
        data, fsamp_new = self.downsample(data)  

        return data, fsamp_new, channel_idx

