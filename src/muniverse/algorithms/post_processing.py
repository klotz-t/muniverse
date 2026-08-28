import warnings
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.signal import find_peaks
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
from .cbss import _BaseCBSS
from .core import (
    est_spike_times,
    find_outliers,
    spike_dict_to_long_df,
    get_duplicates_mask, 
    get_bad_source_mask,
    spike_dict_to_long_df,
    filter_spikes
)    
from ..evaluation.evaluate import (
    pseudo_sil_score,
    calc_pnr,
    get_bin_spikes,
    match_spike_trains,
    evaluate_spike_matches
)

class PostProcessSpikes:
    """
    Class to post process motor unit spike trains.
    Availible processing steps are
    - Reject bad sources
    - Remove duplicate spike trains
    - Calculate additional spike-based metrics
    - Validate your decomposition results against 
        some ground truth / reference decomposition


    Parameters
    ----------
    steps : list of dict
        List of post processing steps. Each step is a dictionary describing
        the processing operation.

    Examples
    --------

    Post decomposition outputs by removing duplicates and 
    rejecting bad sources

    >>> from muniverse.algorithms.post_processing import PostProcessSpikes
    >>> model = PostProcessSpikes(steps = [
    ...     {
    ...         "step": "remove_duplicates",
    ...         "max_shift": 0.01,
    ...         "tolerance": 0.001,
    ...         "threshold": "0.3",
    ...         "quality_metric": "sil",
    ...         "mode": "max"
    ...     },
    ...     {
    ...         "step": "bad_source_detection",
    ...         "quality_metric": "sil",
    ...         "threshold": 0.9,
    ...         "min_spikes": 10,
    ...         "mode": "below"
    ...     },
    ... ])
    >>> out = model.post_process(
    ...     spikes=spikes,
    ...     fsamp=2048,
    ...     sources=sources,
    ...     scores=scores
    ... )                     

    """

    def __init__(
            self, 
            steps: list[dict] = []     
    ):

        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]

    class RemoveDuplicates(BaseModel):
        """
        Automatically detect duplicates in your spike trains.

        Parameters
        ----------
        max_shift : float , default 0.01
            maximum delay between two spike trains (in seconds) 
        tolerance : float , default 0.001
            Two-sided window (in seconds) in which two spikes are 
            considered idendical 
        threshold : float , default 0.3
            Fraction of spikes required to consider two spike trains 
            corresponding to the same unit 
        mode : {"max", "min", "fisrt"} , default "max" 
            Determines which source to keep. If "first", the 
            first unit is kept. If "min" or "max" the selection is based on the
            specified ``quality_metric``
        quality_metric : str , default "sil"
            Source quality metric used to select the best estimate of 
            sources that are considered idendical
        window : tuple , default (0, -1) 
            Consider only spikes in the specified intervall (in seconds).
            If -1, the largest spike time is used as the end of the intervall 
        description : str , default "Duplicate source" 
            String used during logging 

        Examples
        --------

        Reject duplicate units and always keep the one with the
        highest silhouette-like score

        >>> from muniverse.algorithms.post_processing import PostProcessSpikes
        >>> step = {
        ...     "step": "remove_duplicates",
        ...     "max_shift": 0.01, 
        ...     "tolerance": 0.001, 
        ...     "threshold": 0.3, 
        ...     "quality_metric": "sil",
        ...     "mode": "max",
        ...     "window": (0, -1),
        ...     "description": "Duplicate source"
        ... }
        >>> model = PostProcessSpikes()
        >>> model.add_step(step)
        >>> out = model.post_process(
        ...     spikes=spikes,
        ...     fsamp=2048,
        ...     sources=sources,
        ...     scores=scores
        ... ) 

        """
        step: Literal["remove_duplicates"]
        max_shift: float = 0.01
        tolerance: float = 0.001
        threshold: float = 0.3
        quality_metric: str = "sil"
        window: tuple[float, float] = (0, -1)
        mode: Literal["max", "min", "first"] = "max"
        description: str = "Duplicate source"

    class GetDischargeMetric(BaseModel):
        """
        Calculate a spike-based metric that provides insights 
        on the physiological pluasibility of a detected spike train.
        Implemented metrics are the mean firing rate ("mean_fr"), 
        the median firing rate ("med_fr"),
        the cofficient of variation of the interspike intervalls ("cov_isi") 
        and the coefficient of dispersion of the interspike intervalls ("cod_isi"). 
        Outlier spikes can be exluded to improve the robustness of
        the method. 

        Parameters
        ----------

        metric : {"mean_fr", "med_fr", "cov_isi", "cod_isi"} , default "mean_fr"
            Metric to be calculated
        window : {tuple, None} , default None
            Only consider the selected time interval (in seconds) to
            calculate the specified metric
        reject_outliers : bool , default False
            Whether to reject outliers or not
        rejection_method : {"zscore", "threshold"} , default "zscore"
            If ``zscore``, the metric is zscore normalized prior
            to outlier detection. If ``theshold``outliers are detected
            based on a fixed theshold. 
        rejection threshold : float , default 3
            Threshold value used to detect outliers
        rejection_mode : {"above", "below", "two-sided"} , default "above"
            Whether an outlier is considered for values above or below 
            the given threshold. If the rejection method is ``zscore" you 
            can also reject outliers on both ends of the distribution.

        Examples
        --------

        Calcualte the mean dicharge rate of each unit

        >>> from muniverse.algorithms.post_processing import PostProcessSpikes
        >>> step = {
        ...     "step": "get_discharge_metric",
        ...     "metric": "mean_fr",
        ... }
        >>> model = PostProcessSpikes()
        >>> model.add_step(step)
        >>> out = model.post_process(
        ...     spikes=spikes,
        ...     fsamp=2048,
        ...     sources=sources,
        ...     scores=scores
        ... ) 
        """
        step: Literal["get_discharge_metric"]
        metric: Literal["cov_isi", "cod_isi", "mean_fr", "med_fr"] = "mean_fr"
        window: tuple[float, float] | None = None
        reject_outliers: bool = False
        rejection_method: Literal["zscore", "threshold"] = "zscore"
        rejection_threshold: float = 3
        rejection_mode: Literal["above", "below", "two-sided"] = "above"    

    class BadSourceDetection(BaseModel):
        """
        Automatically detect bad sources 

        Parameters
        ----------

        quality_metric : str , default "sil"
            Quality metric used to detect bad sources
        theshold : float , default 0.9 
            Threshold value to seperate good and bad sources
        mode : {"below", "above"} , default "below"
            specifies whether to keep units above or below the
            specified theshold
        min_spikes : int , default 10
            Only accept a unit if it has a minimum number of spikes
        description : str , default "Below quality threshold" 
            String used during logging  

        Examples
        --------

        Reject all sources with a silhouette-like score below 0.89
        and less than 15 spikes

        >>> from muniverse.algorithms.post_processing import PostProcessSpikes
        >>> step = {
        ...     "step": "bad_source_detection",
        ...     "quality_metric": "sil", 
        ...     "threshold_value": 0.89,
        ...     "min_spikes": 15,
        ...     "mode": "below",
        ...     "description": "Below quality threshold"
        ... }
        >>> model = PostProcessSpikes()
        >>> model.add_step(step)
        >>> out = model.post_process(
        ...     spikes=spikes,
        ...     fsamp=2048,
        ...     sources=sources,
        ...     scores=scores
        ... ) 
        
        """
        step: Literal["bad_source_detection"]
        quality_metric: str = "sil"
        threshold: float = 0.9
        min_spikes: int = 10
        mode: Literal["above", "below"] = "below" 
        description: str = "Below quality threshold"

    class MaskSources(BaseModel):
        """
        Mask all sources given in "sources_list" 
        to be excluded in the following. Can be either used to reject 
        known bad sources or limit the analysis to a subset of your data

        Parameters
        ----------

        source_list : list of int , default []
            List of unit_ids that should be rejected
        description : str , default "Manually masked"     
            Free-text describing why the unit has been rejected 
            and which is stored in the processing metadata 

        Examples
        --------

        Reject unit 5 and 7 for further analysis

        >>> from muniverse.algorithms.post_processing import PostProcessSpikes
        >>> step = {
        ...     "step": "mask_sources",
        ...     "source_list": [5, 7],
        ...     "description": "Irregular firing unit" 
        ... }
        >>> model = PostProcessSpikes()
        >>> model.add_step(step)
        >>> out = model.post_process(
        ...     spikes=spikes,
        ...     fsamp=2048,
        ...     sources=sources,
        ...     scores=scores
        ... ) 
        
        """
        step: Literal["mask_sources"]
        unit_ids: list[int] = []  
        description: str = "Manually masked"   

    class ValidateSpikePrediction(BaseModel):
        """
        Validate the predicted spike trains given some
        reference spikes. The parameter "tol" is the maximum delay (in seconds)
        to mark two spikes idendical, "max_shift" ist the maximum global 
        delay between two spike trains (in seconds) and "threshold" is the 
        minimum fraction of common spikes that two spike trains are considered
        to be the same

        Parameters
        ----------

        t_start : float , default 0
            Only consider spikes after t_start (in seconds)
        t_end : float , default -1
            Only consider spikes before t_end (in seconds). If 
            t_end is ``-1`` the largest spike time is used 
        tol : float , default 0.001
            Two spikes are considered the same if they appear within 
            the given two-sided toleance window (in seconds)
        max_shift : float , default 0.1
            Align the spike trains prior to spike matching in 
            a two-sided window with the maximum delay specified in seconds
        threshold : float , default 0.3
            Two spike trains are considered the same if they share
            the specified fraction of common spikes

        Examples
        --------

        Validate you decomposition based on existing ground truth
        spikes stored in ``reference_spikes``

        >>> from muniverse.algorithms.post_processing import PostProcessSpikes
        >>> model = PostProcessSpikes()
        >>> step = {
        ...     "step": "validate_prediction",
        ...     "t_start": 5.0, 
        ...     "t_end": 25.0, 
        ...     "tol": 0.001,
        ...     "max_shift": 0.1, 
        ...     "threshold": 0.3
        ... }
        >>> model.add_step(step)
        >>> out = model.post_process(
        ...     spikes=spikes,
        ...     fsamp=2048,
        ...     sources=sources,
        ...     scores=scores,
        ...     ground_truth=reference_spikes
        ... ) 
        
        """
        step: Literal["validate_prediction"]
        t_start: float = 0
        t_end: float = -1
        tol: float = 0.001
        max_shift: float = 0.1
        threshold: float = 0.3

                          
    PostProcessStep = Annotated[
        Union[
            RemoveDuplicates, 
            GetDischargeMetric,
            BadSourceDetection,
            MaskSources,
            ValidateSpikePrediction
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PostProcessStep)

    def add_step(self, step):
        """ 
        Add an additional post processing step
        
        Parameters
        ----------
        step : dict
            Dictonary of the processing step that is
            added to your processing pipeline
        
        """
        
        self.steps.append(
            self._adapter.validate_python(step)
        )

    def _get_discharge_metric(
            self, 
            spikes: pd.DataFrame, 
            metric: Literal["cov_isi", "cod_isi", "mean_fr", "med_fr"], 
            window: tuple[float, float] | None,
            reject_outliers: bool,  
            rejection_method: Literal["zscore", "threshold"], 
            rejection_theshold: float, 
            rejection_mode: Literal["above", "below", "two-sided"] = "above"
    ):
        """
        Compute spike-based metrics

        Parameters
        ----------
        spikes : pd.DataFrame
            Table of motor unit spikes    
        metric: {"cov_isi", "cod_isi", "mean_fr", "med_fr"}
            Metric to be computed  
        window : tuple or None
            Time window used to calculate the given metric     
        reject_outliers : bool
            If True, outlier spikes are removed to calculate 
            the given metric     
        rejection_method : {"zscore", "threshold"}
            Method used to detect outlier spikes. If "zscore" 
            scores are z-score normalized prior to thresholding.
        rejection_threshold : float
            Treshold for bad spike detection
        mode : {"above", "below", "two-sided"} , default "above"
            If "above" flag all values above the threshold;
            If "below" flag all values below the threshold;
            If "two-sided" (only availible if method = "zscore"), flag all 
            channels with an absolute score above the threshold                    

        Returns
        -------
        values : np.array
            Array of spike-based metrics (n_units, )  
        
        """

        # Get unique units
        units = sorted(spikes["unit_id"].unique())
        n_source = len(units)

        # Init output
        values = np.zeros(n_source)

        for i, label in enumerate(units):
            # Get spike times
            spike_times = spikes[spikes["unit_id"] == label]["onset"].values 

            # Apply window
            if window is not None:
                if window[1] == -1:
                    window[1] = np.max(spike_times) + 0.1
                spike_times = spike_times[
                    (spike_times >= window[0]) & (spike_times < window[1])
                ]

            # Get ISI
            isi = np.diff(spike_times)

            # Reject outlier spikes for the metric calculation
            if reject_outliers:
                if rejection_method == "zscore":
                    mask = find_outliers(
                        isi, rejection_theshold, max_iter=1, mode=rejection_mode
                    )
                elif rejection_method == "threshold": 
                    if rejection_mode =="above":
                        mask = isi > rejection_theshold
                    elif rejection_mode == "below":
                        mask = isi < rejection_theshold
                    else:
                        raise ValueError(
                            f"For rejection_method '{rejection_method}'" 
                             "the rejection mode must be 'above' or 'below'."
                        )
                    
                isi = isi[~mask]

            if metric == "cov_isi":
                values[i] = np.std(isi) / np.mean(isi)
            elif metric == "cod_isi":
                values[i] = median_abs_deviation(isi) / np.median(isi)
            elif metric == "mean_fr":
                values[i] = np.mean(1 / isi)
            elif metric == "med_fr":
                values[i] = np.median(1 / isi)

        return values    

    def _apply_base_step(
        self,
        step,
        spikes,
        sources,
        scores,
        fsamp,
        unit_status,
        source_mask,
        ground_truth
    ):

        if isinstance(step, self.RemoveDuplicates):

            if step.mode in ["max", "min"]:
                if not step.quality_metric in scores.keys():
                    raise ValueError(
                        f"The slected qaulity metric {step.quality_metric} is not defined"
                    )
                local_scores = scores[step.quality_metric]
            else:
                local_scores = np.ones(len(unit_status))    

            local_mask, new_labels = get_duplicates_mask(
                spikes=spikes,
                scores=local_scores,
                fsamp=fsamp,
                mode=step.mode,
                t_start=step.window[0],
                t_end=step.window[1],
                duplicate_theshold=step.threshold,
                max_shift=step.max_shift,
                tol=step.tolerance,
                mask=source_mask
            )
            source_mask = source_mask & local_mask
            unit_status.loc[~local_mask, "status"] = "masked"
            unit_status.loc[
                ~local_mask, "status_description"
            ] = step.description
            unit_status["duplicate_unit_id"] = new_labels.astype(int)

        elif isinstance(step, self.BadSourceDetection):

            if not step.quality_metric in scores.keys():
                raise ValueError(
                    f"The slected qaulity metric {step.quality_metric} is not defined"
                )

            local_mask = get_bad_source_mask(
                spikes=spikes,
                score=scores[step.quality_metric],
                threshold=step.threshold,
                mode=step.mode,
                min_num_spikes=step.min_spikes
            )
            source_mask = source_mask & local_mask
            unit_status.loc[~local_mask, "status"] = "masked"
            unit_status.loc[
                ~local_mask, "status_description"
            ] = step.description

        elif isinstance(step, self.GetDischargeMetric):

            values = self._get_discharge_metric(
                spikes=spikes,
                metric=step.metric,
                window=step.window,
                reject_outliers=step.reject_outliers,
                rejection_method=step.rejection_method,
                rejection_theshold=step.rejection_threshold,
                rejection_mode=step.rejection_mode
            )
            scores[step.metric] = values
            unit_status[step.metric] = values

        elif isinstance(step, self.MaskSources):

            n_source = len(spikes["unit_id"].unique())
            local_mask = np.ones(n_source, dtype=bool)
            local_mask[step.unit_ids] = False
            source_mask = source_mask & local_mask
            unit_status.loc[~local_mask, "status"] = "masked"
            unit_status.loc[
                ~local_mask, "status_description"
            ] = step.description

        elif isinstance(step, self.ValidateSpikePrediction):

            df = evaluate_spike_matches(
                df1=spikes,
                df2=ground_truth,
                fsamp=fsamp,
                t_start=step.t_start,
                t_end=step.t_end,
                tol=step.tol,
                max_shift=step.max_shift,
                threshold=step.threshold,
                mask=source_mask
            )

            unit_status = pd.merge(unit_status, df, on="unit_id", how="left")    

        return source_mask, unit_status
           
    def post_process(
            self, 
            spikes: pd.DataFrame, 
            fsamp: float,
            scores: dict | None = None,
            sources: np.ndarray | None = None,
            ground_truth: pd.DataFrame | None = None
    ):
        
        """
        Post process decomposed motor unit spike trains
        specified list of steps.

        Parameters
        ----
        data : np.ndarray 
            EMG data with shape ``(n_channels, n_samples)``
        spikes : pd.DataFrame
            Table of motor unit spikes    
        fsamp : float 
            Sampling rate in Hz
        scores : dict | None 
            A Dictonary of source quality scores     
        sources : np.ndarray | None 
            The predicted sources with shape ``(n_sources, n_samples)``

        Returns
        -------
        spikes : pd.DataFrame
            Table of motor unit spikes
        sources : np.ndarray (n_units, n_samples)
            The predicted sources / latents
        score : dict
            A dictonary of source quality scores    
        metadata : dict
            A dictonary of processing metadata     
        
        """

        # Handle empty spikes file
        if spikes.empty:
            warnings.warn(
                "Empty spikes file is empty"
                "Cannot continue savely"
            )
            
            return spikes, sources, scores, {"steps": []}

        # Mask bad sources
        unit_ids = sorted(spikes["unit_id"].unique())
        n_units = len(unit_ids)
        source_mask = np.ones(n_units, dtype=bool)

        unit_status = pd.DataFrame({
            "unit_id": unit_ids,
            "status": ["good"] * n_units,
            "status_description": ["n/a"] * n_units,
            "duplicate_unit_id": ["n/a"] * n_units
        })

        if scores is None:
            scores = {}
        else:
            for k, v in scores.items():
                unit_status[k] = v
               
        if self.steps is not None:
            for step in self.steps:

                source_mask, unit_status = self._apply_base_step(
                    step = step,
                    spikes = spikes,
                    sources = sources,
                    scores = scores,
                    fsamp = fsamp,
                    unit_status= unit_status,
                    source_mask = source_mask,
                    ground_truth = ground_truth
                )

        # Filter outputs and only keep valid sources  
        new_spikes, label_map = filter_spikes(spikes, source_mask)
        unit_status["output_unit_id"] = unit_status["unit_id"].map(label_map)

        if sources is None:
            new_sources = None
        else:
            new_sources = sources[source_mask, :]

        if scores is None:
            new_scores = scores
        else:    
            new_scores = {}
            for k, v in scores.items():
                if isinstance(v, np.ndarray) and v.shape[0] == len(source_mask):
                    new_scores[k] = v[source_mask]
                else:
                    new_scores[k] = v

        # Package the applied processing steps
        steps = [step.model_dump() for step in self.steps] 

        metadata = {
            "fsamp": fsamp,
            "source_mask": source_mask,
            "unit_status": unit_status,
            "steps": steps,
            "label_map": label_map
        }            
 
        return new_spikes, new_sources, new_scores, metadata
    

class PostProcessCBSS(_BaseCBSS, PostProcessSpikes):
    """
    
    Class to post process motor unit spike trains
    within a CBSS model framework. Availible processing 
    steps are
    
    - Reject bad sources
    - Remove duplicate spike trains
    - Calculate additional spike-based metrics
    - Validate your decomposition results against 
        some ground truth / reference decomposition 
    - Apply learned unmixing weights to new segments of
        multi-channel EMG data
    - Suppervised learning of the unmixing weighs given
        motor neuron discharge times
    
    Parameters
    ----------

    steps : list of dict
        List of post processing steps. Each step is a dictionary describing
        the processing operation.

    ext_fact : int , default 12
        Extension factor

    whitening_method : {"ZCA", "PCA", "Cholesky"}, default "ZCA" 
        Method used for whitening

    whitening_regularization : {"auto", float, None}, default "auto" 
        Adds a small value to the eigenvalues for regularization. 
        If "auto", the mean of the second half of the eigenvalues is used.

    whitening_backend : {"ed", "svd"}, default "ed" 
        Method used to calculate eigenvalues and eigenvectors. Can be
        either based on singular value decomposition ("svd") or an
        eigendecomposition ("ed"). Only needed if method is "ZCA" or "PCA"    

    spike_detection_exp : float , default 2
        Exponent of asymetric power law applied to the extracted sources
        before spike detection

    spike_detection_min_delay : float , default 0.01
        Minimum distance between two detected spikes in seconds  

    verbose : bool , default True
        Verbose mode   

    Examples
    --------

    Post process decomposition predictions by 
    
    1.) Fit a CBSS model to the given spikes

    2.) Reject bad sources below a silhouette score threshold

    >>> from muniverse.algorithms.post_processing import PostProcessSpikes
    >>> model = PostProcessSpikes(steps = [
    ...     {
    ...         "step": "fit_from_spikes",
    ...         "max_delay": 0.01, 
    ...     },
    ...     {
    ...         "step": "bad_source_detection",
    ...         "quality_metric": "sil",
    ...         "threshold": 0.9,
    ...         "min_spikes": 10,
    ...         "mode": "below"
    ...     },
    ... ])
    >>> out = model.post_process(
    ...     data=emg_data
    ...     spikes=spikes,
    ...     fsamp=2048
    ... )        

    
    """

    def __init__(
            self, 
            steps = [],
            ext_fact = 12, 
            whitening_method = "ZCA", 
            whitening_backend: Literal["ed", "svd"] = "ed",
            whitening_reg = "auto", 
            spike_detection_exp = 2, 
            spike_detection_min_delay = 0.01, 
            verbose = False
        ):
        super().__init__(
            ext_fact = ext_fact, 
            whitening_method = whitening_method, 
            whitening_backend = whitening_backend,
            whitening_reg = whitening_reg, 
            spike_detection_exp = spike_detection_exp, 
            spike_detection_min_delay = spike_detection_min_delay, 
            verbose = verbose
        )

        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]

    class PredictSpikes(BaseModel):
        """
        Predict motor unit spike trains given multi-channel 
        EMG data using the given unmixing weights.

        Parameters
        ----------

        rewhiten : bool , default True
            If ``True``, the data is rewhitened based on the 
            given data
        t_start : float , default 0
            Start time (in seconds) of the considered time window
        t_end : float , default -1
            End time (in seconds) of the considered time window.
            If t_end = -1, the time of the last spike is used. 

        Examples
        --------

        Use the unmixing weights learned during a steady isometric
        contraction and apply the unmixing weights also to the ramp 
        segments of a ramp and hold contarction 

        >>> from muniverse.algorithms.post_processing import PostProcessCBSS
        >>> steps = [{
        ...     "step": "predict_spikes",
        ...     "rewhiten": false,
        ...     "t_start": 2.5,
        ...     "t_end": 37.5 
        ... }]
        >>> model = PostProcessCBSS(
        ...     steps=steps,
        ...     ext_fact=16
        ... )
        >>> out = model.post_process(
        ...     data=emg_data,
        ...     spikes=spikes,
        ...     fsamp=2048,
        ...     unmixing_weights=my_weights,
        ...     unmixing_format="white",
        ...     whitening_matrix=my_whitening_matrix
        ... )

        
        """
        step: Literal["predict_spikes"]
        rewhiten: bool = True
        t_start: float = 0
        t_end: float = -1 

    class FitFromSpikes(BaseModel):
        """
        Supervised learning of the unmixing weights of a 
        CBSS model given a set of motor unit spike labels. 
        The learned unmixing weights are then applied to the data.

        Parameters
        ----------

        t_start : float , default 0
            Start time (in seconds) of the considered time window
        t_end : float , default -1
            End time (in seconds) of the considered time window.
            If t_end = -1, the time of the last spike is used. 
        max_delay : float , default 0.01
            Considers all delayed versions of the spike train given
            the specified range and return the unmixing weights yielding
            the highest silhouette-like score.

        Examples
        --------

        Fit a CBSS model given multi-channel EMG data and motor unit
        spike times 

        >>> from muniverse.algorithms.post_processing import PostProcessCBSS
        >>> steps = [{
        ...     "step": "fit_from_spikes",
        ...     "max_delay": 0.01, 
        ... }]
        >>> model = PostProcessCBSS(
        ...     steps=steps,
        ...     ext_fact=16
        ... )
        >>> out = model.post_process(
        ...     data=emg_data,
        ...     spikes=spikes,
        ...     fsamp=2048
        ... ) 

        """   
        step: Literal["fit_from_spikes"]
        rewhiten: bool = True
        t_start: float = 0
        t_end: float = -1
        max_delay: float = 0.01    

    PostProcessStep = Annotated[
        Union[
            PostProcessSpikes.RemoveDuplicates, 
            PostProcessSpikes.BadSourceDetection,
            PostProcessSpikes.GetDischargeMetric,
            PostProcessSpikes.MaskSources,
            PostProcessSpikes.ValidateSpikePrediction,
            PredictSpikes,
            FitFromSpikes,
        ],
        Field(discriminator="step")
    ]

    _adapter = TypeAdapter(PostProcessStep)
    
    def _get_win_samples(
            self, 
            data: np.ndarray, 
            fsamp: float,
            t_start: float,
            t_end: float
        ):
        """Extract time window samples"""

        duration = (data.shape[1] - 1) / fsamp
        if t_end > duration or t_end == -1:
            t_end = duration
        if t_start < 0:
            t_start = 0    
        t = np.linspace(0, duration, data.shape[1])
        sample_idx = (t >= t_start) & (t <= t_end)

        return sample_idx     

    def rewhiten(self, data):
        """Recalculate the whiening matrix"""

        ext_sig = self._extension(data)
        self._whitening(ext_sig, return_data=False)
                
    def fit_predict_from_spike_labels(
            self, 
            sig: np.ndarray, 
            spikes: pd.DataFrame, 
            fsamp: float, 
            max_delay: float = 0.01,
            rewhiten: bool = True,
            mask: np.ndarray | None = None
        ):
        """
        Supervised fitting of a CBSS model given EMG
        data and motor unit spike labels

        Parameters
        ----
        sig : np.ndarray (n_channels, n_samples)
            EMG data matrix

        spikes : pd.DataFrame
            Table of motor unit spike labels

        fsamp : float
            Sampling rate in Hz

        max_delay: float, default 0.01
            Maximum delay in seconds that is considered for
            finding the unmixing weights.   

        mask : np.ndarray of bool | None , default None
            Boolean mask describing the unit stattus. If False,
            the unit is neglected.     

        Returns
        -------
        spikes : pd.DataFrame 
            Table of motor unit spikes (can be temporally shifted)

        sources : np.ndarray 
            Estimated sources / ica components (n_components, n_samples)     

        scores : dict
            Dictonary of source quality scores            
        
        """

        ext_sig = self._extension(sig)

        if rewhiten or (self.whiten_ is None):
            white_sig = self._whitening(ext_sig)
        else:
            white_sig = self.whiten_ @ ext_sig

        units = sorted(spikes["unit_id"].unique())
        n_units = len(units)

        new_spikes = {i: [] for i in range(n_units)}
        scores = {
            "sil": np.zeros(n_units) * np.nan,
            "cov_isi": np.zeros(n_units) * np.nan,
            "pnr": np.zeros(n_units) * np.nan
        }
        self.unmixing_weights_ = np.zeros((white_sig.shape[0], n_units))

        sources = np.zeros((n_units, white_sig.shape[1]))

        if mask is None:
            mask = np.ones(n_units, dtype=bool)

        for i in range(n_units):

            local_spikes = spikes[
                spikes["unit_id"] == units[i]]["sample"].values
            
            if ~mask[i]:
                new_spikes[i] = local_spikes
                continue

            w, new_spikes[i], sil = self._optimze_delay(
                X = white_sig,
                spikes=local_spikes,
                fsamp=fsamp,
                max_delay=max_delay
            )

            sources[i, :] = w @ white_sig
            self.unmixing_weights_[:, i] = w
            scores["sil"][i] = sil
            scores["cov_isi"][i] = self._calc_cov_isi(new_spikes[i], fsamp)
            scores["pnr"][i], _ = calc_pnr(sources[i], new_spikes[i])

        # Convert dict of spikes to long-formated spike table 
        new_spikes = spike_dict_to_long_df(new_spikes, fsamp=fsamp)    

        # Update the unmixing format
        self.unmixing_format_ = "white"

        return new_spikes, sources, scores

    def _optimze_delay(self, X, spikes, fsamp, max_delay):
        """
        Helper function to find optimal delay for 
        a set of motor unit spike labels (single unit)
        
        """

        if max_delay < (1 / fsamp):
            delays = [0]
        else:
            max_shift = int(max_delay * fsamp)
            delays = range(-max_shift,max_shift+1)

        W = np.zeros((X.shape[0], len(delays)))
        local_scores = np.zeros(len(delays))   

        for j in delays:

            local_spikes = spikes + delays[j]

            w = np.mean(X[:, local_spikes], axis=1)
            w = w / np.linalg.norm(w)
            local_source = w.T @ X
            local_scores[j], _ = pseudo_sil_score(
                source=local_source,
                spikes=local_spikes,
                fsamp=fsamp,
                min_peak_dist=self.spike_detection_min_delay,
                match_dist=0.001
            )
            W[:, j] = w 

        idx = np.argmax(local_scores)
        w = W[:, idx]
        sil = local_scores[idx]
        new_spikes = spikes + delays[idx]
 
        return w, new_spikes, sil  
    
    def _rand_permutation(self, X, spikes, fsamp, iter=100, seed=42):


        st1 = get_bin_spikes(spikes, X.shape[1])

        n_spikes = len(spikes)

        new_spikes =  {i: [] for i in range(iter)}   
        tp = np.zeros(iter)
        fp = np.zeros(iter)
        fn = np.zeros(iter)

        rng = np.random.default_rng(seed)

        for i in range(iter):

            tmp = rng.choice(spikes, size=int(n_spikes*0.8), replace=False)
            w = np.mean(X[:, tmp], axis=1)
            w = w / np.linalg.norm(w)
            local_source = w.T @ X

            new_spikes, _ = est_spike_times(
                source = local_source, 
                fsamp = fsamp, 
                a = 2,
                min_delay = self.spike_detection_min_delay
            )

            st2 = get_bin_spikes(new_spikes, X.shape[1])
            tp[i], fp[i], fn[i] = match_spike_trains(st1, st2, shift=0, tol=0.001, fsamp=fsamp)

        f1 = 2 * tp / (2 * tp + fp + fn)

        return f1.mean(), f1.std()

        
    def post_process(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            spikes: pd.DataFrame, 
            fsamp: float,
            scores: dict | None = None,
            sources: np.ndarray | None = None, # (n_sources x n_samples)
            unmixing_weights: np.ndarray | None = None,
            whitening_matrix: np.ndarray | None = None, 
            unmixing_format: Literal["white", "extended"] = "white",
            ground_truth: pd.DataFrame | None = None
    ):
        
        """
        Post process decomposed motor unit spike trains
        using the specified list of steps.

        Parameters
        ----
        data : np.ndarray 
            EMG data of shape ``(n_channels, n_samples)``

        spikes : pd.DataFrame
            Lits of motor unit spikes  

        fsamp : float 
            Sampling rate in Hz

        scores : dict | None , default None
            Dictonary of source quality scores     

        sources : np.ndarray | None , default None
            The predicted sources with shape ``(n_sources, n_samples)``

        unmixing_weights: np.ndarray or None , default None
            Weights of the unmixing matrix 

        whitening_matrix : np.ndarray or None , default None
            Whitening matrix 

        unmixing_format : {"white", "extended"} , default "white"    
            Format in which the unmixing weights are provided

        ground_truth : pd.DataFrame | None , default None
            Optionally parse a dictonary of spike times to
            validate your predictions.


        Returns
        -------
        spikes : pd.DataFrame
            Table of motor unit spikes

        sources : np.ndarray 
            The predicted sources of shape ``(n_units, n_samples)``

        score : dict
            A dictonary of source quality scores   

        metadata : dict
            A dictonary of processing metadata     
           
        
        """

        # Handle empty spikes file
        if spikes.empty:
            warnings.warn(
                "Empty spikes file is empty"
                "Cannot continue savely"
            )
            return spikes, sources, scores, {"steps": []}

        self.unmixing_weights_ = unmixing_weights
        self.whiten_ = whitening_matrix
        self.unmixing_format_ = unmixing_format

        # Mask bad sources
        unit_ids = sorted(spikes["unit_id"].unique())
        n_units = len(unit_ids)
        source_mask = np.ones(n_units, dtype=bool)

        unit_status = pd.DataFrame({
            "unit_id": unit_ids,
            "status": ["good"] * n_units,
            "status_description": ["n/a"] * n_units,
            "duplicate_unit_id": ["n/a"] * n_units
        })

        if scores is None:
            scores = {}
        else:
            for k, v in scores.items():
                unit_status[k] = v
         
        if self.steps is not None:
            for step in self.steps:
                
                source_mask, unit_status = self._apply_base_step(
                    step = step,
                    spikes = spikes,
                    sources = sources,
                    scores = scores,
                    fsamp = fsamp,
                    unit_status = unit_status, 
                    source_mask = source_mask,
                    ground_truth = ground_truth
                )

                if isinstance(step, self.PredictSpikes):

                    sample_idx = self._get_win_samples(
                        data, fsamp, step.t_start, step.t_end
                    )
                    if step.rewhiten:
                        self.rewhiten(data[:, sample_idx])  

                    if unmixing_format == "extended":
                        self.unmixing_weights_ = self.whiten_ @ self.unmixing_weights_
                        self.unmixing_format_ = "white"

                    spikes, sources, local_scores = self.predict(
                        sig=data[:, sample_idx],
                        fsamp=fsamp
                    )
                    scores.update(local_scores)
                    for k, v in local_scores.items():
                        unit_status[k] = v

                    
                elif isinstance(step, self.FitFromSpikes):

                    duration = (data.shape[1] - 1) / fsamp
                    if step.t_end > duration or step.t_end == -1:
                        t_end = duration

                    filtered_spikes = spikes[
                        (spikes["onset"] > step.t_start) &
                        (spikes["onset"] < t_end)
                    ]

                    spikes, sources, local_scores = self.fit_predict_from_spike_labels(
                        sig=data,
                        fsamp=fsamp,
                        spikes=filtered_spikes,
                        max_delay=step.max_delay,
                        rewhiten=step.rewhiten,
                        mask=source_mask
                    )
                    scores.update(local_scores)
                    for k, v in local_scores.items():
                        unit_status[k] = v    

        # Filter outputs and only keep valid sources  
        new_spikes, label_map = filter_spikes(spikes, source_mask)
        unit_status["output_unit_id"] = unit_status["unit_id"].map(label_map)

        if sources is None:
            new_sources = None
        else:
            new_sources = sources[source_mask, :]

        if scores is None:
            new_scores = scores
        else:    
            new_scores = {}
            for k, v in scores.items():
                if isinstance(v, np.ndarray) and v.shape[0] == len(source_mask):
                    new_scores[k] = v[source_mask]
                else:
                    new_scores[k] = v

        # Package the applied processing steps
        steps = [step.model_dump() for step in self.steps] 

        metadata = {
            "fsamp": fsamp,
            "source_mask": source_mask,
            "unit_status": unit_status,
            "steps": steps,
            "label_map": label_map
        }              
 
        return new_spikes, new_sources, new_scores, metadata

        
    



