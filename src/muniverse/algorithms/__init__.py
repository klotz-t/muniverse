"""
Benchmark algorithms for decomposition.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Literal

import numpy as np
from pyedflib.highlevel import read_edf

from ..utils.containers import pull_container, verify_container_engine
from .decomposition import decompose_cbss, decompose_scd, decompose_ae


def init():
    """
    Initialize the algorithms module.
    This includes verifying container engines and pulling container images if needed.
    If both Docker and Singularity are available, Singularity will be used by default.

    Returns
    -------
    engine : str 
        The selected container engine ("docker" or "singularity")

    """
    # Check availability of both engines
    docker_available = verify_container_engine("docker")
    singularity_available = verify_container_engine("singularity")

    # Select engine based on availability
    if singularity_available:
        engine = "singularity"
    elif docker_available:
        engine = "docker"
    else:
        raise RuntimeError("No container engine (Docker or Singularity) is available. Please install one first.")

    # Get container name (using default)
    container_name = "pranavm19/muniverse:scd"

    # Pull container if needed
    pull_container(container_name, engine)
    print(f"[INFO] Algorithms module initialized using {engine}.")

    return engine


def decompose_recording(
    data: Union[str, np.ndarray],
    fsamp: float, 
    method: Literal["cbss", "scd", "ae"] = "cbss",
    algorithm_config: Optional[Dict] = None,
    engine: Literal["local", "docker", "singularity"] = "local",
    container: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Highlevel API to decompose an EMG recording using the specified method.

    Parameters
    ----------
    data : {str, np.ndarray} 
        Either a path to input data file (``.npy`` or ``.edf``) 
        or a numpy array with shape ``(n_channels, n_samples)``

    fsamp : float
        Sampling rate in Hz    

    method: {"scd", "cbss", "ae"} , default "cbss"
        Decomposition method to use. Use "scd" for SwarmContrastiveDecomposition,
        "cbss" for FastIcaCBSS, or "ae" for AEDecoder.

    algorithm_config : dict 
        Dictionary containing the algorithm configuration. If None, the default 
        configs will be used

    engine : {"docker", "singularity", "local"}, default "local"
        Engine used to run the algorith,. If "local", no container is used and code is 
        evaluated locally (only required if the selected ``method``is ``"scd"``)

    container : str 
        Path to container image (only required if the selected ``method``is ``"scd"``)

    Returns
    -------
    results : dict
        Dictonary containing

            data : np.ndarray
                Pre-processed data with shape (n_channels, n_samples)
            spikes : pd.DataFrame
                Table of motor unit spikes
            sources : np.ndarray 
                Predicted sources with shape (n_sources, n_samples)
            scores : dict 
                Dictonary of source quality metrics
            pre_process_metadata : dict
                 Metadata correspoding to pre processing steps (Optional)
            post_process_metadata : dict 
                Metadata correspoding to post processing steps (Optional)

    log_data : dict
        Dictonary of processing metadata 
        
    Note
    ----
        For UpperBound decomposition, use decompose_upperbound(...) directly.

    Examples
    --------

    Run CBSS decomposition using the default parameters

    >>> from muniverse.algorithms import decompose_recording
    >>> results, log_data = decompose_recording(
    ...     data=emg_data,
    ...     fsamp=2048,
    ...     method="cbss",
    ... )

    Run SCD in a container and using the default parameters

    >>> results, log_data = decompose_recording(
    ...     data=emg_data,
    ...     fsamp=2048,
    ...     method="scd",
    ...     container="path/to/muniverse_scd.sif",
    ...     engine="singularity"
    ... ) 

    Run AE decomposition using the default parameters
    
        >>> results, log_data = decompose_recording(
        ...     data=emg_data,
        ...     fsamp=2048,
        ...     method="ae",
        ... )

    Run CBSS decomposition using the specified parameter

    1. Preprocessing by applying a band pass filter
    2. Run the CBSS decomposition with an extension factor of 12 
        and 10 ICA iterations
    3. Remove duplicate spike trains and sources with a silhouette-like
        score below 0.9

    >>> cfg = {
    ...     "preProcessingConfig": [
    ...         {
    ...             "step": "bandpass",
    ...             "high_pass": 20,
    ...             "low_pass": 500,
    ...             "method": "butter",
    ...             "order": 2
    ...         }
    ...     ],
    ...     "algorithmConfig": {
    ...         "ext_fact": 12,
    ...         "ica_iterations": 10
    ...     },
    ...     "postProcessingConfig": [
    ...         {
    ...             "step": "remove_duplicates",
    ...             "max_shift": 0.01,
    ...             "tolerance": 0.001,
    ...             "threshold": 0.3,
    ...             "quality_metric": "sil",
    ...             "mode": "max"
    ...         },
    ...         {
    ...             "step": "bad_source_detection",
    ...             "quality_metric": "sil",
    ...             "threshold": 0.9,
    ...             "min_spikes": 10,
    ...             "mode": "below"
    ...         }
    ...     ]
    ... }
    >>> results, log_data = decompose_recording(
    ...     data=emg_data,
    ...     fsamp=2048,
    ...     method="cbss",
    ...     algorithm_config=cfg
    ... )

    """

    # Check input data
    if isinstance(data, str):
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"Input data file not found: {data_path}")
        if data_path.suffix not in [".npy", ".edf", ".bdf", ".edf+", ".bdf+"]:
            raise ValueError(
                f"Unsupported file format: {data_path.suffix}. Must be .npy or .edf"
            )

        # Load data into numpy array
        if data_path.suffix in [".edf", ".bdf", ".edf+", ".bdf+"]:            
            data = read_edf(data_path)
        else:  # .npy
            data = np.load(data_path)
    
    # Validate numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be either a file path (str) or numpy array")
    if data.ndim != 2:
        raise ValueError("EMG data must be a 2D array (channels x samples)")

    # Route to appropriate method
    if method == "scd":
        if (engine in ["docker", "singularity"] and container is None):
            raise ValueError(
                "Container path must be provided to run SCD in a container."
            )
                
        # Run SCD decomposition
        return decompose_scd(
            data=data,
            fsamp=fsamp,
            algorithm_config=algorithm_config,
            engine=engine,
            container=container,
        )
    
    elif method == "cbss":

        # Call FastIcaCBSS method
        return decompose_cbss(
            data=data, 
            fsamp=fsamp,
            algorithm_config=algorithm_config
        )
    
    elif method == "ae":

        # Call AEDecomposer
        return decompose_ae(
            data=data,
            fsamp=fsamp,
            algorithm_config=algorithm_config
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. Must be one of: scd, cbss, ae"
        )
