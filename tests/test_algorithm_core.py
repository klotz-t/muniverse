"""
Test routines for the MUniverse algorithm core functionalities

"""

import numpy as np
from muniverse.algorithms.core import (
    whitening
)

def test_whitening_simple():
    """ 
    Test the whitening function for a case
    with analytical solution

    X = np.array([
        [1, -1, 0, 0], 
        [0, 0, 2, -2]
    ])

    Xw_ref = np.array([
        [np.sqrt(3/2), -np.sqrt(3/2), 0, 0], 
        [0, 0, np.sqrt(3/2), -np.sqrt(3/2)]
    ])

    Z_ref = np.array([
        [np.sqrt(3/2), 0],
        [0, np.sqrt(3/8)]
    ])
    
    
    """

    # Define some simple test data
    X = np.array([
        [1, -1, 0, 0], 
        [0, 0, 2, -2]
    ])


    methods = ["ZCA", "PCA", "Cholesky"]
    backends = ["ed", "svd"] 

    for method in methods:
        for backend in backends:
            Xw, _, _ = Xw, Z, _ = whitening(
                Y=X, 
                method=method, 
                backend=backend, 
                regularization=None, 
                eps=0
            )  

            covariance = Xw @ Xw.T / (X.shape[1] - 1)

            assert np.allclose(covariance, np.eye(2), atol=1e-6), (
                f"For method={method} and backend={backend}," 
                "the whitened covariance matrix not the idendity matrix."
            )
            assert np.allclose(Xw.mean(axis=1), 0, atol=1e-6), (
                f"For method={method} and backend={backend}," 
                "the whitening signals don't have zero mean."
            )

def test_whitening_mixture():
    """ 
    Test the whitening function for some multivariate 
    normal mixture
    
    
    """

    # Make test data
    n_samples = 1000
    rng = np.random.default_rng(42)

    cov = np.array([
        [1.0, 0.3],
        [0.3, 2.0]
    ])

    X = rng.multivariate_normal(
        mean=[0, 0],
        cov=cov,
        size=n_samples
    ).T

    X = X - X.mean(axis=1).reshape(-1,1)

    methods = ["ZCA", "PCA", "Cholesky"]
    backends = ["ed", "svd"] 

    for method in methods:
        for backend in backends:
            Xw, _, _ = Xw, Z, _ = whitening(
                Y=X, 
                method=method, 
                backend=backend, 
                regularization=None, 
                eps=1e-10
            )  

            covariance = Xw @ Xw.T / (X.shape[1] - 1)

            assert np.allclose(covariance, np.eye(2), atol=1e-6), (
                f"For method={method} and backend={backend}," 
                "the whitened covariance matrix not the idendity matrix."
            )
            assert np.allclose(Xw.mean(axis=1), 0, atol=1e-3), (
                f"For method={method} and backend={backend}," 
                "the whitening signals don't have zero mean."
            )            