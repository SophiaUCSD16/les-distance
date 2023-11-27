import numpy as np
import matplotlib.pyplot as plt
from les import les_desc_comp, les_dist_comp
from comparisons import CompareIMD, CompareIMDOurApproach, CompareTDA, CompareGS, CompareGW
import pandas as pd 
import seaborn as sns 

# Simulation parameters:
N = 1000  # Number of samples - reduced from N=3000 for faster computation times
ITER_NUM = 10  # Number of trials to average on
R1 = 10  # Major radius
R2 = 3  # Minor/middle radius in 2D/3D
R3 = 1  # Minor radius in 3D
NOISE_VAR = 1  # STD of added noise to the tori data
R_RATIOS = np.arange(0.2, 2.01, 0.2)  # Radius ratio (c parameter)
DICT_KEYS = ['t2D_2DSc', 't2D_3D', 't2D_3DSc', 't3D_2DSc', 't3D_3DSc']

def tori_2d_gen(c, NOISE_VAR=0.01):
    ang1, ang2, ang3 = 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N)
    tor2d = np.concatenate(([(R1 + c * R2 * np.cos(ang2)) * np.cos(ang1)],
                            [(R1 + c * R2 * np.cos(ang2)) * np.sin(ang1)],
                            [c * R2 * np.sin(ang2)]),
                           axis=0)
    tor2d += NOISE_VAR * np.random.randn(3, N)
    return tor2d


def tori_3d_gen(c, NOISE_VAR=0.01):
    ang1, ang2, ang3 = 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N)
    tor3d = np.concatenate(([(R1 + (R2 + c * R3 * np.cos(ang3)) * np.cos(ang2)) * np.cos(ang1)],
                            [(R1 + (R2 + c * R3 * np.cos(ang3)) * np.cos(ang2)) * np.sin(ang1)],
                            [(R2 + c * R3 * np.cos(ang3)) * np.sin(ang2)],
                            [c * R3 * np.sin(ang3)]),
                           axis=0)
    tor3d += NOISE_VAR * np.random.randn(4, N)
    return tor3d

def random_projection(data_dim, embedding_dimension, loc=0, scale=None):
    """
    Parameters
    ----------
    data : np.ndarray
        Data to embed, shape=(M, N)
    embedding_dimension : int
        Embedding dimension, dimensionality of the space to project to.
    loc : float or array_like of floats
        Mean (“centre”) of the distribution.
    scale : float or array_like of floats
        Standard deviation (spread or “width”) of the distribution.

    Returns
    -------
    np.ndarray
       Random (normal) projection of input data, shape=(dim, N)

    See Also
    --------
    np.random.normal()

    """
    if scale is None:
        scale = 1 / np.sqrt(data_dim)
    projection_matrix = np.random.normal(loc, scale, (embedding_dimension, data_dim))
    # return np.dot(projection_matrix, data)
    return projection_matrix
