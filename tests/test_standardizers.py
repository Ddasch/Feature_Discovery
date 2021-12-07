import cupy as cp
import numpy as np
import pandas as pd


from featurediscovery.standardizers.standardizers import *

def test_mean_centralizer():

    std = Mean_Centralizer()

    x = cp.array([
        [1, 2],
        [1, 2],
        [1, 2],
        [3, 6],
        [3, 6],
        [3, 6],
    ])

    std.fit(x)

    x_cen = std.transform(x)

    x_expected = cp.array([
        [-1, -2],
        [-1, -2],
        [-1, -2],
        [1, 2],
        [1, 2],
        [1, 2],
    ])

    np.testing.assert_array_equal(x_cen.get(),x_expected.get())



def test_standardizer():

    std = Stand_Scaler()

    x = cp.array([
        [1, 2],
        [1, 2],
        [1, 2],
        [3, 6],
        [3, 6],
        [3, 6],
    ])

    std.fit(x)

    x_cen = std.transform(x)

    x_expected = cp.array([
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [1, 1],
        [1, 1],
        [1, 1],
    ])

    np.testing.assert_array_equal(x_cen.get(),x_expected.get())



