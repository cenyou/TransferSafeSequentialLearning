"""
// Copyright (c) 2024 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Part of the code was also published in the following repository
(copyright also to Bosch under GNU Affero General Public License)
https://github.com/boschresearch/bosot/blob/main/bosot/utils/utils.py
"""

from typing import Tuple, Union, Sequence, List
import json
import os
from typing import List, Union
import gpflow
import numpy as np
from scipy import integrate
from scipy.stats import norm
import math
import tensorflow as tf
from distutils import util

def create_grid(a, b, n_per_dim, dimensions):
    grid_points = np.linspace(a, b, n_per_dim)
    n = int(np.power(n_per_dim, dimensions))
    X = np.zeros((n, dimensions))
    for i in range(0, dimensions):
        repeats_per_item = int(np.power(n_per_dim, i))
        block_size = repeats_per_item * n_per_dim
        block_repeats = int(n / block_size)
        for block in range(0, block_repeats):
            for j in range(0, n_per_dim):
                point = grid_points[j]
                for l in range(0, repeats_per_item):
                    index = block * block_size + j * repeats_per_item + l
                    X[index, i] = point
    return X


def create_grid_multi_bounds(a: Sequence[float], b: Sequence[float], n_per_dim: Sequence[float]):
    aa = np.reshape(a, -1)
    bb = np.reshape(b, -1)
    nn = np.reshape(n_per_dim, -1)
    assert len(aa) == len(bb)
    assert len(bb) == len(nn)

    n = int(np.prod(nn))
    dimensions = len(nn)
    X = np.zeros((n, dimensions))

    for i in range(dimensions):
        grids = np.linspace(aa[i], bb[i], nn[i])
        n_repeat = np.prod(nn[:i])  # np.prod([]) is 1
        n_tile = np.prod(nn[i + 1 :])

        X[:, i] = np.tile(np.repeat(grids, n_repeat), n_tile)
    return X

def filter_nan(X, y):
    r"""
    get input data pair X, y and return data pair without nan values
    input:
        X [N, D] array
        y [N, 1] array
    output:
        X [M, D] array, M <= N
        y [M, 1] array
    """
    mask = ~np.isnan(y).reshape(-1)
    return np.atleast_2d(X)[mask], np.atleast_2d(y)[mask]

def normal_entropy(sigma):
    entropy = np.log(sigma * np.sqrt(2 * np.pi * np.exp(1)))
    return entropy


def string2bool(b):
    if isinstance(b, bool):
        return b
    if b.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif b.lower() in ("no", "false", "f", "n", "0"):
        return False

def normalize_data(x: np.array):
    assert len(x.shape) == 2
    x_normalized = (x - np.expand_dims(np.mean(x, axis=0), axis=0)) / np.expand_dims(np.std(x, axis=0), axis=0)
    return x_normalized

def min_max_normalize_data(x: np.array):
    assert len(x.shape) == 2
    x_normalized = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    return x_normalized

def manhatten_distance(X: np.array, X2: np.array) -> tf.Tensor:
    differences = gpflow.utilities.ops.difference_matrix(X, X2)
    return tf.reduce_sum(tf.math.abs(differences), axis=2)

def row_wise_compare(x, y):
    r"""

    :param x: [N1, D] array
    :param y: [N2, D] array
    :return: [N1, ] boolean array, True if the row in x is contained in y
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    d1 = x.shape[1]
    d2 = y.shape[1]  # if d2 != d1, the result should be all False, so no need to waste time on checking

    struct_x = x.view(x.dtype.descr * d1)
    struct_y = y.view(y.dtype.descr * d2)

    return np.in1d(struct_x, struct_y)

def row_wise_unique(x):
    r"""
    :param x: [N1, D] array
    :return:
        output [N, D] array, N <= N1, the unique rows of x (sorted)
        idx    [N, ] array, idx, output = x[idx]
    """
    x = np.atleast_2d(x)

    d = x.shape[1]

    struct_x = x.view(x.dtype.descr * d)
    _, idx = np.unique(struct_x, return_index=True)

    return x[idx], idx

def check1Dlist(variables, dimension: int):
    r"""
    variables: a value or an array
    dimension: integer

    return a 1D list of 'dimension' num of element, could be duplicate of variables if it is a value or variables if it is of len dimension
    """
    if hasattr(variables, "__len__"):
        output = np.reshape(variables, -1).tolist()
        if len(output) == 1:  # duplicate to d elements list
            output = output * dimension
        assert len(output) == dimension
    else:
        output = [variables] * dimension
    return output
