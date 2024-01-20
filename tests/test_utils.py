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
from tssl.utils.utils import normalize_data, min_max_normalize_data
from tssl.utils.utils import manhatten_distance
from tssl.utils.utils import row_wise_compare, row_wise_unique
from tssl.utils.utils import check1Dlist
from tssl.utils.utils import filter_nan
from tssl.utils.utils import create_grid, create_grid_multi_bounds
import pytest
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def test_create_grid_multi_bounds():
    grid = create_grid_multi_bounds([-2, -4, 0, 2], [0, -2, 2, 4], [5, 3, 5, 3])
    result = np.array(
        [
            [-2.0, -4, 0, 2],
            [-1.5, -4, 0, 2],
            [-1.0, -4, 0, 2],
            [-0.5, -4, 0, 2],
            [0.0, -4, 0, 2],
            [-2.0, -3, 0, 2],
            [-1.5, -3, 0, 2],
            [-1.0, -3, 0, 2],
            [-0.5, -3, 0, 2],
            [0.0, -3, 0, 2],
            [-2.0, -2, 0, 2],
            [-1.5, -2, 0, 2],
            [-1.0, -2, 0, 2],
            [-0.5, -2, 0, 2],
            [0.0, -2, 0, 2],
            [-2.0, -4, 0.5, 2],
            [-1.5, -4, 0.5, 2],
            [-1.0, -4, 0.5, 2],
            [-0.5, -4, 0.5, 2],
            [0.0, -4, 0.5, 2],
            [-2.0, -3, 0.5, 2],
            [-1.5, -3, 0.5, 2],
            [-1.0, -3, 0.5, 2],
            [-0.5, -3, 0.5, 2],
            [0.0, -3, 0.5, 2],
            [-2.0, -2, 0.5, 2],
            [-1.5, -2, 0.5, 2],
            [-1.0, -2, 0.5, 2],
            [-0.5, -2, 0.5, 2],
            [0.0, -2, 0.5, 2],
            [-2.0, -4, 1, 2],
            [-1.5, -4, 1, 2],
            [-1.0, -4, 1, 2],
            [-0.5, -4, 1, 2],
            [0.0, -4, 1, 2],
            [-2.0, -3, 1, 2],
            [-1.5, -3, 1, 2],
            [-1.0, -3, 1, 2],
            [-0.5, -3, 1, 2],
            [0.0, -3, 1, 2],
            [-2.0, -2, 1, 2],
            [-1.5, -2, 1, 2],
            [-1.0, -2, 1, 2],
            [-0.5, -2, 1, 2],
            [0.0, -2, 1, 2],
            [-2.0, -4, 1.5, 2],
            [-1.5, -4, 1.5, 2],
            [-1.0, -4, 1.5, 2],
            [-0.5, -4, 1.5, 2],
            [0.0, -4, 1.5, 2],
            [-2.0, -3, 1.5, 2],
            [-1.5, -3, 1.5, 2],
            [-1.0, -3, 1.5, 2],
            [-0.5, -3, 1.5, 2],
            [0.0, -3, 1.5, 2],
            [-2.0, -2, 1.5, 2],
            [-1.5, -2, 1.5, 2],
            [-1.0, -2, 1.5, 2],
            [-0.5, -2, 1.5, 2],
            [0.0, -2, 1.5, 2],
            [-2.0, -4, 2, 2],
            [-1.5, -4, 2, 2],
            [-1.0, -4, 2, 2],
            [-0.5, -4, 2, 2],
            [0.0, -4, 2, 2],
            [-2.0, -3, 2, 2],
            [-1.5, -3, 2, 2],
            [-1.0, -3, 2, 2],
            [-0.5, -3, 2, 2],
            [0.0, -3, 2, 2],
            [-2.0, -2, 2, 2],
            [-1.5, -2, 2, 2],
            [-1.0, -2, 2, 2],
            [-0.5, -2, 2, 2],
            [0.0, -2, 2, 2],
            [-2.0, -4, 0, 3],
            [-1.5, -4, 0, 3],
            [-1.0, -4, 0, 3],
            [-0.5, -4, 0, 3],
            [0.0, -4, 0, 3],
            [-2.0, -3, 0, 3],
            [-1.5, -3, 0, 3],
            [-1.0, -3, 0, 3],
            [-0.5, -3, 0, 3],
            [0.0, -3, 0, 3],
            [-2.0, -2, 0, 3],
            [-1.5, -2, 0, 3],
            [-1.0, -2, 0, 3],
            [-0.5, -2, 0, 3],
            [0.0, -2, 0, 3],
            [-2.0, -4, 0.5, 3],
            [-1.5, -4, 0.5, 3],
            [-1.0, -4, 0.5, 3],
            [-0.5, -4, 0.5, 3],
            [0.0, -4, 0.5, 3],
            [-2.0, -3, 0.5, 3],
            [-1.5, -3, 0.5, 3],
            [-1.0, -3, 0.5, 3],
            [-0.5, -3, 0.5, 3],
            [0.0, -3, 0.5, 3],
            [-2.0, -2, 0.5, 3],
            [-1.5, -2, 0.5, 3],
            [-1.0, -2, 0.5, 3],
            [-0.5, -2, 0.5, 3],
            [0.0, -2, 0.5, 3],
            [-2.0, -4, 1, 3],
            [-1.5, -4, 1, 3],
            [-1.0, -4, 1, 3],
            [-0.5, -4, 1, 3],
            [0.0, -4, 1, 3],
            [-2.0, -3, 1, 3],
            [-1.5, -3, 1, 3],
            [-1.0, -3, 1, 3],
            [-0.5, -3, 1, 3],
            [0.0, -3, 1, 3],
            [-2.0, -2, 1, 3],
            [-1.5, -2, 1, 3],
            [-1.0, -2, 1, 3],
            [-0.5, -2, 1, 3],
            [0.0, -2, 1, 3],
            [-2.0, -4, 1.5, 3],
            [-1.5, -4, 1.5, 3],
            [-1.0, -4, 1.5, 3],
            [-0.5, -4, 1.5, 3],
            [0.0, -4, 1.5, 3],
            [-2.0, -3, 1.5, 3],
            [-1.5, -3, 1.5, 3],
            [-1.0, -3, 1.5, 3],
            [-0.5, -3, 1.5, 3],
            [0.0, -3, 1.5, 3],
            [-2.0, -2, 1.5, 3],
            [-1.5, -2, 1.5, 3],
            [-1.0, -2, 1.5, 3],
            [-0.5, -2, 1.5, 3],
            [0.0, -2, 1.5, 3],
            [-2.0, -4, 2, 3],
            [-1.5, -4, 2, 3],
            [-1.0, -4, 2, 3],
            [-0.5, -4, 2, 3],
            [0.0, -4, 2, 3],
            [-2.0, -3, 2, 3],
            [-1.5, -3, 2, 3],
            [-1.0, -3, 2, 3],
            [-0.5, -3, 2, 3],
            [0.0, -3, 2, 3],
            [-2.0, -2, 2, 3],
            [-1.5, -2, 2, 3],
            [-1.0, -2, 2, 3],
            [-0.5, -2, 2, 3],
            [0.0, -2, 2, 3],
            [-2.0, -4, 0, 4],
            [-1.5, -4, 0, 4],
            [-1.0, -4, 0, 4],
            [-0.5, -4, 0, 4],
            [0.0, -4, 0, 4],
            [-2.0, -3, 0, 4],
            [-1.5, -3, 0, 4],
            [-1.0, -3, 0, 4],
            [-0.5, -3, 0, 4],
            [0.0, -3, 0, 4],
            [-2.0, -2, 0, 4],
            [-1.5, -2, 0, 4],
            [-1.0, -2, 0, 4],
            [-0.5, -2, 0, 4],
            [0.0, -2, 0, 4],
            [-2.0, -4, 0.5, 4],
            [-1.5, -4, 0.5, 4],
            [-1.0, -4, 0.5, 4],
            [-0.5, -4, 0.5, 4],
            [0.0, -4, 0.5, 4],
            [-2.0, -3, 0.5, 4],
            [-1.5, -3, 0.5, 4],
            [-1.0, -3, 0.5, 4],
            [-0.5, -3, 0.5, 4],
            [0.0, -3, 0.5, 4],
            [-2.0, -2, 0.5, 4],
            [-1.5, -2, 0.5, 4],
            [-1.0, -2, 0.5, 4],
            [-0.5, -2, 0.5, 4],
            [0.0, -2, 0.5, 4],
            [-2.0, -4, 1, 4],
            [-1.5, -4, 1, 4],
            [-1.0, -4, 1, 4],
            [-0.5, -4, 1, 4],
            [0.0, -4, 1, 4],
            [-2.0, -3, 1, 4],
            [-1.5, -3, 1, 4],
            [-1.0, -3, 1, 4],
            [-0.5, -3, 1, 4],
            [0.0, -3, 1, 4],
            [-2.0, -2, 1, 4],
            [-1.5, -2, 1, 4],
            [-1.0, -2, 1, 4],
            [-0.5, -2, 1, 4],
            [0.0, -2, 1, 4],
            [-2.0, -4, 1.5, 4],
            [-1.5, -4, 1.5, 4],
            [-1.0, -4, 1.5, 4],
            [-0.5, -4, 1.5, 4],
            [0.0, -4, 1.5, 4],
            [-2.0, -3, 1.5, 4],
            [-1.5, -3, 1.5, 4],
            [-1.0, -3, 1.5, 4],
            [-0.5, -3, 1.5, 4],
            [0.0, -3, 1.5, 4],
            [-2.0, -2, 1.5, 4],
            [-1.5, -2, 1.5, 4],
            [-1.0, -2, 1.5, 4],
            [-0.5, -2, 1.5, 4],
            [0.0, -2, 1.5, 4],
            [-2.0, -4, 2, 4],
            [-1.5, -4, 2, 4],
            [-1.0, -4, 2, 4],
            [-0.5, -4, 2, 4],
            [0.0, -4, 2, 4],
            [-2.0, -3, 2, 4],
            [-1.5, -3, 2, 4],
            [-1.0, -3, 2, 4],
            [-0.5, -3, 2, 4],
            [0.0, -3, 2, 4],
            [-2.0, -2, 2, 4],
            [-1.5, -2, 2, 4],
            [-1.0, -2, 2, 4],
            [-0.5, -2, 2, 4],
            [0.0, -2, 2, 4],
        ]
    )
    assert np.all(grid == result)


def test_create_grid():
    grid = create_grid(-1, 1, 3, 2)
    result = np.array(
        [
            [-1, -1],
            [0, -1],
            [1, -1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
        ]
    )
    assert np.all(grid == result)


def test_normalization():
    x1 = np.random.randn(100, 1) * 10 + 100
    x2 = np.random.randn(100, 1) * 5 - 10
    X = np.concatenate((x1, x2), axis=1)
    assert len(X.shape) == 2
    assert X.shape[1] == 2
    X_normalized = normalize_data(X)
    assert np.isclose(np.mean(X_normalized[:, 0]), 0.0)
    assert np.isclose(np.std(X_normalized[:, 0]), 1.0)
    assert np.isclose(np.mean(X_normalized[:, 1]), 0.0)
    assert np.isclose(np.std(X_normalized[:, 1]), 1.0)
    X = np.random.randn(50, 5) * 10 - 3.0
    X_normalized = normalize_data(X)
    assert np.allclose(np.mean(X_normalized, axis=0), np.repeat(0.0, 5))
    assert np.allclose(np.std(X_normalized, axis=0), np.repeat(1.0, 5))


def test_min_max_normalization():
    x1 = np.random.randn(100, 1) * 10 + 100
    x2 = np.random.randn(100, 1) * 5 - 10
    X = np.concatenate((x1, x2), axis=1)
    assert len(X.shape) == 2
    assert X.shape[1] == 2
    X_normalized = min_max_normalize_data(X)
    assert np.isclose(np.min(X_normalized[:, 0]), 0.0)
    assert np.isclose(np.max(X_normalized[:, 0]), 1.0)
    assert np.isclose(np.min(X_normalized[:, 1]), 0.0)
    assert np.isclose(np.max(X_normalized[:, 1]), 1.0)
    assert np.argmax(X_normalized[:, 0]) == np.argmax(x1)
    assert np.argmax(X_normalized[:, 1]) == np.argmax(x2)


def test_manhatten_distance():
    X = np.array([[0, 0], [1, 1], [1, 2]])
    X2 = np.array([[0, 1], [1, 1], [2, 1]])
    results = np.array([[1, 2, 3], [1, 0, 1], [2, 1, 2]])
    assert np.allclose(results, manhatten_distance(X, X2))


def test_row_wise_compare():
    x1 = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.03], [3.01, 3.02, 3.03]])
    x2 = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.01]])
    result = np.array([True, False, False])
    assert np.allclose(row_wise_compare(x1, x2), [True, False, False])


def test_row_wise_unique():
    x = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.03], [1.01, 1.02, 1.03]])

    result_x = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.03]])
    result_idx = np.array([0, 1])

    x_uniq, idx = row_wise_unique(x)
    assert np.allclose(x_uniq, result_x)
    assert np.allclose(idx, result_idx)


def test_check1Dlist():
    assert np.all(check1Dlist(1.0, 2) == [1.0, 1.0])
    assert np.all(check1Dlist([1.0], 2) == [1.0, 1.0])
    assert np.all(check1Dlist([1.0, 2.0], 2) == [1.0, 2.0])


def test_filter_nan():
    X = np.arange(16, dtype=float).reshape([8, 2])
    y = np.arange(8, dtype=float).reshape([8, 1])
    y[4, 0] = np.nan
    xx, yy = filter_nan(X, y)

    assert np.all(xx == X[[0, 1, 2, 3, 5, 6, 7]])
    assert np.all(yy == y[[0, 1, 2, 3, 5, 6, 7]])
