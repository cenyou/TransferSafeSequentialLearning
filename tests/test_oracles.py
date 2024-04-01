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
import numpy as np
import pytest
import tssl
from tssl.oracles import (
    BraninHoo,
    Eggholder,
    Hartmann3,
    Hartmann6,
    OracleNormalizer
)

@pytest.mark.parametrize("oracle_class", [
    BraninHoo, Eggholder, Hartmann3, Hartmann6
])
def test_standard_oracles(oracle_class):
    oracle = oracle_class(observation_noise=0.1)

    a, b = oracle.get_box_bounds()
    D = oracle.get_dimension()
    X, Y = oracle.get_random_data(1, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == D
    assert Y.shape[0] == 1
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y = oracle.get_random_data_in_box(10, a_set, length*0.4, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)
    

def test_oracle_normalizer():
    oracle = OracleNormalizer(
        BraninHoo(observation_noise=0.1)
    )
    oracle.set_normalization_manually(2.0, 3.0)
    mu, std = oracle.get_normalization()
    assert mu ==2
    assert std ==3
    oracle.set_normalization_by_sampling()

    a, b = oracle.get_box_bounds()
    D = oracle.get_dimension()
    X, Y = oracle.get_random_data(1, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == D
    assert Y.shape[0] == 1
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y = oracle.get_random_data_in_box(10, a_set, length*0.4, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)

