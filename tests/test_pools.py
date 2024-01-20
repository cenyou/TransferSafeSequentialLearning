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
from scipy.stats import tstd
from tssl.oracles.flexible_oracle import Flexible1DOracle
from tssl.oracles.flexible_example_functions import f
from tssl.oracles.branin_hoo import BraninHoo
from tssl.data_sets.pytest_set import PytestSet, PytestMOSet
from tssl.pools import PoolFromOracle, PoolWithSafetyFromOracle
from tssl.utils.utils import row_wise_compare, row_wise_unique

def test_pool_from_oracle_basic():
    oracle = Flexible1DOracle(1e-6)
    oracle.set_f(f)
    pool = PoolFromOracle(oracle, seed=123, set_seed=True)
    pool.discretize_random(200)
    assert pool.get_dimension() == 1
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == 200
    y = pool.query(xx[0], noisy=False)
    assert y == oracle.query(xx[0], noisy=False)
    assert pool.possible_queries().shape[0] == 200

    pool.set_replacement(False)
    y = pool.query(xx[10], noisy=False)
    assert y == oracle.query(xx[10], noisy=False)
    assert pool.possible_queries().shape[0] == 199
    
def test_pool_from_oracle_get_unconstrained_data():
    oracle = Flexible1DOracle(1e-6)
    oracle.set_f(f)
    pool = PoolFromOracle(oracle, seed=123, set_seed=True)

    X, Y = pool.get_random_data(100, False)
    assert X.shape[0] == 100
    for i in range(100):
        assert Y[i] == oracle.query(X[i], False)
    
    X, Y = pool.get_random_data_in_box(10, -0.5, 1, False)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
    assert X.min() >= -0.5
    assert X.max() <= 0.5 # -0.5 + 1

def test_pool_from_oracle_get_constrained_data():
    oracle = Flexible1DOracle(1e-6)
    oracle.set_f(f)
    pool = PoolFromOracle(oracle, seed=123, set_seed=True)

    X, Y = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
    assert Y.min() >= -0.5
    assert Y.max() <= 0.6

    X, Y = pool.get_random_constrained_data_in_box(10, -0.1, 1, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
    assert X.min() >= -0.1
    assert X.max() <= 0.9
    assert Y.min() >= -0.5
    assert Y.max() <= 0.6


def test_pool_with_safety_from_oracle_basic():
    oracle = BraninHoo(1e-6)
    safety_oracles = [
        BraninHoo(1e-6, np.array([1, 1, 2, 4, 8, 1])),
        BraninHoo(1e-6, np.array([1.1, 2, 1.8, 3.4, 8.5, 2]))
    ]
    pool = PoolWithSafetyFromOracle(oracle, safety_oracles, seed=123, set_seed=True)
    pool.discretize_random(200)
    assert pool.get_dimension() == 2
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == 200
    y, z = pool.query(xx[0], noisy=False)
    assert y == oracle.query(xx[0], noisy=False)
    assert np.all(z == np.array([so.query(xx[0], noisy=False) for so in safety_oracles]))
    assert z.shape == (2,)
    assert pool.possible_queries().shape[0] == 200

    pool.set_replacement(False)
    y, z = pool.query(xx[10], noisy=False)
    assert y == oracle.query(xx[10], noisy=False)
    assert np.all(z == np.array([so.query(xx[10], noisy=False) for so in safety_oracles]))
    assert pool.possible_queries().shape[0] == 199
    
def test_pool_with_safety_from_oracle_get_unconstrained_data():
    oracle = BraninHoo(1e-6)
    safety_oracles = [
        BraninHoo(1e-6, np.array([1, 1, 2, 4, 8, 1])),
        BraninHoo(1e-6, np.array([1.1, 2, 1.8, 3.4, 8.5, 2]))
    ]
    pool = PoolWithSafetyFromOracle(oracle, safety_oracles, seed=123, set_seed=True)

    X, Y, Z = pool.get_random_data(100, False)
    assert X.shape[0] == 100
    for i in range(100):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    
    X, Y, Z = pool.get_random_data_in_box(10, -0.5, 1, False)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    assert X.min() >= -0.5
    assert X.max() <= 0.5 # -0.5 + 1

def test_pool_with_safety_from_oracle_get_constrained_data():
    oracle = BraninHoo(1e-6)
    safety_oracles = [
        BraninHoo(1e-6, np.array([1, 1, 2, 4, 8, 1]), normalize_output=True, normalize_mean=0.0, normalize_scale=300),
        BraninHoo(1e-6, np.array([1.1, 2, 1.8, 3.4, 8.5, 2]), normalize_output=True, normalize_mean=0.0, normalize_scale=300)
    ]
    pool = PoolWithSafetyFromOracle(oracle, safety_oracles, seed=123, set_seed=True)

    X, Y, Z = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    assert Z.min() >= -0.5
    assert Z.max() <= 0.6

    X, Y, Z = pool.get_random_constrained_data_in_box(10, -0.1, 1, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    assert X.min() >= -0.1
    assert X.max() <= 0.9
    assert Z.min() >= -0.5
    assert Z.max() <= 0.6


