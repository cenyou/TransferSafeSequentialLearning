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
from tssl.pools import (
    PoolFromOracle, PoolWithSafetyFromOracle,
    PoolFromDataSet, PoolWithSafetyFromDataSet,
)
from tssl.pools import TransferPoolFromPools, MultitaskPoolFromPools
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

def test_transfer_pool():
    ps = PoolFromOracle(BraninHoo(0.01))
    pt = PoolFromOracle(BraninHoo(0.01))
    ps.discretize_random(300)
    pt.discretize_random(300)
    pool = TransferPoolFromPools(ps, pt)
    pool.set_task_mode(False)
    pool.set_replacement(True)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 300
    pool.set_replacement(False)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 299

    pool.set_task_mode(True)
    pool.set_replacement(True)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 300
    pool.set_replacement(False)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 299

def test_multitask_pool():
    P = 4
    pool_list = [PoolFromOracle(BraninHoo(0.01)) for _ in range(P)]
    for p in pool_list:
        p.discretize_random(300)
    pool = MultitaskPoolFromPools(pool_list)
    for i in range(P):
        pool.set_task_mode(i)
        pool.set_replacement(True)
        x_pool = pool.possible_queries()
        y = pool.query(x_pool[0], noisy=False)
        assert pool.possible_queries().shape[0] == 300
        pool.set_replacement(False)
        x_pool = pool.possible_queries()
        y = pool.query(x_pool[0], noisy=False)
        assert pool.possible_queries().shape[0] == 299



############
### PoolFromDataSet
############
def test_pool_from_data_basic():
    dataset = PytestSet()
    dataset.load_data_set()
    pool = PoolFromDataSet(dataset, [0,1,2], data_is_noisy=False)
    assert pool.get_dimension() == 3
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == dataset.length
    y = pool.query(xx[0], noisy=False)
    assert y == dataset.y[0,0]
    assert pool.possible_queries().shape[0] == dataset.length

    pool.set_replacement(False)
    y = pool.query(xx[9], noisy=False)
    assert y == dataset.y[9,0]
    assert pool.possible_queries().shape[0] == dataset.length-1

    
def test_pool_from_data_get_unconstrained_data():
    dataset = PytestSet()
    dataset.load_data_set()
    pool = PoolFromDataSet(dataset, [0,1,2], data_is_noisy=False)
    
    xx, yy = pool.get_random_data(50, False)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y, yy)
    assert np.all(mask_x == mask_y)
    assert mask_x.sum() == 50
    for i in range(50):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])

    xx, yy = pool.get_random_data_in_box(10, -0.5, 1, False)
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])
    assert xx.min() >= -0.5
    assert xx.max() <= 0.5 # -0.5 + 1

def test_pool_from_data_get_constrained_data():
    dataset = PytestSet()
    dataset.load_data_set()
    pool = PoolFromDataSet(dataset, [0,1,2], data_is_noisy=False)
    
    xx, yy = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])
    assert yy.min() >= -0.5
    assert yy.max() <= 0.6

    xx, yy = pool.get_random_constrained_data_in_box(10, -0.5, 1, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])
    assert xx.min() >= -0.5
    assert xx.max() <= 0.5
    assert yy.min() >= -0.5
    assert yy.max() <= 0.6


############
### PoolWithSafetyFromDataSet
############
def test_pool_with_safety_from_data_basic():
    dataset = PytestMOSet()
    dataset.load_data_set()
    pool = PoolWithSafetyFromDataSet(dataset, [0,1,2], [0], [1], data_is_noisy=False)
    assert pool.get_dimension() == 3
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == dataset.length
    y, z = pool.query(xx[0], noisy=False)
    assert y == dataset.y[0,0]
    assert z == dataset.y[0,1]
    assert pool.possible_queries().shape[0] == dataset.length

    pool.set_replacement(False)
    y, z = pool.query(xx[9], noisy=False)
    assert y == dataset.y[9,0]
    assert z == dataset.y[9,1]
    assert pool.possible_queries().shape[0] == dataset.length-1

    
def test_pool_with_safety_from_data_get_unconstrained_data():
    dataset = PytestMOSet()
    dataset.load_data_set()
    pool = PoolWithSafetyFromDataSet(dataset, [0,1,2], [0], [1], data_is_noisy=False)
    
    xx, yy, zz = pool.get_random_data(50, False)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 50
    for i in range(50):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])

    xx, yy, zz = pool.get_random_data_in_box(10, [-1, -0.5, 0], [0.5, 1, 0.5], False)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 10
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])
    assert np.all(xx.min(axis=0) >= [-1, -0.5, 0])
    assert np.all(xx.max(axis=0) <= [-0.5, 0.5, 0.5])

def test_pool_with_safety_from_data_get_constrained_data():
    dataset = PytestMOSet()
    dataset.load_data_set()
    pool = PoolWithSafetyFromDataSet(dataset, [0,1,2], [0], [1], data_is_noisy=False)
    
    xx, yy, zz = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 10
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])
    assert zz.min() >= -0.5
    assert zz.max() <= 0.6

    xx, yy, zz = pool.get_random_constrained_data_in_box(10, [-1, -0.5, 0], [0.5, 1, 0.5], False, constraint_lower=-0.5, constraint_upper=0.6)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 10
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])
    assert np.all(xx.min(axis=0) >= [-1, -0.5, 0])
    assert np.all(xx.max(axis=0) <= [-0.5, 0.5, 0.5])
    assert zz.min() >= -0.5
    assert zz.max() <= 0.6

