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
from tssl.oracles.branin_hoo import BraninHoo

def test_get_grid_data():
    oracle = BraninHoo(0.01)
    oracle.set_context([1], [0.5])
    X, Y = oracle.get_grid_data(10, noisy = False)
    assert X.shape[0] == 10
    assert np.all(X[:,1] == 0.5)

def test_get_random_data():
    oracle = BraninHoo(0.01)
    oracle.set_context([1], [0.5])
    X, Y = oracle.get_random_data(100, noisy=False)
    assert np.all(X[:,1] == 0.5)

def test_get_random_data_in_box():
    oracle = BraninHoo(0.01)
    oracle.set_context([1], [0.5])
    X, Y = oracle.get_random_data_in_box(
        100,
        [0, -np.inf],
        [0.2, 0],
        noisy=False
    )
    
    assert np.all(X[:,0] >= 0)
    assert np.all(X[:,0] <= 0.2)
    
    assert np.all(X[:,1] == 0.5)

if __name__ == "__main__":
    test_get_grid_data()
    test_get_random_data()
    test_get_random_data_in_box()