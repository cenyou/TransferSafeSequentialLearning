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
import os
from typing import Tuple, Union, Sequence, List

class InitialDataHandler:
    @staticmethod
    def return_data(
        n: int,
        data_path: str=None,
        oracle_type: str="mogp1d0"
    ):
        otl = oracle_type.lower()
        if otl.startswith('branin') or otl.startswith('multi_sources_branin'):
            folder = os.path.join(data_path, otl)
            
            x = np.loadtxt( os.path.join(folder, 'init_X.txt') ).reshape([-1,2])
            y = np.loadtxt( os.path.join(folder, 'init_Y.txt') ).reshape([-1,1])
            return x[:n], y[:n]
        elif otl.startswith('hartmann3'):
            folder = os.path.join(data_path, otl)
                
            x = np.loadtxt( os.path.join(folder, 'init_X.txt') ).reshape([-1,3])
            y = np.loadtxt( os.path.join(folder, 'init_Y.txt') ).reshape([-1,1])
            return x[:n], y[:n]
        elif otl.startswith('mogp1dz'):
            folder = os.path.join(data_path, 'mogp1d_z', otl.split('dz')[-1])
            
            x = np.loadtxt( os.path.join(folder, 'init_X.txt') ).reshape([-1,1])
            y = np.loadtxt( os.path.join(folder, 'init_Y.txt') ).reshape([-1,1])
            z = np.loadtxt( os.path.join(folder, 'init_Z.txt') ).reshape([-1,1])
            return x[:n], y[:n], z[:n]
        elif otl.startswith('mogp2dz'):
            folder = os.path.join(data_path, 'mogp2d_z', otl.split('dz')[-1])
            
            x = np.loadtxt( os.path.join(folder, 'init_X.txt') ).reshape([-1,2])
            y = np.loadtxt( os.path.join(folder, 'init_Y.txt') ).reshape([-1,1])
            z = np.loadtxt( os.path.join(folder, 'init_Z.txt') ).reshape([-1,1])
            return x[:n], y[:n], z[:n]
        else:
            raise NotImplementedError('invalid \'oracle_type\'')
        
