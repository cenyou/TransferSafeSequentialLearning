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
import os
import numpy as np
import argparse
from tssl.utils.utils import string2bool
from tssl.oracles import Hartmann3, OracleNormalizer
from tssl.kernels.kernel_factory import KernelFactory

from tssl.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from tssl.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig

from copy import deepcopy
from tssl.pools import PoolFromOracle, PoolWithSafetyFromOracle, TransferPoolFromPools
from tssl.utils.safety_metrices import SafetyAreaMeasure

def parse_args():
    parser = argparse.ArgumentParser(description="Branin sampler")
    parser.add_argument("--folder", default="U:\\Projects\\experiments\\data")
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--parallel", default=False, type=string2bool)
    parser.add_argument("--parallel_idx", default=0, type=int, choices=[i for i in range(100)])
    parser.add_argument("--normalized_output", default=True, type=string2bool)

    parser.add_argument("--save_samples", default=True, type=string2bool)
    args = parser.parse_args()
    return args
root_path = ''

if __name__ == "__main__":
    args = parse_args()
    if args.parallel:
        seed = args.seed + args.parallel_idx
    else:
        seed = args.seed
    np.random.seed(seed)
    D = 3

    if args.save_samples:
        oracle = OracleNormalizer(
            Hartmann3(observation_noise=0.01)
        )
        oracle.set_normalization_manually(-0.9266432501735707, 0.9351442960268146)
        
        for i in range(5):
            if args.parallel:
                if i != args.parallel_idx:
                    continue
            print(f'sample hartmann3 function {i}')
            for n_try in range(20):
                path = os.path.join(args.folder, f'hartmann3_{i}')
                
                try:
                    os.mkdir(path)
                except:
                    pass
                
                try:
                    os.remove(os.path.join(path, 'constants.txt'))
                    os.remove(os.path.join(path, 'normalize.txt'))
                    os.remove(os.path.join(path, 'init_X.txt'))
                    os.remove(os.path.join(path, 'init_Y.txt'))
                except:
                    pass
                
                constants = oracle.base_oracle.sample_constants()

                oracle_s = OracleNormalizer(
                    Hartmann3(observation_noise=0.01, constants=constants)
                )
                oracle_s.set_normalization_by_sampling()
                mean, scale = oracle_s.get_normalization()

                np.savetxt(os.path.join(path, 'constants.txt'), constants)
                np.savetxt(os.path.join(path, 'normalize.txt'), np.array([mean, scale]))
                
                main_oracle_s = deepcopy(oracle_s)
                main_oracle_t = deepcopy(oracle)
                #####
                print(f'   {n_try}-th try: generate small num of initial safe points')
                x_grid, y_grid = main_oracle_s.get_random_data(5000, noisy=False)
                S_s = (y_grid[:,0] >= 0)
                y_grid = main_oracle_t.batch_query(x_grid, noisy=False).reshape([-1, 1])
                S_t = (y_grid[:,0] >= 0)
                
                # want:
                #       1. check areas overlapped with source task, overlapped with area >= 0.1
                #       2. return constrained data in one of the regions
                overlapped_area = (S_s & S_t).astype(int).mean()

                if overlapped_area >= 0.2:
                    pool = TransferPoolFromPools(
                        PoolFromOracle(oracle=main_oracle_s),
                        PoolFromOracle(oracle=main_oracle_t)
                    )
                    pool.set_task_mode(learning_target=True)
                    x_target, y_target = pool.get_random_constrained_data(500, noisy=True, constraint_lower=0)

                    print('      found qualified starting safe data, save them')
                    np.savetxt(os.path.join(path, 'init_X.txt'), x_target[...,:-1])
                    np.savetxt(os.path.join(path, 'init_Y.txt'), y_target)
                    break
                print('      no qualified initial points')
                        

                