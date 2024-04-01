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
import glob
import numpy as np
import argparse
from tssl.utils.utils import string2bool
from tssl.oracles import BraninHoo, OracleNormalizer
from tssl.kernels.kernel_factory import KernelFactory

from tssl.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from tssl.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig

from copy import deepcopy
from tssl.pools import PoolFromOracle, PoolWithSafetyFromOracle, TransferPoolFromPools
from tssl.utils.safety_metrices import SafetyAreaMeasure

def parse_args():
    parser = argparse.ArgumentParser(description="Branin sampler")
    parser.add_argument("--folder", default="./experiments/data")
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--parallel", default=False, type=string2bool)
    parser.add_argument("--num_sources", default=5, type=int)
    parser.add_argument("--parallel_idx", default=0, type=int, choices=[i for i in range(100)])
    parser.add_argument("--normalized_output", default=True, type=string2bool)

    parser.add_argument("--save_samples", default=True, type=string2bool)
    parser.add_argument("--save_plots", default=True, type=string2bool)
    args = parser.parse_args()
    return args
root_path = ''

def sample_branin_constant():
    # values designed according to
    # Tighineanu et al. AISTATS 2022 Transfer Learning with Gaussian Processes for Bayesian Optimization
    a = np.random.uniform(low=0.5, high=1.5)
    b = np.random.uniform(low=0.1, high=0.15)
    c = np.random.uniform(low=1.0, high=2.0)
    r = np.random.uniform(low=5.0, high=7.0)
    s = np.random.uniform(low=8.0, high=12.0)
    t = np.random.uniform(low=0.03, high=0.05)
    
    return np.array([a, b, c, r, s, t])

if __name__ == "__main__":
    args = parse_args()
    if args.parallel:
        seed = args.seed + args.parallel_idx
    else:
        seed = args.seed
    np.random.seed(seed)
    D = 2
    PS = args.num_sources
    n_per_dim = int(100)

    safe_metric = SafetyAreaMeasure()
    safe_metric.set_object_detector(2)

    if args.save_samples:
        oracle = OracleNormalizer(
            BraninHoo(observation_noise=0.01)
        )
        oracle.set_normalization_manually(60.088767740805736, 62.34134408167649)
        
        for i in range(5):
            if args.parallel:
                if i != args.parallel_idx:
                    continue
            print(f'sample branin function {i}')
            path = os.path.join(args.folder, f'multi_sources_branin{i}')
            
            try:
                os.mkdir(path)
            except:
                pass
            
            try:
                for file in glob.glob(os.path.join(path, 'constants_*.txt')):
                    os.remove(file)
                for file in glob.glob(os.path.join(path, 'normalize_*.txt')):
                    os.remove(file)
                os.remove(os.path.join(path, 'init_X.txt'))
                os.remove(os.path.join(path, 'init_Y.txt'))
                for file in glob.glob(os.path.join(path, 'data_source_pre_normalize_*.txt')):
                    os.remove(file)
                os.remove(os.path.join(path, 'data_target_pre_normalize.txt'))
            except:
                pass
            pool_list = []
            for j in range(PS):
                constants = sample_branin_constant()
                oraclej = OracleNormalizer(
                    BraninHoo(observation_noise=0.01, constants=constants)
                )
                oraclej.set_normalization_by_sampling()
                mean, scale = oraclej.get_normalization()
                np.savetxt(os.path.join(path, f'constants_{j}.txt'), constants)
                np.savetxt(os.path.join(path, f'normalize_{j}.txt'), np.array([mean, scale]))
                pool_list.append(PoolFromOracle(oracle=oraclej))
            pool_list.append(PoolFromOracle(oracle=oracle))
            if args.save_plots:
                if hasattr(oracle, 'save_color_map'):
                    for j, pool in enumerate(pool_list[:-1]):
                        pool.oracle.save_color_map(0, np.inf, file_name=f'data_source_pre_normalize_{j}', file_path=path)
                    oracle.save_color_map(0, np.inf, file_name='data_target_pre_normalize', file_path=path)
                else:
                    for j, pool in enumerate(pool_list[:-1]):
                        pool.oracle.save_plot(file_name=f'data_source_pre_normalize_{j}', file_path=path)
                    oracle.save_plot(file_name='data_target_pre_normalize', file_path=path)
                
            x_target, y_target = pool_list[-1].get_random_constrained_data_in_box(200, 0.5, 0.5, noisy=True, constraint_lower=0)
            np.savetxt(os.path.join(path, 'init_X.txt'), x_target)
            np.savetxt(os.path.join(path, 'init_Y.txt'), y_target)


                