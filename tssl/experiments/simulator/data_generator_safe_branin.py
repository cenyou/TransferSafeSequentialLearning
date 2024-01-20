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
from tssl.oracles import BraninHoo
from tssl.kernels.kernel_factory import KernelFactory

from tssl.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from tssl.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig

from copy import deepcopy
from tssl.pools import PoolFromOracle, PoolWithSafetyFromOracle, TransferPoolFromPools
from tssl.utils.safety_metrices import SafetyAreaMeasure

def parse_args():
    parser = argparse.ArgumentParser(description="MOGP sampler (LMC)")
    parser.add_argument("--folder", default='./experiments/data')
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--parallel", default=False, type=string2bool)
    parser.add_argument("--parallel_idx", default=0, type=int, choices=[i for i in range(100)])
    parser.add_argument("--normalized_output", default=True, type=string2bool)

    parser.add_argument("--init_and_show", default=False, type=string2bool)
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
    n_per_dim = int(100)

    if args.init_and_show:
        print(f'sample gp function to show')
        oracle = BraninHoo(observation_noise=0.1, constants=sample_branin_constant(), normalize_output=False)
        xx, yy = oracle.get_random_data(5000, False)
        mean = yy.mean()
        scale = np.sqrt(np.var(yy))
        oracle = BraninHoo(observation_noise=0.1, constants=sample_branin_constant(), normalize_output=True, normalize_mean=mean, normalize_scale=scale)
        oracle.plot()

    safe_metric = SafetyAreaMeasure()
    safe_metric.set_object_detector(2)

    if args.save_samples:
        oracle = BraninHoo(observation_noise=0.1, normalize_output=True)
        
        for i in range(5):
            if args.parallel:
                if i != args.parallel_idx:
                    continue
            print(f'sample branin function {i}')
            for n_try in range(20):
                path = os.path.join(args.folder, f'branin{i}')
                if not os.path.isdir(args.folder):
                    os.mkdir(args.folder)
                if not os.path.isdir(path):
                    os.mkdir(path)

                try:
                    os.remove(os.path.join(path, 'init_X.txt'))
                    os.remove(os.path.join(path, 'init_Y.txt'))
                except:
                    pass
                
                constants = sample_branin_constant()

                oracle_s = BraninHoo(0.1, constants, False)
                xx, yy = oracle_s.get_random_data(5000, False)
                mean = yy.mean()
                scale = np.sqrt(np.var(yy))
                oracle_s = BraninHoo(observation_noise=0.1, constants=constants, normalize_output=True, normalize_mean=mean, normalize_scale=scale)

                np.savetxt(os.path.join(path, 'constants.txt'), constants)
                np.savetxt(os.path.join(path, 'normalize.txt'), np.array([mean, scale]))

                if args.save_plots:
                    if hasattr(oracle, 'save_color_map'):
                        oracle_s.save_color_map(0, np.inf, file_name='data_source', file_path=path)
                        oracle.save_color_map(0, np.inf, file_name='data_target', file_path=path)
                    else:
                        oracle_s.save_plot(file_name='data_source', file_path=path)
                        oracle.save_plot(file_name='data_target', file_path=path)
                
                main_oracle_s = deepcopy(oracle_s)
                main_oracle_t = deepcopy(oracle)
                
                #####
                print(f'   {n_try}-th try: generate small num of initial safe points')
                pool = TransferPoolFromPools(
                    PoolFromOracle(oracle=main_oracle_s),
                    PoolFromOracle(oracle=main_oracle_t)
                )
                pool.set_task_mode(learning_target=False)
                x_grid, y_grid = pool.get_grid_data(100, noisy=False)
                S_s = (y_grid[:,0] >= 0)
                pool.set_task_mode(learning_target=True)
                x_grid, y_grid = pool.get_grid_data(100, noisy=False)
                S_t = (y_grid[:,0] >= 0)
                
                # want:
                #       1. target has 2 safe regions
                #       2. check areas overlapped with source task, need to have 2 regions >= 0.1
                #       3. return constrained data in one of the regions
                safe_metric.reset()
                safe_metric.true_safe_lands(x_grid[:,:-1], S_t)
                if safe_metric.num_lands >= 2:
                    overlapped_labels = safe_metric.true_positive_lands(S_s)
                    
                    pool.set_task_mode(learning_target=True)
                    
                    overlapped_area = np.asarray( safe_metric.safeland_individual_hit_area[0] )
                    if (overlapped_area >= 0.05).sum() >=2:
                        desired_l = np.argmax(overlapped_area) + 1
                        
                        x_target, y_target = pool.get_random_constrained_data(500, noisy=True, constraint_lower=0)
                        
                        mask_t = safe_metric.label_points(x_target[:,:-1]) == desired_l
                        
                        xx = x_target[mask_t]
                        yy = y_target[mask_t]
                        
                        SS = np.zeros(xx.shape[0], dtype=bool)
                        for nn in range(xx.shape[0]):
                            z_check = pool.source_pool.oracle.query(xx[nn,:-1], noisy=False)
                            SS[nn] = np.all(z_check >= 0)
                        
                        print('      found qualified starting safe data, save them')
                        np.savetxt(os.path.join(path, 'init_X.txt'), xx[SS,:-1])
                        np.savetxt(os.path.join(path, 'init_Y.txt'), yy[SS])
                        break
                print('      no qualified initial points')
                        

                