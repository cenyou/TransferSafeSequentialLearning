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
from tssl.oracles import MOGP1DOracle, MOGP2DOracle
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
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--z_num", default=1, type=int)
    parser.add_argument("--normalized_output", default=True, type=string2bool)

    parser.add_argument("--init_and_show", default=False, type=string2bool)
    parser.add_argument("--save_samples", default=True, type=string2bool)
    parser.add_argument("--save_plots", default=True, type=string2bool)
    args = parser.parse_args()
    return args
root_path = ''

def sample_kernel(D=1, P=2, R=2, L=2):
    kernel_list = [
        KernelFactory.build(BasicMatern52Config(
            input_dimension=D,
            base_lengthscale=list(np.random.uniform(0.1, 1, size=[D])),
            base_variance=1
        )) for _ in range(L)
    ]
    W = np.zeros([L, P, R], dtype=float)
    for l in range(L):
        w_l = np.random.uniform(-1, 1, size=[P, R])
        w_norm = np.linalg.norm(w_l, axis=1).reshape([P,1])
        w_l = w_l/w_norm
        W[l,:,:] = w_l
    
    return kernel_list, W

if __name__ == "__main__":
    args = parse_args()
    if args.parallel:
        seed = args.seed + args.parallel_idx
    else:
        seed = args.seed
    np.random.seed(seed)
    D = args.dim
    z_dim = args.z_num
    n_per_dim = int(100)

    print('generate kernel')
    
    if args.init_and_show:
        print(f'sample gp function to show')
        kernel_list, W = sample_kernel(D, P=2, R=2, L=2)
        if D == 1:
            oracle = MOGP1DOracle(kernel_list, W, observation_noise=0.1, specified_output = -1)
        elif D == 2:
            oracle = MOGP2DOracle( kernel_list, W, observation_noise=0.1, specified_output = -1)

        oracle.initialize(-2, 2, n_per_dim, normalized_output=args.normalized_output)
        oracle.plot()


    safe_metric = SafetyAreaMeasure()
    safe_metric.set_object_detector(D)

    if args.save_samples:
        for i in range(30):
            if args.parallel:
                if i != args.parallel_idx:
                    continue
            print(f'sample gp function to save {i}')
            for n_try in range(20):
                path = os.path.join(args.folder, f'mogp{D}d_z', f'{i}')
                if not os.path.isdir(args.folder):
                    os.mkdir(args.folder)
                if not os.path.isdir(os.path.join(args.folder, f'mogp{D}d_z')):
                    os.mkdir(os.path.join(args.folder, f'mogp{D}d_z'))
                if not os.path.isdir(path):
                    os.mkdir(path)
                
                kernel_list, W = sample_kernel(D, P=2, R=2, L=2)
                
                try:
                    os.remove(os.path.join(path, 'init_X.txt'))
                    os.remove(os.path.join(path, 'init_Y.txt'))
                    os.remove(os.path.join(path, 'init_Z.txt'))
                except:
                    pass
                if D == 1:
                    oracle = MOGP1DOracle(kernel_list, W, observation_noise=0.1, specified_output = -1)
                elif D == 2:
                    oracle = MOGP2DOracle(kernel_list, W, observation_noise=0.1, specified_output = -1)

                oracle.save_gp_initialization_to_txt(-2, 2, n_per_dim, path, normalized_output=args.normalized_output)

                if args.save_plots:
                    oracle.initialize_from_txt(path)
                    if hasattr(oracle, 'save_color_map'):
                        oracle.set_specified_output(0)
                        oracle.save_color_map(0, np.inf, file_name='data_source', file_path=path)
                        oracle.set_specified_output(1)
                        oracle.save_color_map(0, np.inf, file_name='data_target', file_path=path)
                    else:
                        oracle.set_specified_output(0)
                        oracle.save_plot(file_name='data_source', file_path=path)
                        oracle.set_specified_output(1)
                        oracle.save_plot(file_name='data_target', file_path=path)
                
                main_oracle_s = deepcopy(oracle)
                main_oracle_s.set_specified_output(0)
                main_oracle_t = deepcopy(oracle)
                main_oracle_t.set_specified_output(1)
                safety_oracles_s = []
                safety_oracles_t = []
                for j in range(z_dim):
                    oracle.save_gp_initialization_to_txt(-2, 2, n_per_dim, path, output_name=f'Z{j}.txt', normalized_output=args.normalized_output)

                    if args.save_plots:
                        oracle.initialize_from_txt(path, output_name=f'Z{j}.txt')
                        if hasattr(oracle, 'save_color_map'):
                            oracle.set_specified_output(0)
                            oracle.save_color_map(0, np.inf, file_name=f'data_source_Z{j}', file_path=path)
                            oracle.set_specified_output(1)
                            oracle.save_color_map(0, np.inf, file_name=f'data_target_Z{j}', file_path=path)
                        else:
                            oracle.set_specified_output(0)
                            oracle.save_plot(file_name=f'data_source_Z{j}', file_path=path)
                            oracle.set_specified_output(1)
                            oracle.save_plot(file_name=f'data_target_Z{j}', file_path=path)
                    
                    so_s = deepcopy(oracle)
                    so_s.set_specified_output(0)
                    safety_oracles_s.append(so_s)
                    
                    so_t = deepcopy(oracle)
                    so_t.set_specified_output(1)
                    safety_oracles_t.append(so_t)
                

                #####
                print(f'   {n_try}-th try: generate small num of initial safe points')
                pool = TransferPoolFromPools(
                    PoolWithSafetyFromOracle(oracle=main_oracle_s, safety_oracle=safety_oracles_s),
                    PoolWithSafetyFromOracle(oracle=main_oracle_t, safety_oracle=safety_oracles_t)
                )
                pool.set_task_mode(learning_target=False)
                x_grid, y_grid, z_grid = pool.get_grid_data(100, noisy=False)
                S_s = np.all(z_grid >= 0, axis=1)
                
                pool.set_task_mode(learning_target=True)
                x_grid, y_grid, z_grid = pool.get_grid_data(100, noisy=False)
                S_t = np.all(z_grid >= 0, axis=1)
                
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
                    if (overlapped_area >= 0.1).sum() >=2:
                        desired_l = np.argmax(overlapped_area) + 1
                        x_target, y_target, z_target = pool.get_random_constrained_data(500, noisy=True, constraint_lower=0)
                        
                        mask_t = safe_metric.label_points(x_target[:,:-1]) == desired_l
                        
                        xx = x_target[mask_t]
                        yy = y_target[mask_t]
                        zz = z_target[mask_t]
                        
                        SS = np.zeros(xx.shape[0], dtype=bool)
                        for nn in range(xx.shape[0]):
                            z_check = np.array([so.query(xx[nn,:-1], noisy=False) for so in pool.source_pool.safety_oracle])
                            SS[nn] = np.all(z_check >= 0)
                        
                        print('      found qualified starting safe data, save them')
                        np.savetxt(os.path.join(path, 'init_X.txt'), xx[SS,:-1])
                        np.savetxt(os.path.join(path, 'init_Y.txt'), yy[SS])
                        np.savetxt(os.path.join(path, 'init_Z.txt'), zz[SS])
                        break
                print('      no qualified initial points')
                        

                