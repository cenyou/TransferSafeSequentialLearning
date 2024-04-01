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
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from tssl.utils.utils import string2bool
from tssl.active_learner.safe_active_learner import SafeActiveLearner
from tssl.enums.simulator_enums import InitialDataGenerationMethod
from tssl.enums.active_learner_enums import ValidationType
from tssl.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from tssl.models.model_factory import ModelFactory
from tssl.experiments.simulator.simulator_factory import SimulatorFactory
from tssl.experiments.simulator.initial_data_handler import InitialDataHandler

from tssl.configs.config_picker import ConfigPicker

import gpflow
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
#f64 = gpflow.utilities.to_default_float


def parse_args():
    parser = argparse.ArgumentParser(description="This is a script for safe transfer AL with multiple source tasks")
    # general AL experiment arguments
    parser.add_argument("--experiment_data_dir", default='./experiments/data', type=str)
    parser.add_argument("--experiment_output_dir", default='./experiments/safe_AL', type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument('--acquisition_function_config', default='BasicSafePredEntropyAllConfig', type=str)
    parser.add_argument('--validation_type', default='RMSE', type=lambda name: ValidationType[name.upper()], choices=list(ValidationType))
    parser.add_argument("--kernel_config", default='BasicCoregionalizationPLConfig', type=str)
    parser.add_argument("--model_config", default='BasicSOMOGPModelConfig', type=str)
    parser.add_argument("--plot_iterations", default=True, type=string2bool, help="this only works for 1 or 2 dim input though")
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for transfer setting
    parser.add_argument("--n_pool", default=5000, type=int)
    parser.add_argument("--n_data_s_per_dim", default=20, type=int)
    parser.add_argument("--dim_s", default=5, type=int)
    parser.add_argument("--n_data_initial", default=10, type=int)
    parser.add_argument("--n_steps", default=50, type=int)
    parser.add_argument("--n_data_test", default=200, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    # data arguments
    parser.add_argument("--simulator_config", default='TransferTaskMultiSourcesBranin0Config', type=str)
    parser.add_argument("--safe_lower_source", default=[0.0], type=float, nargs='+')
    parser.add_argument("--safe_upper_source", default=[np.inf], type=float, nargs='+')
    parser.add_argument("--safe_lower", default=[0.0], type=float, nargs='+')
    parser.add_argument("--safe_upper", default=[np.inf], type=float, nargs='+')
    
    parser.add_argument("--label_safeland", default=True, type=string2bool)
    args = parser.parse_args()
    return args

def experiment(args):
    exp_idx = args.experiment_idx
    # source task
    n_pool = args.n_pool
    n_data_s_per_dim = args.n_data_s_per_dim
    dim_s = args.dim_s
    # target task
    n_data_initial = args.n_data_initial
    n_steps = args.n_steps
    n_data_test = args.n_data_test
    # experiment settings
    safe_lower_s = args.safe_lower_source
    safe_upper_s = args.safe_upper_source
    safe_lower = args.safe_lower
    safe_upper = args.safe_upper
    acq_config_name = args.acquisition_function_config
    val_type = args.validation_type
    save_results = args.save_results
    
    # create an acquisition function
    function_config = ConfigPicker.pick_acquisition_function_config(acq_config_name)(
        safety_thresholds_lower=safe_lower,
        safety_thresholds_upper=safe_upper
    )
    acq_func = AcquisitionFunctionFactory.build(function_config)

    # need a pool here
    simulator_config = ConfigPicker.pick_experiment_simulator_config(args.simulator_config)(
        n_pool = n_pool,
        dim_s = dim_s,
        data_path = args.experiment_data_dir,
        seed = 2022 + exp_idx
    )
    pool = SimulatorFactory.build(simulator_config)

    # initial data & test data
    print('generate constrained data')
    for source_i in range(dim_s):
        pool.set_task_mode(pool_index=source_i)
        data_s_i = pool.get_random_constrained_data(n_data_s_per_dim, noisy=True, constraint_lower=safe_lower_s, constraint_upper=safe_upper_s)
        if source_i == 0:
            data_s = data_s_i
        else:
            data_s = [np.concatenate([data_s[j], data_s_i[j]], axis=0) for j in range(len(data_s))]

    pool.set_task_mode(pool_index=pool.output_dimension-1) # target task, the last pool idx

    if simulator_config.initial_data_source == InitialDataGenerationMethod.FILE:
        data_init_t = InitialDataHandler.return_data(n_data_initial, data_path=args.experiment_data_dir, oracle_type=simulator_config.oracle_type.lower())
        data_init_t = pool.data_tuple_decorator(
            data_init_t,
            pool.get_dimension(),
            pool.task_index
        )
    elif simulator_config.initial_data_source == InitialDataGenerationMethod.GENERATE:
        data_init_t = pool.get_random_constrained_data_in_box(n_data_initial, simulator_config.target_box_a, simulator_config.target_box_width, noisy=True, constraint_lower=safe_lower, constraint_upper=safe_upper)
    elif simulator_config.initial_data_source == InitialDataGenerationMethod.POOL_SELECT:
        data_init_t = pool.get_data_from_idx(simulator_config.target_data_idx, noisy=True)

    data_test_t = pool.get_random_constrained_data(n_data_test, noisy=True, constraint_lower=safe_lower, constraint_upper=safe_upper)
    if args.label_safeland:
        data_grid_t = pool.get_grid_data(40, noisy=False)
    else:
        data_grid_t = pool.get_random_data(n_pool, noisy=False)

    # create kernels and models
    kernel_config = ConfigPicker.pick_kernel_config(args.kernel_config)(
        input_dimension=pool.get_variable_dimension(),
        output_dimension=pool.output_dimension
    )
    
    model_config = ConfigPicker.pick_model_config(args.model_config)(
        kernel_config=kernel_config,
        optimize_hps = True, perform_multi_start_optimization=True,
        observation_noise=0.1, train_likelihood_variance=True
        )

    model = ModelFactory.build(model_config)
    if simulator_config.additional_safety:
        safety_model_config = [deepcopy(model_config) for _ in range(acq_func.number_of_constraints)]
        safety_models = [ModelFactory.build(sm_config) for sm_config in safety_model_config]
    else:
        safety_models = None
    if args.model_config == 'BasicMetaGPModelConfig':
        try:
            model.set_input_domain(*pool.pool_list[-1].oracle.get_variable_box_bounds())
            if not safety_models is None:
                for sm in safety_model:
                    sm.set_input_domain(*pool.pool_list[-1].oracle.get_variable_box_bounds())
            print('### use variable_box_bounds', *pool.pool_list[-1].oracle.get_variable_box_bounds())
        except:
            model.set_input_domain(*pool.pool_list[-1].get_box_bounds())
            if not safety_models is None:
                for sm in safety_model:
                    sm.set_input_domain(*pool.pool_list[-1].get_box_bounds())

    # save settings
    exp_path = Path( args.experiment_output_dir) / (
        '%s_%s_%s_Ns_%dx%d_Ninit_%d'%(
            acq_config_name,
            args.kernel_config,
            args.model_config,
            dim_s,
            n_data_s_per_dim,
            n_data_initial
        )
    ) / args.simulator_config
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)
    # save experiment setup
    with open(exp_path / f'{exp_idx}_experiment_description.txt', mode='w') as fp:
        for name, content in args.__dict__.items():
            print(f'{name}  :  {content}', file=fp)

    # initialize optimizer
    learner = SafeActiveLearner(
        acq_func, val_type,
        query_noisy=args.query_noisy,
        run_ccl=args.label_safeland,
        model_is_safety_model= not simulator_config.additional_safety,
        save_results=save_results,
        experiment_path=exp_path
    )
    learner.set_pool(pool)
    learner.set_model(model, safety_models=safety_models)
    
    # perform the main experiment
    learner.set_train_data(*data_s)
    # target task
    learner.add_train_data(*data_init_t)
    learner.set_test_data(*data_test_t)
    learner.initialize_safe_area_measure(*data_grid_t)

    regret_t, _, _ = learner.learn(n_steps)

    # experiment summary
    name = f"{exp_idx}_SafeAL_result.csv"
    learner.save_experiment_summary(filename=name)

if __name__ == "__main__":
    args = parse_args()
    experiment(args)
