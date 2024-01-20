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
    parser = argparse.ArgumentParser(description="This is a script for safe AL")
    # general AL experiment arguments
    parser.add_argument("--experiment_data_dir", default='./experiments/data', type=str)
    parser.add_argument("--experiment_output_dir", default='./experiments/safe_AL', type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument('--acquisition_function_config', default='BasicSafePredEntropyAllConfig', type=str)
    parser.add_argument('--validation_type', default='RMSE', type=lambda name: ValidationType[name.upper()], choices=list(ValidationType))
    parser.add_argument("--kernel_config", default='BasicMatern52Config', type=str, choices=['BasicRBFConfig', 'RBFWithPriorConfig', 'BasicMatern52Config', 'Matern52WithPriorConfig'])
    parser.add_argument("--model_config", default='BasicGPModelConfig', type=str, choices=['BasicGPModelConfig', 'BasicGPModelMarginalizedConfig'])
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_pool", default=5000, type=int)
    parser.add_argument("--n_data_initial", default=15, type=int)
    parser.add_argument("--n_steps", default=30, type=int)
    parser.add_argument("--n_data_test", default=200, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    # data arguments
    parser.add_argument("--simulator_config", default='SingleTaskIllustrateConfig', type=str)
    parser.add_argument("--safe_lower", default=[0.0], type=float, nargs='+')
    parser.add_argument("--safe_upper", default=[np.inf], type=float, nargs='+')
    parser.add_argument("--label_safeland", default=False, type=string2bool)
    args = parser.parse_args()
    return args

def experiment(args):
    exp_idx = args.experiment_idx
    
    n_data_initial = args.n_data_initial
    n_pool = args.n_pool
    n_steps = args.n_steps
    n_data_test = args.n_data_test
    # experiment settings
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
        data_path = args.experiment_data_dir,
        seed = 2022 + exp_idx
    )
    pool = SimulatorFactory.build(simulator_config)

    # initial data & test data
    print('generate constrained data')
    if simulator_config.initial_data_source == InitialDataGenerationMethod.FILE:
        data_init = InitialDataHandler.return_data(n_data_initial, data_path=args.experiment_data_dir, oracle_type=simulator_config.oracle_type.lower())
    elif simulator_config.initial_data_source == InitialDataGenerationMethod.GENERATE:
        data_init = pool.get_random_constrained_data_in_box(n_data_initial, simulator_config.target_box_a, simulator_config.target_box_width, noisy=True, constraint_lower=safe_lower, constraint_upper=safe_upper)
    elif simulator_config.initial_data_source == InitialDataGenerationMethod.POOL_SELECT:
        data_init = pool.get_data_from_idx(simulator_config.target_data_idx, noisy=True)

    data_test = pool.get_random_constrained_data(n_data_test, noisy=True, constraint_lower=safe_lower, constraint_upper=safe_upper)

    if args.label_safeland:
        data_grid = pool.get_grid_data(40, noisy=False)

    # create kernels and models
    kernel_config = ConfigPicker.pick_kernel_config(args.kernel_config)(
        input_dimension=pool.get_variable_dimension(),
        fix_variance=True
    )
    model_config = ConfigPicker.pick_model_config(args.model_config)(
        kernel_config=kernel_config,
        observation_noise=0.01,
        train_likelihood_variance=True
    )
    safety_model_config = [deepcopy(model_config) for _ in range(acq_func.number_of_constraints)]

    model = ModelFactory.build(model_config)
    safety_model = [ModelFactory.build(sm_config) for sm_config in safety_model_config]

    # save settings
    exp_path = os.path.join(
        args.experiment_output_dir,
        '%s_%s_%s_%s'%(
            acq_config_name,
            args.kernel_config,
            args.model_config,
            args.simulator_config
        )
    )
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    # save experiment setup
    with open(os.path.join(exp_path, f'{exp_idx}_experiment_description.txt'), mode='w') as fp:
        for name, content in args.__dict__.items():
            print(f'{name}  :  {content}', file=fp)

    # initialize optimizer
    learner = SafeActiveLearner(
        acq_func, val_type,
        query_noisy=args.query_noisy,
        model_is_safety_model= not simulator_config.additional_safety,
        save_results=save_results,
        experiment_path=exp_path
        )
    learner.set_pool(pool)
    learner.set_model(model, safety_models=safety_model)
    
    # perform the main experiment
    learner.set_train_data(*data_init)
    learner.set_test_data(*data_test)
    if args.label_safeland:
        learner.initialize_safe_area_measure(*data_grid)
    
    regret, _, _ = learner.learn(n_steps)

    # experiment summary
    learner.save_experiment_summary(filename=f"{exp_idx}_SafeAL_result.csv")

if __name__ == "__main__":
    args = parse_args()
    experiment(args)