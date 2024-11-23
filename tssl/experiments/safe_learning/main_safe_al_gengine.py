"""
// Copyright (c) 2024 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0

Authors: Cen-You Li <cen-you.li@de.bosch.com>
"""
import numpy as np
import argparse
import os
import sys
import logging
from copy import deepcopy
from pathlib import Path
from tssl.utils.utils import string2bool
from tssl.active_learner.safe_active_learner import SafeActiveLearner
from tssl.enums.simulator_enums import InitialDataGenerationMethod
from tssl.enums.active_learner_enums import ValidationType
from tssl.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from tssl.models.model_factory import ModelFactory
from tssl.experiments.simulator.simulator_factory import SimulatorFactory

from tssl.configs.config_picker import ConfigPicker

import gpflow
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
#f64 = gpflow.utilities.to_default_float
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="This is a script for safe AL")
    # general AL experiment arguments
    parser.add_argument("--experiment_data_dir", default='./experiments/data', type=str)
    parser.add_argument("--experiment_output_dir", default='./experiments/safe_AL', type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument('--acquisition_function_config', default='BasicSafePredEntropyAllConfig', type=str)
    parser.add_argument('--validation_type', default='RMSE', type=lambda name: ValidationType[name.upper()], choices=list(ValidationType))
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_pool", default=10000, type=int)
    parser.add_argument("--n_data_initial", default=20, type=int)
    parser.add_argument("--n_steps", default=200, type=int)
    parser.add_argument("--update_gp_hps_every_n_iter", default=50, type=int)
    parser.add_argument("--n_data_test", default=10000, type=int)
    parser.add_argument("--safe_lower", default=[-1.5], type=float, nargs='+')
    parser.add_argument("--safe_upper", default=[0.5], type=float, nargs='+')
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
    simulator_config = ConfigPicker.pick_experiment_simulator_config('SingleTaskGEngineTestConfig')(
        n_pool = -1,
        data_path = args.experiment_data_dir,
        seed = 2022 + exp_idx
    )
    pool_test = SimulatorFactory.build(simulator_config)
    simulator_config = ConfigPicker.pick_experiment_simulator_config('SingleTaskGEngineConfig')(
        n_pool = -1, # we take full dataset, otherwise specifying initial data will go wrong
        data_path = args.experiment_data_dir,
        seed = 2022 + exp_idx
    )
    pool = SimulatorFactory.build(simulator_config)

    # initial data & test data
    print('generate constrained data')
    assert simulator_config.initial_data_source == InitialDataGenerationMethod.POOL_SELECT
    data_init = pool.get_data_from_idx(simulator_config.target_data_idx[:n_data_initial], noisy=True)

    pool_test.set_replacement(True)
    data_test = pool_test.get_random_constrained_data(n_data_test, noisy=True, constraint_lower=safe_lower, constraint_upper=safe_upper)
    data_grid = pool_test.get_random_data(n_data_test, noisy=True)

    # create kernels and models
    assert simulator_config.additional_safety
    print('create model(s)')
    kernel_config = ConfigPicker.pick_kernel_config('BasicMatern52Config')(
        input_dimension=pool.get_variable_dimension(),
        fix_variance=False,
    )
    model_config = ConfigPicker.pick_model_config('BasicGPModelConfig')(
        kernel_config=kernel_config,
        optimize_hps = True, perform_multi_start_optimization=True, n_starts_for_multistart_opt=5,
        observation_noise=0.1, train_likelihood_variance=True,
    )
    model = ModelFactory.build(model_config)
    safety_models = [ModelFactory.build(model_config) for _ in range(acq_func.number_of_constraints)]
    # save settings
    exp_path = Path(args.experiment_output_dir) / f'{acq_config_name}_BasicMatern52Config_BasicGPModelConfig_SingleTaskGEngineConfig'
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)
    # save experiment setup
    with open(exp_path / f'{exp_idx}_experiment_description.txt', mode='w') as fp:
        for name, content in args.__dict__.items():
            print(f'{name}  :  {content}', file=fp)

    # initialize optimizer
    learner = SafeActiveLearner(
        acq_func, val_type,
        query_noisy=True,
        model_is_safety_model= not simulator_config.additional_safety,
        run_ccl=False,
        update_gp_hps_every_n_iter=args.update_gp_hps_every_n_iter,
        save_results=save_results,
        experiment_path=exp_path,
        max_pool_size=n_pool
    )
    learner.set_pool(pool)
    learner.set_model(model, safety_models=safety_models)
    
    # perform the main experiment
    learner.set_train_data(*data_init)
    learner.set_test_data(*data_test)
    learner.initialize_safe_area_measure(*data_grid)

    regret, _, _ = learner.learn(n_steps)

    # experiment summary
    learner.save_experiment_summary(filename=f"{exp_idx}_SafeAL_result.csv")


if __name__ == "__main__":
    args = parse_args()
    experiment(args)