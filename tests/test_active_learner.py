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
import pytest
import gpflow
from tssl.enums.global_model_enums import PredictionQuantity
from tssl.pools.pool_from_oracle import PoolFromOracle
import numpy as np
from tssl.oracles.branin_hoo import BraninHoo
from tssl.models.model_factory import ModelFactory
from tssl.kernels.kernel_factory import KernelFactory
from tssl.configs.models.gp_model_config import GPModelFastConfig
from tssl.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from tssl.enums.active_learner_enums import (
    ValidationType,
)
from tssl.active_learner.safe_active_learner import SafeActiveLearner
from tssl.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from tssl.configs.acquisition_function.safe_acquisition_functions.safe_pred_entropy_config import BasicSafePredEntropyAllConfig

def test_safe_active_learner(tmp_path):
    oracle = BraninHoo(0.1, normalize_output=True)
    X, Y = oracle.get_random_data(100, noisy=False)
    Y_max = np.max(Y)
    Y_min = np.min(Y)
    safe_lower = 0.9 * Y_min + 0.1 * Y_max
    safe_upper = 0.1 * Y_min + 0.9 * Y_max

    pool = PoolFromOracle(oracle)
    pool.discretize_random(2000)
    acq_func = AcquisitionFunctionFactory.build(
        BasicSafePredEntropyAllConfig(safety_thresholds_lower=safe_lower, safety_thresholds_upper=safe_upper)
    )
    model = ModelFactory.build(
        GPModelFastConfig(kernel_config = BasicRBFConfig(input_dimension=oracle.get_dimension()))
    )
    data_init = pool.get_random_data(10, noisy=True)
    data_test = pool.get_random_data(100, noisy=True)

    learner = SafeActiveLearner(
        acq_func, ValidationType.RMSE,
        query_noisy=True,
        model_is_safety_model=True,
        save_results=True,
        experiment_path=tmp_path
    )
    learner.set_pool(pool)
    learner.set_model(model, safety_models=None)
    
    # perform the main experiment
    learner.set_train_data(*data_init)
    learner.set_test_data(*data_test)
    _, _, _ = learner.learn(5)
