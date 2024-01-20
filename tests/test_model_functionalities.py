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
import logging
from statistics import mode
import time
import gpflow
from tssl.models.model_factory import ModelFactory
from gpflow.utilities.utilities import print_summary
from tssl.configs.kernels.matern52_configs import Matern52WithPriorConfig
from tssl.configs.kernels.matern32_configs import Matern32WithPriorConfig
from tssl.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from tssl.configs.models.gp_model_config import BasicGPModelConfig, GPModelFastConfig, GPModelWithNoisePriorConfig
from tssl.enums.global_model_enums import PredictionQuantity as PredictionQuantMarg
from tssl.enums.global_model_enums import InitialParameters
from tssl.oracles import BraninHoo
import pytest
import numpy as np


@pytest.mark.parametrize("config_class", [BasicGPModelConfig])
def test_gp_models_inference(config_class):
    oracle = BraninHoo(0.01)
    x_data, y_data = oracle.get_random_data(20)
    x_test, _ = oracle.get_random_data(10)
    kernel_config = Matern52WithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = config_class(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    gp_model.infer(x_data, y_data)
    gp_model.predictive_dist(x_test)

@pytest.mark.parametrize(
    "start_parameters_class,kernel_config_class",
    [
        (InitialParameters.PERTURB, Matern52WithPriorConfig),
        (InitialParameters.UNIFORM_DISTRIBUTION, BasicRBFConfig),
    ],
)
def test_gp_model_mutlistart(start_parameters_class, kernel_config_class):
    oracle = BraninHoo(0.01)
    x_data, y_data = oracle.get_random_data(20)
    x_test, _ = oracle.get_random_data(10)
    kernel_config = kernel_config_class(input_dimension=oracle.get_dimension())
    model_config = BasicGPModelConfig(kernel_config=kernel_config, initial_parameter_strategy=start_parameters_class)
    gp_model = ModelFactory.build(model_config)
    time_before = time.perf_counter()
    gp_model.infer(x_data, y_data)
    time_after = time.perf_counter()
    print(time_after - time_before)
    infer_time = gp_model.get_last_inference_time()
    print(infer_time)
    assert np.isclose(gp_model.training_loss(), np.min(gp_model.multi_start_losses), rtol=1e-05, atol=1e-08, equal_nan=False)


def test_gp_model_reset():
    oracle = BraninHoo(0.01)
    x_data, y_data = oracle.get_random_data(20)
    kernel_config = Matern52WithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    lengthscale_before_training = gp_model.kernel.kernel.lengthscales.numpy()
    variance_before_training = gp_model.kernel.kernel.variance.numpy()
    time_before = time.perf_counter()
    gp_model.infer(x_data, y_data)
    time_after = time.perf_counter()
    print(time_after - time_before)
    print(gp_model.get_last_inference_time())
    lengthscale_after_training = gp_model.kernel.kernel.lengthscales.numpy()
    variance_after_training = gp_model.kernel.kernel.variance.numpy()
    assert not np.allclose(lengthscale_before_training, lengthscale_after_training)
    assert not np.allclose(variance_before_training, variance_after_training)
    gp_model.reset_model()
    lengthscale_after_reset = gp_model.kernel.kernel.lengthscales.numpy()
    variance_after_reset = gp_model.kernel.kernel.variance.numpy()
    assert np.allclose(lengthscale_before_training, lengthscale_after_reset)
    assert np.allclose(variance_before_training, variance_after_reset)

