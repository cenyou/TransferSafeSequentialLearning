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
import pytest
from tssl.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from tssl.configs.acquisition_function import (
    BasicRandomConfig,
    BasicSafeRandomConfig,
    BasicPredVarianceConfig,
    BasicPredEntropyConfig,
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
)
from tssl.acquisition_function import (
    StandardAlphaAcquisitionFunction,
    StandardBetaAcquisitionFunction,

    Random,
    SafeRandom,
    PredVariance,
    PredEntropy,
    SafePredEntropy,
    SafePredEntropyAll,
)


from tssl.configs.config_picker import ConfigPicker
from tssl.models.model_factory import ModelFactory

kernel_config = ConfigPicker.pick_kernel_config('Matern52WithPriorConfig')(
    input_dimension = 3,
    fix_variance = True,
    add_prior=False
)
model_config = ConfigPicker.pick_model_config('BasicGPModelConfig')(
    kernel_config = kernel_config,
    observation_noise = 0.1,
    optimize_hps = False
)
model = ModelFactory.build(model_config)
N = 50
N_test = 20
X = np.random.normal(0, 0.5, size=[N+N_test, 3])
Y = np.exp(X[:,[0]]**2 - np.sin(X[:, [1]]) + X[:, [2]]) + np.random.normal(0, 0.2, size=[N+N_test, 1])


Xtrain = X[:N]
Ytrain = Y[:N]
Xt = X[N:]
Yt = Y[N:]
model.infer(Xtrain, Ytrain)


config2function = {
    0: (BasicRandomConfig, Random),
    1: (BasicPredVarianceConfig, PredVariance),
    2: (BasicPredEntropyConfig, PredEntropy),
}

safe_config2function = {
    0: (BasicSafeRandomConfig, SafeRandom),
    1: (BasicSafePredEntropyConfig, SafePredEntropy),
    2: (BasicSafePredEntropyAllConfig, SafePredEntropyAll),
}


@pytest.mark.parametrize(
    "config,acquisition_class",
    [
        (config_class(), function_class) for config_class, function_class in config2function.values()
    ] + [
        (config_class(safety_thresholds_lower=-np.inf, safety_thresholds_upper=np.inf), function_class) for config_class, function_class in safe_config2function.values()
    ]
)
def test_acquisition_function_factory(config, acquisition_class):
    assert isinstance(AcquisitionFunctionFactory.build(config), acquisition_class)

@pytest.mark.parametrize(
    "config",
    [config_class() for config_class, _ in config2function.values()]
)
def test_acquisition_score(config):
    acq_object = AcquisitionFunctionFactory.build(config)
    score = acq_object.acquisition_score(
        Xt,
        model,
        safety_models = None,
        x_data = Xtrain,
        y_data = Ytrain,
    )
    assert score.shape == (N_test, )


@pytest.mark.parametrize(
    "config",
    [config_class(safety_thresholds_lower=-np.inf, safety_thresholds_upper=np.inf) for config_class, _ in safe_config2function.values()]
)
def test_acquisition_safe_set(config):
    acq_object = AcquisitionFunctionFactory.build(config)
    S = acq_object.compute_safe_set(Xt, safety_models = [model])
    assert S.shape == (N_test, )
    assert S.dtype == bool


@pytest.mark.parametrize(
    "config",
    [config_class(safety_thresholds_lower=-np.inf, safety_thresholds_upper=np.inf) for config_class, _ in safe_config2function.values()]
)
def test_acquisition_safe_score(config):
    acq_object = AcquisitionFunctionFactory.build(config)
    score, S = acq_object.acquisition_score(
        Xt,
        model,
        safety_models = None,
        x_data = Xtrain,
        y_data = Ytrain,
        return_safe_set=True
    )
    assert score.shape == (N_test, )
    assert S.shape == (N_test, )
