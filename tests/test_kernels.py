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
from gpflow.kernels.base import Kernel
import torch
from tssl.configs.base_parameters import (
    BASE_KERNEL_VARIANCE,
    BASE_KERNEL_LENGTHSCALE,
)
from tssl.kernels.kernel_factory import KernelFactory
from tssl.configs.models.gp_model_config import BasicGPModelConfig
from tssl.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from tssl.kernels.rbf_kernel import RBFKernel
from tssl.configs.kernels.matern52_configs import BasicMatern52Config
from tssl.configs.kernels.matern32_configs import BasicMatern32Config
import numpy as np
import gpflow
import pytest
import gpytorch
from tssl.models.model_factory import ModelFactory

f64 = gpflow.utilities.to_default_float

@pytest.mark.parametrize("lengthscale", (0.5, 1.0, 2.0))
def test_rbf_kernel(lengthscale):
    kernel_config = BasicRBFConfig(input_dimension=2, base_lengthscale=lengthscale)
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    rbf_kernel = gpflow.kernels.RBF(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale, lengthscale])
    rbf_kernel_eval = rbf_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)
    kernel_evaluated = kernel.K(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    rbf_kernel_eval = rbf_kernel(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)
    kernel_evaluated = kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    rbf_kernel_eval = rbf_kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)
    kernel_config = BasicRBFConfig(
        input_dimension=2,
        base_lengthscale=lengthscale,
        active_on_single_dimension=True,
        active_dimension=1,
    )
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    rbf_kernel = gpflow.kernels.RBF(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale], active_dims=[1])
    rbf_kernel_eval = rbf_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)

@pytest.mark.parametrize("lengthscale", (0.5, 1.0, 2.0))
def test_matern_kernel(lengthscale):
    kernel_config = BasicMatern52Config(input_dimension=2, base_lengthscale=lengthscale)
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    matern_kernel = gpflow.kernels.Matern52(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale, lengthscale])
    matern_kernel_eval = matern_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)
    kernel_evaluated = kernel.K(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    matern_kernel_eval = matern_kernel(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)
    kernel_evaluated = kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    matern_kernel_eval = matern_kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)
    kernel_config = BasicMatern52Config(
        input_dimension=2,
        base_lengthscale=lengthscale,
        active_on_single_dimension=True,
        active_dimension=1,
    )
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    matern_kernel = gpflow.kernels.Matern52(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale], active_dims=[1])
    matern_kernel_eval = matern_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)
