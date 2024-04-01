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
import gpflow
from tssl.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from tssl.kernels.multi_output_kernels.coregionalization_kernel import CoregionalizationSOKernel
from tssl.kernels.multi_output_kernels.coregionalization_Platent_kernel import CoregionalizationPLKernel
from tssl.kernels.multi_output_kernels.multi_source_additive_kernel import MIAdditiveKernel

N = 10
D = 2 # don't change this
P = 2 # don't change this

X1 = np.random.standard_normal(size=[N, D])
X2 = np.random.standard_normal(size=[N, D])
idx_col = np.array([False, True]*5).reshape([10,1]) # bond to P=2

X1_flat = np.hstack((X1, idx_col))
X2_flat = np.hstack((X2, idx_col))

def test_coregionalizationSOkernel():
    kernel = CoregionalizationSOKernel(
        [1.0, 0.9], # bond to D=2
        [0.8, 0.7], # bond to D=2
        D, P,
        add_prior=True,
        lengthscale_prior_parameters=(1,9),
        variance_prior_parameters=(1,0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name='hello'
    )
    W = np.hstack([
        kernel.kernel.W_list[i].numpy().reshape([P, -1]) for i in range(D)
    ]) # [P, D]
    
    k_mat52_L1 = gpflow.kernels.Matern52(1.0, 0.8)
    k_mat52_L2 = gpflow.kernels.Matern52(0.9, 0.7)
    assert np.allclose(
        kernel(X1_flat, X2_flat),
        W[idx_col.astype(int), 0].T * k_mat52_L1(X1, X2) * W[idx_col.astype(int), 0] + \
        W[idx_col.astype(int), 1].T * k_mat52_L2(X1, X2) * W[idx_col.astype(int), 1]
    )
    assert np.allclose(
        kernel(X1_flat, full_cov=False),
        1.0 * W[idx_col.astype(int), 0].reshape(-1) ** 2 + \
        0.9 * W[idx_col.astype(int), 1].reshape(-1) ** 2
    )

def test_coregionalizationPLkernel():
    kernel = CoregionalizationPLKernel(
        base_variance = 1.0,
        base_lengthscale=1.0,
        W_rank=2,
        input_dimension=D,
        output_dimension=P,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name='test'
    )
    k_mat52 = gpflow.kernels.Matern52(1.0, 1.0)
    k_cor = gpflow.kernels.Coregion(output_dim=P,rank=2)
    assert np.allclose( kernel(X1_flat, X2_flat), P* k_mat52(X1, X2) * k_cor(idx_col, idx_col) )
    assert np.allclose( kernel(X1_flat, full_cov=False), P* k_mat52(X1, full_cov=False) * k_cor(idx_col, full_cov=False) )

def test_miakernel():
    kernel = MIAdditiveKernel(
        base_variance = 1.0,
        base_lengthscale =  0.8,
        input_dimension=D,
        output_dimension=P,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name='test'
    )
    k_source = gpflow.kernels.Matern52(1.0, 0.8)
    output_filter = gpflow.kernels.Linear() 
    k_difference = gpflow.kernels.Matern52(1.0, 0.8)
    
    assert np.allclose( kernel(X1_flat, X2_flat), k_source(X1, X2) + output_filter(idx_col.astype(float), idx_col.astype(float))*k_difference(X1, X2) )
    assert np.allclose( kernel(X1_flat, full_cov=False), k_source(X1, full_cov=False) + output_filter(idx_col.astype(float), full_cov=False)*k_difference(X1, full_cov=False) )

