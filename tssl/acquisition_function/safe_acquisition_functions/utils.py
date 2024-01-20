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
from typing import Union, Sequence, Optional
import numpy as np
from scipy.stats import norm
from gpflow.kernels import Matern52
from tssl.models.base_model import BaseModel


def get_safety_models(
    model: BaseModel,
    safety_models: Optional[Sequence[BaseModel]] = None
):
    if safety_models is None:
        return [model]
    else:
        return safety_models


def compute_gp_posterior(
    x_grid: np.ndarray,
    model: BaseModel,
    safety_models: Optional[Sequence[BaseModel]] = None
):
    if safety_models is None:
        num_models = 1
        models = [model]
    else:
        num_models = 1 + len(safety_models)
        models = [model] + safety_models
    
    pred_mu = np.zeros([x_grid.shape[0], num_models])
    pred_sigma = np.zeros([x_grid.shape[0], num_models])

    if num_models == 1:
        pred_mu[:, 0], pred_sigma[:, 0] = model.predictive_dist(x_grid)
    else:
        for j, m in enumerate(models):
            pred_mu[:, j], std = m.predictive_dist(x_grid)
            if hasattr(m.model.kernel, 'prior_scale'):
                pred_sigma[:, j] = std / m.model.kernel.prior_scale
            else:
                raise NotImplementedError
    return pred_mu, pred_sigma

