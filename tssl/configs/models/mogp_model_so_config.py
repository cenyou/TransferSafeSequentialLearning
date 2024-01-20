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
from tssl.configs.models.base_model_config import BaseModelConfig
from tssl.configs.kernels.base_kernel_config import BaseKernelConfig
from tssl.enums.global_model_enums import PredictionQuantity

class BasicSOMOGPModelConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise : float = 0.1
    expected_observation_noise : float = 0.3
    optimize_hps : bool = True
    train_likelihood_variance : bool = True
    pertube_parameters_at_start : bool =False
    pertubation_at_start: float = 0.5
    pertubation_for_multistart_opt: float = 0.5
    perform_multi_start_optimization: bool = True
    n_starts_for_multistart_opt: int = 5
    set_prior_on_observation_noise : bool =False
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name : str = "BasicSOMOGP"

if __name__ == '__main__':
    config = BasicMOGPModelConfig()