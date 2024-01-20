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
from typing import Union
from tssl.configs.acquisition_function.al_acquisition_functions.base_al_acquisition_function_config import BaseALAcquisitionFunctionConfig
from tssl.configs.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function_config import (
    BaseSafeAcquisitionFunctionConfig,
)
from tssl.configs.acquisition_function.al_acquisition_functions.acq_random_config import BasicRandomConfig
from .al_acquisition_functions.acq_random import Random
from tssl.configs.acquisition_function.safe_acquisition_functions.safe_random_config import BasicSafeRandomConfig
from .safe_acquisition_functions.safe_random import SafeRandom
from tssl.configs.acquisition_function.al_acquisition_functions.pred_variance_config import BasicPredVarianceConfig
from tssl.configs.acquisition_function.al_acquisition_functions.pred_sigma_config import BasicPredSigmaConfig
from .al_acquisition_functions.pred_variance import PredVariance
from tssl.configs.acquisition_function.al_acquisition_functions.pred_entropy_config import BasicPredEntropyConfig
from .al_acquisition_functions.pred_entropy import PredEntropy
from .al_acquisition_functions.pred_sigma import PredSigma
from tssl.configs.acquisition_function.safe_acquisition_functions.safe_pred_entropy_config import (
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
)
from .safe_acquisition_functions.safe_pred_entropy import SafePredEntropy, SafePredEntropyAll


class AcquisitionFunctionFactory:
    @staticmethod
    def build(
        function_config: Union[
            BaseALAcquisitionFunctionConfig,
            BaseSafeAcquisitionFunctionConfig
        ]
    ):
        if isinstance(function_config, BasicRandomConfig):
            return Random(**function_config.dict())
        elif isinstance(function_config, BasicSafeRandomConfig):
            return SafeRandom(**function_config.dict())
        elif isinstance(function_config, BasicPredVarianceConfig):
            return PredVariance(**function_config.dict())
        elif isinstance(function_config, BasicPredSigmaConfig):
            return PredSigma(**function_config.dict())
        elif isinstance(function_config, BasicPredEntropyConfig):
            return PredEntropy(**function_config.dict())
        elif isinstance(function_config, BasicSafePredEntropyConfig):
            return SafePredEntropy(**function_config.dict())
        elif isinstance(function_config, BasicSafePredEntropyAllConfig):
            return SafePredEntropyAll(**function_config.dict())
        else:
            raise NotImplementedError("Invalid config")
