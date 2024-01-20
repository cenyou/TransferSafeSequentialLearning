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

from copy import deepcopy
from gpflow import kernels
from tssl.kernels.kernel_factory import KernelFactory

from tssl.configs.models.base_model_config import BaseModelConfig
from tssl.configs.models.gp_model_config import BasicGPModelConfig
from tssl.models.gp_model import GPModel
from tssl.configs.models.mogp_model_so_config import BasicSOMOGPModelConfig
from tssl.models.mogp_model_so import SOMOGPModel
from tssl.configs.models.mogp_model_transfer_config import BasicTransferGPModelConfig
from tssl.models.mogp_model_transfer import TransferGPModel
from tssl.configs.models.metagp_model_config import BasicMetaGPModelConfig
from tssl.models.metagp_model import MetaGPModel
from tssl.configs.models.gp_model_for_engine1_config import Engine1GPModelConfig
from tssl.configs.models.gp_model_for_engine2_config import Engine2GPModelConfig
import gpflow
import numpy as np

"""
The code was adapted from
(copyright also to Bosch under GNU Affero General Public License)
https://github.com/boschresearch/bosot/blob/main/bosot/models/gp_model.py
"""

class ModelFactory:
    @staticmethod
    def build(model_config: BaseModelConfig):
        if isinstance(model_config, BasicGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicSOMOGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = SOMOGPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicTransferGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = TransferGPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicMetaGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = MetaGPModel(kernel=kernel, **model_config.dict())
            return model
        else:
            raise NotImplementedError(f"Invalid config: {model_config.__class__.__name__}")


if __name__ == "__main__":
    pass