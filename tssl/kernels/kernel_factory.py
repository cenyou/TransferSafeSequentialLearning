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
from tssl.configs.kernels.base_kernel_config import BaseKernelConfig
from tssl.configs.kernels.matern52_configs import BasicMatern52Config
from tssl.configs.kernels.matern32_configs import BasicMatern32Config
from tssl.configs.kernels.rbf_configs import BasicRBFConfig
from tssl.kernels.matern52_kernel import Matern52Kernel
from tssl.kernels.matern32_kernel import Matern32Kernel
from tssl.kernels.rbf_kernel import RBFKernel
from tssl.configs.kernels.multi_output_kernels.coregionalization_Platent_kernel_configs import BasicCoregionalizationPLConfig
from tssl.kernels.multi_output_kernels.coregionalization_Platent_kernel import CoregionalizationPLKernel
from tssl.configs.kernels.multi_output_kernels.multi_source_additive_kernel_configs import BasicMIAdditiveConfig
from tssl.kernels.multi_output_kernels.multi_source_additive_kernel import MIAdditiveKernel
from tssl.configs.kernels.multi_output_kernels.fpacoh_kernel_config import BasicFPACOHKernelConfig
from tssl.kernels.multi_output_kernels.fpacoh_kernel import FPACOHKernel

class KernelFactory:
    @staticmethod
    def build(kernel_config: BaseKernelConfig):

        if isinstance(kernel_config, BasicMatern52Config):
            kernel = Matern52Kernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicMatern32Config):
            kernel = Matern32Kernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicRBFConfig):
            kernel = RBFKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicCoregionalizationPLConfig):
            kernel = CoregionalizationPLKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicMIAdditiveConfig):
            kernel = MIAdditiveKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicFPACOHKernelConfig):
            kernel = FPACOHKernel(**kernel_config.dict())
            return kernel
        else:
            raise NotImplementedError("Invalid config")


if __name__ == "__main__":
    pass