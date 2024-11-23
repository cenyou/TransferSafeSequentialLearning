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
from tssl.configs.acquisition_function import (
    BasicRandomConfig,
    BasicSafeRandomConfig,
    BasicPredVarianceConfig,
    BasicPredEntropyConfig,
    BasicPredSigmaConfig,
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
)
from tssl.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig
from tssl.configs.kernels.matern32_configs import BasicMatern32Config, Matern32WithPriorConfig
from tssl.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from tssl.configs.models.gp_model_config import (
    BasicGPModelConfig,
    GPModelExtenseOptimization,
    GPModelFastConfig,
    GPModelFixedNoiseConfig,
    GPModelSmallPertubationConfig,
    GPModelWithNoisePriorConfig,
)
from tssl.configs.kernels.multi_output_kernels.coregionalization_kernel_configs import (
    BasicCoregionalizationSOConfig,
    CoregionalizationSOWithPriorConfig,
)
from tssl.configs.kernels.multi_output_kernels.coregionalization_Platent_kernel_configs import (
    BasicCoregionalizationPLConfig,
    CoregionalizationPLWithPriorConfig
)
from tssl.configs.kernels.multi_output_kernels.multi_source_additive_kernel_configs import (
    BasicMIAdditiveConfig,
    MIAdditiveWithPriorConfig,
)
from tssl.configs.kernels.multi_output_kernels.fpacoh_kernel_config import (
    BasicFPACOHKernelConfig
)
from tssl.configs.models.metagp_model_config import BasicMetaGPModelConfig
from tssl.configs.models.mogp_model_so_config import BasicSOMOGPModelConfig
from tssl.configs.models.mogp_model_transfer_config import BasicTransferGPModelConfig
from tssl.configs.models.gp_model_for_engine1_config import (
    Engine1GPModelBEConfig,
    Engine1GPModelTExConfig,
    Engine1GPModelPI0vConfig,
    Engine1GPModelPI0sConfig,
    Engine1GPModelHCConfig,
    Engine1GPModelNOxConfig,
)
from tssl.configs.models.gp_model_for_engine2_config import (
    Engine2GPModelBEConfig,
    Engine2GPModelTExConfig,
    Engine2GPModelPI0vConfig,
    Engine2GPModelPI0sConfig,
    Engine2GPModelHCConfig,
    Engine2GPModelNOxConfig,
)
from tssl.configs.experiment.simulator_configs.base_simulator_config import BaseSimulatorConfig
from tssl.configs.experiment.simulator_configs.single_task_1d_illustrate_config import SingleTaskIllustrateConfig
from tssl.configs.experiment.simulator_configs.transfer_task_1d_illustrate_config import TransferTaskIllustrateConfig
from tssl.configs.experiment.simulator_configs.single_task_branin_config import (
    SingleTaskBraninConfig,
    SingleTaskBranin0Config,
    SingleTaskBranin1Config,
    SingleTaskBranin2Config,
    SingleTaskBranin3Config,
    SingleTaskBranin4Config,
)
from tssl.configs.experiment.simulator_configs.transfer_task_branin_config import (
    TransferTaskBraninBaseConfig,
    TransferTaskBranin0Config,
    TransferTaskBranin1Config,
    TransferTaskBranin2Config,
    TransferTaskBranin3Config,
    TransferTaskBranin4Config,
)
from tssl.configs.experiment.simulator_configs.transfer_task_multi_sources_branin_config import (
    TransferTaskMultiSourcesBraninBaseConfig,
    TransferTaskMultiSourcesBranin0Config,
    TransferTaskMultiSourcesBranin1Config,
    TransferTaskMultiSourcesBranin2Config,
    TransferTaskMultiSourcesBranin3Config,
    TransferTaskMultiSourcesBranin4Config,
)
from tssl.configs.experiment.simulator_configs.single_task_hartmann3_config import (
    SingleTaskHartmann3Config, SingleTaskHartmann3_0Config, SingleTaskHartmann3_1Config, SingleTaskHartmann3_2Config, SingleTaskHartmann3_3Config, SingleTaskHartmann3_4Config,
)
from tssl.configs.experiment.simulator_configs.transfer_task_hartmann3_config import (
    TransferTaskHartmann3BaseConfig,
    TransferTaskHartmann3_0Config,
    TransferTaskHartmann3_1Config,
    TransferTaskHartmann3_2Config,
    TransferTaskHartmann3_3Config,
    TransferTaskHartmann3_4Config,
)
from tssl.configs.experiment.simulator_configs.single_task_mogp1dz_config import (
    SingleTaskMOGP1DzBaseConfig,
    SingleTaskMOGP1Dz0Config, SingleTaskMOGP1Dz1Config, SingleTaskMOGP1Dz2Config, SingleTaskMOGP1Dz3Config, SingleTaskMOGP1Dz4Config,
    SingleTaskMOGP1Dz5Config, SingleTaskMOGP1Dz6Config, SingleTaskMOGP1Dz7Config, SingleTaskMOGP1Dz8Config, SingleTaskMOGP1Dz9Config,
    SingleTaskMOGP1Dz10Config, SingleTaskMOGP1Dz11Config, SingleTaskMOGP1Dz12Config, SingleTaskMOGP1Dz13Config, SingleTaskMOGP1Dz14Config,
    SingleTaskMOGP1Dz15Config, SingleTaskMOGP1Dz16Config, SingleTaskMOGP1Dz17Config, SingleTaskMOGP1Dz18Config, SingleTaskMOGP1Dz19Config,
)
from tssl.configs.experiment.simulator_configs.transfer_task_mogp1dz_config import (
    TransferTaskMOGP1DzBaseConfig,
    TransferTaskMOGP1Dz0Config, TransferTaskMOGP1Dz1Config, TransferTaskMOGP1Dz2Config, TransferTaskMOGP1Dz3Config, TransferTaskMOGP1Dz4Config,
    TransferTaskMOGP1Dz5Config, TransferTaskMOGP1Dz6Config, TransferTaskMOGP1Dz7Config, TransferTaskMOGP1Dz8Config, TransferTaskMOGP1Dz9Config,
    TransferTaskMOGP1Dz10Config, TransferTaskMOGP1Dz11Config, TransferTaskMOGP1Dz12Config, TransferTaskMOGP1Dz13Config, TransferTaskMOGP1Dz14Config,
    TransferTaskMOGP1Dz15Config, TransferTaskMOGP1Dz16Config, TransferTaskMOGP1Dz17Config, TransferTaskMOGP1Dz18Config, TransferTaskMOGP1Dz19Config,
)

from tssl.configs.experiment.simulator_configs.single_task_mogp2dz_config import (
    SingleTaskMOGP2DzBaseConfig,
    SingleTaskMOGP2Dz0Config, SingleTaskMOGP2Dz1Config, SingleTaskMOGP2Dz2Config, SingleTaskMOGP2Dz3Config, SingleTaskMOGP2Dz4Config,
    SingleTaskMOGP2Dz5Config, SingleTaskMOGP2Dz6Config, SingleTaskMOGP2Dz7Config, SingleTaskMOGP2Dz8Config, SingleTaskMOGP2Dz9Config,
    SingleTaskMOGP2Dz10Config, SingleTaskMOGP2Dz11Config, SingleTaskMOGP2Dz12Config, SingleTaskMOGP2Dz13Config, SingleTaskMOGP2Dz14Config,
    SingleTaskMOGP2Dz15Config, SingleTaskMOGP2Dz16Config, SingleTaskMOGP2Dz17Config, SingleTaskMOGP2Dz18Config, SingleTaskMOGP2Dz19Config,
)
from tssl.configs.experiment.simulator_configs.transfer_task_mogp2dz_config import (
    TransferTaskMOGP2DzBaseConfig,
    TransferTaskMOGP2Dz0Config, TransferTaskMOGP2Dz1Config, TransferTaskMOGP2Dz2Config, TransferTaskMOGP2Dz3Config, TransferTaskMOGP2Dz4Config,
    TransferTaskMOGP2Dz5Config, TransferTaskMOGP2Dz6Config, TransferTaskMOGP2Dz7Config, TransferTaskMOGP2Dz8Config, TransferTaskMOGP2Dz9Config,
    TransferTaskMOGP2Dz10Config, TransferTaskMOGP2Dz11Config, TransferTaskMOGP2Dz12Config, TransferTaskMOGP2Dz13Config, TransferTaskMOGP2Dz14Config,
    TransferTaskMOGP2Dz15Config, TransferTaskMOGP2Dz16Config, TransferTaskMOGP2Dz17Config, TransferTaskMOGP2Dz18Config, TransferTaskMOGP2Dz19Config,
)
from tssl.configs.experiment.simulator_configs.single_task_engine_interpolated_config import (
    SingleTaskEngineInterpolatedBaseConfig,
    SingleTaskEngineInterpolated_be_Config,
    SingleTaskEngineInterpolated_TEx_Config,
    SingleTaskEngineInterpolated_PI0v_Config,
    SingleTaskEngineInterpolated_PI0s_Config,
    SingleTaskEngineInterpolated_HC_Config,
    SingleTaskEngineInterpolated_NOx_Config,
)
from tssl.configs.experiment.simulator_configs.transfer_task_engine_interpolated_config import (
    TransferTaskEngineInterpolatedBaseConfig,
    TransferTaskEngineInterpolated_be_Config,
    TransferTaskEngineInterpolated_TEx_Config,
    TransferTaskEngineInterpolated_PI0v_Config,
    TransferTaskEngineInterpolated_PI0s_Config,
    TransferTaskEngineInterpolated_HC_Config,
    TransferTaskEngineInterpolated_NOx_Config,
)
from tssl.configs.experiment.simulator_configs.single_task_gengine_config import (
    SingleTaskGEngineConfig,
    SingleTaskGEngineTestConfig,
)
from tssl.configs.experiment.simulator_configs.transfer_task_gengine_config import (
    TransferTaskGEngineConfig,
    TransferTaskGEngineTestConfig,
)


class ConfigPicker:
    models_configs_dict = {
        c.__name__: c
        for c in [
            BasicGPModelConfig,
            GPModelFastConfig,
            GPModelWithNoisePriorConfig,
            GPModelSmallPertubationConfig,
            GPModelExtenseOptimization,
            GPModelFixedNoiseConfig,
            BasicSOMOGPModelConfig,
            BasicTransferGPModelConfig,
            BasicMetaGPModelConfig,
            Engine1GPModelBEConfig,
            Engine1GPModelTExConfig,
            Engine1GPModelPI0vConfig,
            Engine1GPModelPI0sConfig,
            Engine1GPModelHCConfig,
            Engine1GPModelNOxConfig,
            Engine2GPModelBEConfig,
            Engine2GPModelTExConfig,
            Engine2GPModelPI0vConfig,
            Engine2GPModelPI0sConfig,
            Engine2GPModelHCConfig,
            Engine2GPModelNOxConfig,
        ]
    }

    kernels_configs_dict = {
        c.__name__: c
        for c in [
            BasicMatern52Config,
            Matern52WithPriorConfig,
            BasicMatern32Config,
            Matern32WithPriorConfig,
            RBFWithPriorConfig,
            BasicRBFConfig,
            BasicCoregionalizationSOConfig,
            CoregionalizationSOWithPriorConfig,
            BasicCoregionalizationPLConfig,
            CoregionalizationPLWithPriorConfig,
            BasicMIAdditiveConfig,
            MIAdditiveWithPriorConfig,
            BasicFPACOHKernelConfig,
        ]
    }

    acquisition_function_configs_dict = {
        c.__name__: c
        for c in [
            BasicRandomConfig,
            BasicSafeRandomConfig,
            BasicPredVarianceConfig,
            BasicPredSigmaConfig,
            BasicPredEntropyConfig,
            BasicSafePredEntropyConfig,
            BasicSafePredEntropyAllConfig,
        ]
    }

    experiment_simulator_configs_dict = {
        c.__name__: c
        for c in [
            BaseSimulatorConfig,
            SingleTaskIllustrateConfig,
            TransferTaskIllustrateConfig,
            SingleTaskBraninConfig,
            SingleTaskBranin0Config, SingleTaskBranin1Config, SingleTaskBranin2Config, SingleTaskBranin3Config, SingleTaskBranin4Config,
            TransferTaskBraninBaseConfig,
            TransferTaskBranin0Config, TransferTaskBranin1Config, TransferTaskBranin2Config, TransferTaskBranin3Config, TransferTaskBranin4Config,
            TransferTaskMultiSourcesBranin0Config, TransferTaskMultiSourcesBranin1Config, TransferTaskMultiSourcesBranin2Config, TransferTaskMultiSourcesBranin3Config, TransferTaskMultiSourcesBranin4Config,
            SingleTaskHartmann3Config,
            SingleTaskHartmann3_0Config, SingleTaskHartmann3_1Config, SingleTaskHartmann3_2Config, SingleTaskHartmann3_3Config, SingleTaskHartmann3_4Config,
            TransferTaskHartmann3BaseConfig,
            TransferTaskHartmann3_0Config, TransferTaskHartmann3_1Config, TransferTaskHartmann3_2Config, TransferTaskHartmann3_3Config, TransferTaskHartmann3_4Config,
            SingleTaskMOGP1DzBaseConfig,
            SingleTaskMOGP1Dz0Config, SingleTaskMOGP1Dz1Config, SingleTaskMOGP1Dz2Config, SingleTaskMOGP1Dz3Config, SingleTaskMOGP1Dz4Config,
            SingleTaskMOGP1Dz5Config, SingleTaskMOGP1Dz6Config, SingleTaskMOGP1Dz7Config, SingleTaskMOGP1Dz8Config, SingleTaskMOGP1Dz9Config,
            SingleTaskMOGP1Dz10Config, SingleTaskMOGP1Dz11Config, SingleTaskMOGP1Dz12Config, SingleTaskMOGP1Dz13Config, SingleTaskMOGP1Dz14Config,
            SingleTaskMOGP1Dz15Config, SingleTaskMOGP1Dz16Config, SingleTaskMOGP1Dz17Config, SingleTaskMOGP1Dz18Config, SingleTaskMOGP1Dz19Config,
            TransferTaskMOGP1DzBaseConfig,
            TransferTaskMOGP1Dz0Config, TransferTaskMOGP1Dz1Config, TransferTaskMOGP1Dz2Config, TransferTaskMOGP1Dz3Config, TransferTaskMOGP1Dz4Config,
            TransferTaskMOGP1Dz5Config, TransferTaskMOGP1Dz6Config, TransferTaskMOGP1Dz7Config, TransferTaskMOGP1Dz8Config, TransferTaskMOGP1Dz9Config,
            TransferTaskMOGP1Dz10Config, TransferTaskMOGP1Dz11Config, TransferTaskMOGP1Dz12Config, TransferTaskMOGP1Dz13Config, TransferTaskMOGP1Dz14Config,
            TransferTaskMOGP1Dz15Config, TransferTaskMOGP1Dz16Config, TransferTaskMOGP1Dz17Config, TransferTaskMOGP1Dz18Config, TransferTaskMOGP1Dz19Config,
            SingleTaskMOGP2DzBaseConfig,
            SingleTaskMOGP2Dz0Config, SingleTaskMOGP2Dz1Config, SingleTaskMOGP2Dz2Config, SingleTaskMOGP2Dz3Config, SingleTaskMOGP2Dz4Config,
            SingleTaskMOGP2Dz5Config, SingleTaskMOGP2Dz6Config, SingleTaskMOGP2Dz7Config, SingleTaskMOGP2Dz8Config, SingleTaskMOGP2Dz9Config,
            SingleTaskMOGP2Dz10Config, SingleTaskMOGP2Dz11Config, SingleTaskMOGP2Dz12Config, SingleTaskMOGP2Dz13Config, SingleTaskMOGP2Dz14Config,
            SingleTaskMOGP2Dz15Config, SingleTaskMOGP2Dz16Config, SingleTaskMOGP2Dz17Config, SingleTaskMOGP2Dz18Config, SingleTaskMOGP2Dz19Config,
            TransferTaskMOGP2DzBaseConfig,
            TransferTaskMOGP2Dz0Config, TransferTaskMOGP2Dz1Config, TransferTaskMOGP2Dz2Config, TransferTaskMOGP2Dz3Config, TransferTaskMOGP2Dz4Config,
            TransferTaskMOGP2Dz5Config, TransferTaskMOGP2Dz6Config, TransferTaskMOGP2Dz7Config, TransferTaskMOGP2Dz8Config, TransferTaskMOGP2Dz9Config,
            TransferTaskMOGP2Dz10Config, TransferTaskMOGP2Dz11Config, TransferTaskMOGP2Dz12Config, TransferTaskMOGP2Dz13Config, TransferTaskMOGP2Dz14Config,
            TransferTaskMOGP2Dz15Config, TransferTaskMOGP2Dz16Config, TransferTaskMOGP2Dz17Config, TransferTaskMOGP2Dz18Config, TransferTaskMOGP2Dz19Config,
            SingleTaskEngineInterpolatedBaseConfig,
            SingleTaskEngineInterpolated_be_Config,
            SingleTaskEngineInterpolated_TEx_Config,
            SingleTaskEngineInterpolated_PI0v_Config,
            SingleTaskEngineInterpolated_PI0s_Config,
            SingleTaskEngineInterpolated_HC_Config,
            SingleTaskEngineInterpolated_NOx_Config,
            TransferTaskEngineInterpolatedBaseConfig,
            TransferTaskEngineInterpolated_be_Config,
            TransferTaskEngineInterpolated_TEx_Config,
            TransferTaskEngineInterpolated_PI0v_Config,
            TransferTaskEngineInterpolated_PI0s_Config,
            TransferTaskEngineInterpolated_HC_Config,
            TransferTaskEngineInterpolated_NOx_Config,
            SingleTaskGEngineConfig,
            SingleTaskGEngineTestConfig,
            TransferTaskGEngineConfig,
            TransferTaskGEngineTestConfig,
        ]
    }

    @staticmethod
    def pick_kernel_config(config_class_name):
        return ConfigPicker.kernels_configs_dict[config_class_name]

    @staticmethod
    def pick_model_config(config_class_name):
        return ConfigPicker.models_configs_dict[config_class_name]

    @staticmethod
    def pick_acquisition_function_config(config_class_name):
        return ConfigPicker.acquisition_function_configs_dict[config_class_name]

    @staticmethod
    def pick_experiment_simulator_config(config_class_name):
        return ConfigPicker.experiment_simulator_configs_dict[config_class_name]

if __name__ == "__main__":
    pass
