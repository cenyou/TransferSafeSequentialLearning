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
from copy import deepcopy
import numpy as np
import os
import sys
import glob

from tssl.configs.experiment.simulator_configs.base_simulator_config import BaseSimulatorConfig
from tssl.configs.experiment.simulator_configs.single_task_1d_illustrate_config import SingleTaskIllustrateConfig
from tssl.configs.experiment.simulator_configs.single_task_branin_config import SingleTaskBraninConfig
from tssl.configs.experiment.simulator_configs.single_task_hartmann3_config import SingleTaskHartmann3Config
from tssl.configs.experiment.simulator_configs.single_task_mogp1dz_config import SingleTaskMOGP1DzBaseConfig
from tssl.configs.experiment.simulator_configs.single_task_mogp2dz_config import SingleTaskMOGP2DzBaseConfig
from tssl.configs.experiment.simulator_configs.single_task_engine_interpolated_config import SingleTaskEngineInterpolatedBaseConfig
from tssl.configs.experiment.simulator_configs.transfer_task_branin_config import TransferTaskBraninBaseConfig
from tssl.configs.experiment.simulator_configs.transfer_task_multi_sources_branin_config import TransferTaskMultiSourcesBraninBaseConfig
from tssl.configs.experiment.simulator_configs.transfer_task_hartmann3_config import TransferTaskHartmann3BaseConfig
from tssl.configs.experiment.simulator_configs.transfer_task_1d_illustrate_config import TransferTaskIllustrateConfig
from tssl.configs.experiment.simulator_configs.transfer_task_mogp1dz_config import TransferTaskMOGP1DzBaseConfig
from tssl.configs.experiment.simulator_configs.transfer_task_mogp2dz_config import TransferTaskMOGP2DzBaseConfig
from tssl.configs.experiment.simulator_configs.transfer_task_engine_interpolated_config import TransferTaskEngineInterpolatedBaseConfig

from tssl.oracles import (
    MOGP1DOracle,
    MOGP2DOracle,
    BraninHoo,
    Flexible1DOracle, Flexible2DOracle, FlexibleOracle,
    Hartmann3,
    OracleNormalizer
)

from tssl.oracles.flexible_example_functions import (
    f_s,
    f
)

from tssl.pools import (
    BasePool, BasePoolWithSafety,
    PoolFromOracle, PoolWithSafetyFromOracle,
    TransferPoolFromPools, MultitaskPoolFromPools,
    EnginePool, EngineCorrelatedPool
)

def oracle_type2path(oracle_type: str):
    otl = oracle_type.lower()
    if otl in [f'mogp1dz{i}' for i in range(100)]:
        return os.path.join('mogp1d_z', otl.split('dz')[-1] )
    elif otl in [f'mogp1dz{i}y' for i in range(100)]:
        return os.path.join('mogp1d_z', otl.split('dz')[-1].split('y')[0])
    elif otl in [f'mogp1dz{i}z' for i in range(100)]:
        return os.path.join('mogp1d_z', otl.split('dz')[-1].split('z')[0])
    elif otl in [f'mogp2dz{i}' for i in range(100)]:
        return os.path.join('mogp2d_z', otl.split('dz')[-1])
    elif otl in [f'mogp2dz{i}y' for i in range(100)]:
        return os.path.join('mogp2d_z', otl.split('dz')[-1].split('y')[0])
    elif otl in [f'mogp2dz{i}z' for i in range(100)]:
        return os.path.join('mogp2d_z', otl.split('dz')[-1].split('z')[0])
    else:
        return otl

class OracleLoader:
    @staticmethod
    def return_oracle(oracle_type: str, observation_noise: float, data_path: str):
        otl = oracle_type.lower()
        if otl.startswith('branin'):
            if otl == 'branin':
                oracle = BraninHoo(observation_noise=observation_noise, normalize_output=True)
            else:
                try:
                    constants = np.loadtxt( os.path.join(data_path, otl, 'constants.txt') )
                    normalize_coeff = np.loadtxt( os.path.join(data_path, otl, 'normalize.txt') )
                    mean, scale = normalize_coeff
                    oracle = BraninHoo(observation_noise=observation_noise, constants=constants, normalize_output=True, normalize_mean=mean, normalize_scale=scale) # Pearson correlation: ~0.75
                except:
                    raise ValueError('oracle files not found')
            return oracle
        elif otl.startswith('multi_sources_branin'):
            oracle_list = []
            for p in glob.glob(os.path.join(data_path, otl, 'constants_*.txt')):
                source_idx = int( p.split('constants_')[-1].split('.txt')[0] )
                constants = np.loadtxt( p )
                normalize_coeff = np.loadtxt( os.path.join(data_path, otl, f'normalize_{source_idx}.txt') )
                mean, scale = normalize_coeff
                oracle = OracleNormalizer(
                    BraninHoo(observation_noise=observation_noise, constants=constants)
                )
                oracle.set_normalization_manually(mean, scale)
                oracle_list.append(oracle)
            return oracle_list
        elif otl.startswith('hartmann3'):
            if otl == 'hartmann3':
                oracle = OracleNormalizer(
                    Hartmann3(observation_noise=observation_noise)
                )
                oracle.set_normalization_manually(-0.9266432501735707, 0.9351442960268146)
                
            else:
                try:
                    constants = np.loadtxt( os.path.join(data_path, otl, 'constants.txt') )
                    normalize_coeff = np.loadtxt( os.path.join(data_path, otl, 'normalize.txt') )
                    mean, scale = normalize_coeff
                    oracle = OracleNormalizer(
                        Hartmann3(observation_noise=observation_noise, constants=constants)
                    )
                    oracle.set_normalization_manually(mean, scale)
                except:
                    raise ValueError('oracle files not found')
            return oracle
        
        elif otl.startswith('mogp1dz'):
            if otl in [f'mogp1dz{i}y' for i in range(100)]:
                try:
                    oracle = MOGP1DOracle(None, None, observation_noise, 0)
                    oracle.initialize_from_txt( os.path.join(data_path, oracle_type2path(otl)) )
                except:
                    raise ValueError('oracle files not found')
            elif otl in [f'mogp1dz{i}z' for i in range(100)]:
                try:
                    oracle = MOGP1DOracle(None, None, observation_noise, 0)
                    oracle.initialize_from_txt( os.path.join(data_path, oracle_type2path(otl)), output_name='Z0.txt' )
                except:
                    raise ValueError('oracle files not found')
            return oracle
        elif otl.startswith('mogp2dz'):
            if otl in [f'mogp2dz{i}y' for i in range(100)]:
                try:
                    oracle = MOGP2DOracle(None, None, observation_noise, 0)
                    oracle.initialize_from_txt( os.path.join(data_path, oracle_type2path(otl)) )
                except:
                    raise ValueError('oracle files not found')
            elif otl in [f'mogp2dz{i}z' for i in range(100)]:
                try:
                    oracle = MOGP2DOracle(None, None, observation_noise, 0)
                    oracle.initialize_from_txt( os.path.join(data_path, oracle_type2path(otl)), output_name='Z0.txt' )
                except:
                    raise ValueError('oracle files not found')
            return oracle
        elif otl.startswith('illustrate'):
            if otl == 'illustrate_s':
                oracle = Flexible1DOracle(observation_noise)
                oracle.set_f(f_s)
                return oracle
            elif otl == 'illustrate':
                oracle = Flexible1DOracle(observation_noise)
                oracle.set_f(f)
                return oracle
        raise ValueError('oracle type not found')


class SimulatorFactory:
    @staticmethod
    def build(simulator_config: BaseSimulatorConfig):
        r"""
        return
            pool: BasePool or BasePoolWithSafety
        """
        if isinstance(simulator_config, SingleTaskIllustrateConfig):
            oracle = OracleLoader.return_oracle(oracle_type = 'illustrate', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)

            if not simulator_config.additional_safety:
                pool = PoolFromOracle(oracle, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle = deepcopy(oracle)
                pool = PoolWithSafetyFromOracle(oracle, safety_oracle, seed = simulator_config.seed, set_seed=True)

            pool.discretize_random(simulator_config.n_pool)
            
            return pool

        elif isinstance(simulator_config, SingleTaskBraninConfig):
            oracle = OracleLoader.return_oracle(oracle_type = 'branin', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)

            if not simulator_config.additional_safety:
                pool = PoolFromOracle(oracle, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle = deepcopy(oracle)
                pool = PoolWithSafetyFromOracle(oracle, safety_oracle, seed = simulator_config.seed, set_seed=True)

            pool.discretize_random(simulator_config.n_pool)
            
            return pool

        elif isinstance(simulator_config, TransferTaskBraninBaseConfig):
            oracle_s = OracleLoader.return_oracle(oracle_type = simulator_config.oracle_type.lower(), observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
            oracle_t = OracleLoader.return_oracle(oracle_type = 'branin', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)

            if not simulator_config.additional_safety:
                pool_s = PoolFromOracle(oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolFromOracle(oracle_t, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle_s = deepcopy(oracle_s)
                safety_oracle_t = deepcopy(oracle_t)
                pool_s = PoolWithSafetyFromOracle(oracle_s, safety_oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolWithSafetyFromOracle(oracle_t, safety_oracle_t, seed = simulator_config.seed, set_seed=False)

            pool_s.discretize_random(simulator_config.n_pool)
            pool_t.set_data(pool_s.possible_queries())

            return TransferPoolFromPools(pool_s, pool_t)

        elif isinstance(simulator_config, TransferTaskMultiSourcesBraninBaseConfig):
            dim_s = simulator_config.dim_s
            all_source_oracle_list = OracleLoader.return_oracle(oracle_type = simulator_config.oracle_type.lower(), observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
            source_oracle_list = all_source_oracle_list[:dim_s]
            oracle_t = OracleLoader.return_oracle(oracle_type = 'branin', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)

            assert not simulator_config.additional_safety
            pool_list = [PoolFromOracle(oracle, seed = simulator_config.seed, set_seed=True) for oracle in source_oracle_list]
            pool_list.append(PoolFromOracle(oracle_t, seed = simulator_config.seed, set_seed=True))
            pool_list[-1].discretize_random(simulator_config.n_pool)
            xx = pool_list[-1].possible_queries()
            for p in pool_list[:-1]:
                p.set_data(xx)
            return MultitaskPoolFromPools(pool_list)

        elif isinstance(simulator_config, SingleTaskHartmann3Config):
            oracle = OracleLoader.return_oracle(oracle_type = 'hartmann3', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)

            if not simulator_config.additional_safety:
                pool = PoolFromOracle(oracle, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle = deepcopy(oracle)
                pool = PoolWithSafetyFromOracle(oracle, safety_oracle, seed = simulator_config.seed, set_seed=True)

            pool.discretize_random(simulator_config.n_pool)
            
            return pool

        elif isinstance(simulator_config, TransferTaskHartmann3BaseConfig):
            oracle_s = OracleLoader.return_oracle(oracle_type = simulator_config.oracle_type.lower(), observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
            oracle_t = OracleLoader.return_oracle(oracle_type = 'hartmann3', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)

            if not simulator_config.additional_safety:
                pool_s = PoolFromOracle(oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolFromOracle(oracle_t, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle_s = deepcopy(oracle_s)
                safety_oracle_t = deepcopy(oracle_t)
                pool_s = PoolWithSafetyFromOracle(oracle_s, safety_oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolWithSafetyFromOracle(oracle_t, safety_oracle_t, seed = simulator_config.seed, set_seed=False)

            pool_s.discretize_random(simulator_config.n_pool)
            pool_t.set_data(pool_s.possible_queries())

            return TransferPoolFromPools(pool_s, pool_t)

        elif isinstance(simulator_config, TransferTaskIllustrateConfig):
            oracle_s = OracleLoader.return_oracle(oracle_type = 'illustrate_s', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
            oracle_t = OracleLoader.return_oracle(oracle_type = 'illustrate', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)

            if not simulator_config.additional_safety:
                pool_s = PoolFromOracle(oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolFromOracle(oracle_t, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle_s = deepcopy(oracle_s)
                safety_oracle_t = deepcopy(oracle_t)
                pool_s = PoolWithSafetyFromOracle(oracle_s, safety_oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolWithSafetyFromOracle(oracle_t, safety_oracle_t, seed = simulator_config.seed, set_seed=False)

            pool_s.discretize_random(simulator_config.n_pool)
            pool_t.set_data(pool_s.possible_queries())

            return TransferPoolFromPools(pool_s, pool_t)

        elif isinstance(simulator_config, SingleTaskMOGP1DzBaseConfig) or \
            isinstance(simulator_config, SingleTaskMOGP2DzBaseConfig):

            oracle = OracleLoader.return_oracle(oracle_type = simulator_config.oracle_type.lower()+'y', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
            oracle.set_specified_output(1)

            if not simulator_config.additional_safety:
                pool = PoolFromOracle(oracle, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle = OracleLoader.return_oracle(oracle_type = simulator_config.oracle_type.lower()+'z', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
                safety_oracle.set_specified_output(1)
                pool = PoolWithSafetyFromOracle(oracle, safety_oracle, seed = simulator_config.seed, set_seed=True)

            pool.discretize_random(simulator_config.n_pool)
            
            return pool

        elif isinstance(simulator_config, TransferTaskMOGP1DzBaseConfig) or \
            isinstance(simulator_config, TransferTaskMOGP2DzBaseConfig):

            oracle_s = OracleLoader.return_oracle(oracle_type = simulator_config.oracle_type.lower()+'y', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
            oracle_t = deepcopy(oracle_s)
            oracle_t.set_specified_output(1)

            if not simulator_config.additional_safety:
                pool_s = PoolFromOracle(oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolFromOracle(oracle_t, seed = simulator_config.seed, set_seed=True)
            else:
                safety_oracle_s = OracleLoader.return_oracle(oracle_type = simulator_config.oracle_type.lower()+'z', observation_noise = simulator_config.observation_noise, data_path = simulator_config.data_path)
                safety_oracle_t = deepcopy(safety_oracle_s)
                safety_oracle_t.set_specified_output(1)
                pool_s = PoolWithSafetyFromOracle(oracle_s, safety_oracle_s, seed = simulator_config.seed, set_seed=True)
                pool_t = PoolWithSafetyFromOracle(oracle_t, safety_oracle_t, seed = simulator_config.seed, set_seed=False)

            pool_s.discretize_random(simulator_config.n_pool)
            pool_t.set_data(pool_s.possible_queries())

            return TransferPoolFromPools(pool_s, pool_t)

        elif isinstance(simulator_config, SingleTaskEngineInterpolatedBaseConfig):
            flip_function = (np.squeeze(simulator_config.output_idx)==0)
            dsl = simulator_config.data_set.lower()
            if dsl == 'engine_oracle1':
                engine1or2 = 1
            elif dsl == 'engine_oracle2':
                engine1or2 = 2
            else:
                raise ValueError
            try:
                pool = EnginePool(
                    os.path.join(simulator_config.data_path, 'engine'),
                    engine1or2 = engine1or2,
                    input_idx = simulator_config.input_idx,
                    output_idx = simulator_config.output_idx,
                    safety_idx = simulator_config.safety_idx,
                    context_idx = [2,3],
                    context_values = [0.0, 0.0],
                    seed = simulator_config.seed,
                    set_seed = True,
                    flip_output = flip_function
                )
                pool.discretize_random(simulator_config.n_pool)
            except:
                raise ValueError("data set not found, please check the path again")
            return pool

        elif isinstance(simulator_config, TransferTaskEngineInterpolatedBaseConfig):
            flip_function = (np.squeeze(simulator_config.output_idx)==0)
            try:
                pool_s = EngineCorrelatedPool(
                    os.path.join(simulator_config.data_path, 'engine'),
                    engine1or2 = 1,
                    input_idx = simulator_config.input_idx,
                    output_idx = simulator_config.output_idx,
                    safety_idx = simulator_config.safety_idx,
                    context_idx = [2,3],
                    context_values=[0.0, 0.0],
                    seed = simulator_config.seed,
                    set_seed = True,
                    flip_output = flip_function
                )
                pool_t = EngineCorrelatedPool(
                    os.path.join(simulator_config.data_path, 'engine'),
                    engine1or2 = 2,
                    input_idx = simulator_config.input_idx,
                    output_idx = simulator_config.output_idx,
                    safety_idx = simulator_config.safety_idx,
                    context_idx = [2,3],
                    context_values=[0.0, 0.0],
                    seed = simulator_config.seed,
                    set_seed = True,
                    flip_output = flip_function
                )
                
                a, b = pool_t.oracle.get_contextual_box_bound()
                pool_s.discretize_random_in_box(simulator_config.n_pool, a, b-a)
                pool_t.set_data(pool_s.possible_queries())
            except:
                raise ValueError("data set not found, please check the path again")
            
            return TransferPoolFromPools(pool_s, pool_t)

        else:
            raise NotImplementedError(f"Invalid config: {oracle_config.__class__.__name__}")


if __name__ == "__main__":
    pass