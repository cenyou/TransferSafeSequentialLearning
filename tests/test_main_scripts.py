"""
// Copyright (c) 2024 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0

Authors: Cen-You Li <cen-you.li@de.bosch.com> & Matthias Bitzer <matthias.bitzer3@de.bosch.com>
"""
import pathlib
import os
import numpy as np
from tssl.oracles import BraninHoo, OracleNormalizer


def test_script_psal_from_simulator(tmp_path, capsys):
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / 'branin0').mkdir()
    data_generator = OracleNormalizer(BraninHoo())
    data_generator.set_normalization_by_sampling()
    x, y = data_generator.get_random_data(100, noisy=True)
    np.savetxt(data_path / 'branin0' / 'constants.txt', data_generator.base_oracle._constants)
    np.savetxt(data_path / 'branin0' / 'normalize.txt', np.array( data_generator.get_normalization() ))
    np.savetxt(data_path / 'branin0' / 'init_X.txt', x)
    np.savetxt(data_path / 'branin0' / 'init_Y.txt', y)
    
    EXP_OUT_PATH = tmp_path / 'PSAL'
    EXP_OUT_PATH.mkdir()
    EXP_ID = 0
    ACQ_FUNC_CONFIG = 'BasicSafePredEntropyAllConfig'
    KERNEL_CONFIG = 'BasicMatern52Config'
    MODEL_CONFOG = 'BasicGPModelConfig'
    N_STEPS = 3

    with capsys.disabled():
        os.system(
            "python -m tssl.experiments.safe_learning.main_safe_al_from_simulator --experiment_data_dir="
            + str(data_path)
            + " --experiment_output_dir="
            + str(EXP_OUT_PATH)
            + " --experiment_idx="
            + str(EXP_ID)
            + " --acquisition_function_config="
            + str(ACQ_FUNC_CONFIG)
            + " --validation_type=RMSE"
            + " --kernel_config="
            + str(KERNEL_CONFIG)
            + " --model_config="
            + str(MODEL_CONFOG)
            + " --save_results=True --query_noisy=True"
            + " --n_pool=1000 --n_data_test=200 --n_data_initial=20"
            + " --n_steps="
            + str(N_STEPS)
            + " --simulator_config=SingleTaskBranin0Config"
        )
    working_dir_for_run = EXP_OUT_PATH / f'{ACQ_FUNC_CONFIG}_{KERNEL_CONFIG}_{MODEL_CONFOG}_SingleTaskBranin0Config'
    metrics_file = working_dir_for_run / "0_SafeAL_result.csv"
    assert metrics_file.exists()

    KERNEL_CONFIG = 'BasicMIAdditiveConfig'
    MODEL_CONFOG = 'BasicSOMOGPModelConfig'
    with capsys.disabled():
        os.system(
            "python -m tssl.experiments.safe_learning.main_safe_transfer_al_from_simulator --experiment_data_dir="
            + str(data_path)
            + " --experiment_output_dir="
            + str(EXP_OUT_PATH)
            + " --experiment_idx="
            + str(EXP_ID)
            + " --acquisition_function_config="
            + str(ACQ_FUNC_CONFIG)
            + " --validation_type=RMSE"
            + " --kernel_config="
            + str(KERNEL_CONFIG)
            + " --model_config="
            + str(MODEL_CONFOG)
            + " --save_results=True --query_noisy=True"
            + " --n_pool=1000 --n_data_s 100 --n_data_test=200 --n_data_initial=30"
            + " --n_steps="
            + str(N_STEPS)
            + " --simulator_config=TransferTaskBranin0Config"
        )
    working_dir_for_run = EXP_OUT_PATH / f'{ACQ_FUNC_CONFIG}_{KERNEL_CONFIG}_{MODEL_CONFOG}_TransferTaskBranin0Config'
    metrics_file = working_dir_for_run / "0_SafeAL_result.csv"
    assert metrics_file.exists()

