"""
// Copyright (c) 2024 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0

Authors: Cen-You Li <cen-you.li@de.bosch.com> & Matthias Bitzer <matthias.bitzer3@de.bosch.com>
"""
from typing import Union, Sequence, List
import numpy as np
from tssl.pools.pool_from_data import PoolFromData
from tssl.data_sets.base_data_set import StandardDataSet

class PoolFromDataSet(PoolFromData):
    def __init__(
        self,
        dataset: StandardDataSet,
        input_idx: List[Union[int, bool]],
        output_idx: List[Union[int, bool]]=[0],
        data_is_noisy: bool=True,
        observation_noise: float = None,
        seed:int=123,
        set_seed:bool=False
    ):
        x, y = dataset.get_complete_dataset()
        super().__init__(
            x[..., input_idx], y[..., output_idx],
            data_is_noisy=data_is_noisy,
            observation_noise=observation_noise,
            seed=seed,
            set_seed=set_seed
        )

