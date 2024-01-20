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
from .base_oracle import BaseOracle, StandardOracle, Standard1DOracle, Standard2DOracle
from .branin_hoo import BraninHoo
from .eggholder import Eggholder
from .flexible_oracle import Flexible1DOracle, Flexible2DOracle, FlexibleOracle
from .gp_oracle_from_data import GPOracleFromData
from .mogp_oracle_1d import MOGP1DOracle
from .mogp_oracle_2d import MOGP2DOracle

__all__=[
    "BaseOracle",
    "BaseObjectOracle",
    "StandardOracle",
    "Standard1DOracle",
    "Standard2DOracle",

    "BraninHoo",
    "Eggholder",
    "Flexible1DOracle",
    "Flexible2DOracle",
    "FlexibleOracle",
    "GPOracleFromData",
    "MOGP1DOracle",
    "MOGP2DOracle"
]