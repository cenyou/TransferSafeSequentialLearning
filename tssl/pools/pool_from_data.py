from typing import Union, Sequence
import numpy as np
from tssl.utils.utils import row_wise_compare, check1Dlist
from tssl.pools.base_pool import BasePool
import logging
logger = logging.getLogger(__name__)

class PoolFromData(BasePool):
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        data_is_noisy: bool=True,
        observation_noise: float = None,
        seed:int=123,
        set_seed:bool=False
        ):
        super().__init__()
        self.set_data( np.atleast_2d(x_data), np.atleast_2d(y_data) )
        self.set_data_is_noisy(data_is_noisy)
        self.observation_noise = observation_noise
        if set_seed:
            np.random.seed(seed)
    
    def set_data(self, x_data: np.ndarray, y_data: np.ndarray):
        r"""
        set x and y manually
        """
        self.__x = np.atleast_2d(x_data).copy()
        self.__y = np.atleast_2d(y_data).copy()
        
        self.__dimension = self.__x.shape[1]

        assert self.__x.shape[0] == self.__y.shape[0]

    def set_data_is_noisy(self, noisy:bool):
        self.__data_is_noisy = noisy
    
    def get_max(self):
        return max(self.__y)
        
    def get_constrained_max(
        self, 
        constraint_lower: float =-np.inf,
        constraint_upper: float = np.inf
        ):
        
        max_unconst = self.get_max()
        assert max_unconst >= constraint_lower
        return min(max_unconst, constraint_upper)

    def query(self, x, noisy: bool=True):
        if np.shape(np.atleast_2d(x))[0] > 1:
            raise ValueError("please query only 1 point")
        
        idx = row_wise_compare(self.__x, x)
        idx = np.where(idx)[0] # np.where return a tuple, idx here is an array
        
        if len(idx) < 1 and not self._query_non_exist_points:
            raise ValueError('queried point does not exist')
        if len(idx) < 1 and self._query_non_exist_points:
            raise NotImplementedError('queried point does not exist')
        elif len(idx) > 1:
            idx = np.random.choice(idx)
        else:
            idx = idx[0]

        y = self.__y[idx, 0]
        
        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return y
        else:
            return y

    def get_grid_data(self, n_per_dim: int, noisy: bool):
        """
        input: n_per_dim noisy
        """
        raise NotImplementedError
    
    def get_data_from_idx(self, idx: Sequence[int], noisy: bool):

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y

    def get_random_data(self, n: int, noisy: bool):
        N_pool = self.__x.shape[0]
        if n > N_pool:
            logger.warning("not enough data in the pool")
            logger.warning(f"return all data in the pool (N_pool = {N_pool})")
        idx = np.random.choice(N_pool, min(n, N_pool), replace = self._with_replacement)

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y
    
    def get_random_data_in_box(
        self, n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy: bool=False
    ):
        N_pool = self.__x.shape[0]
        if n > N_pool:
            logger.warning("not enough data in the pool")
            logger.warning(f"return all data in the pool (N_pool = {N_pool})")

        a = np.array(check1Dlist(a, self.get_dimension()))
        box_width = np.array(check1Dlist(box_width, self.get_dimension()))
        b = a+box_width
        
        mask = np.all(self.__x >= a, axis=1) * np.all(self.__x <= b, axis=1)
        if mask.sum() < min(n, N_pool):
            raise ValueError("does not have enough data within the specified box")
        
        idx = np.where(mask)[0]
        idx = np.random.choice(idx, min(n, N_pool), replace= self._with_replacement)

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y

    def get_random_constrained_data(
        self, n : int,
        noisy : bool=False,
        constraint_lower: float =-np.inf,
        constraint_upper: float = np.inf
        ):
        N_pool = self.__x.shape[0]
        if n > N_pool:
            logger.warning("not enough data in the pool")
            logger.warning(f"return all data in the pool (N_pool = {N_pool})")

        mask = np.all(self.__y >= constraint_lower, axis=1) * np.all(self.__y <= constraint_upper, axis=1)
        if mask.sum() < min(n, N_pool):
            raise ValueError("does not have enough data under constraint")
        
        idx = np.where(mask)[0]
        idx = np.random.choice(idx, min(n, N_pool), replace= self._with_replacement)

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y
    
    def get_random_constrained_data_in_box(
        self, n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy : bool=False,
        constraint_lower: float =-np.inf,
        constraint_upper: float = np.inf
        ):
        N_pool = self.__x.shape[0]
        if n > N_pool:
            logger.warning("not enough data in the pool")
            logger.warning(f"return all data in the pool (N_pool = {N_pool})")

        a = np.array(check1Dlist(a, self.get_dimension()))
        box_width = np.array(check1Dlist(box_width, self.get_dimension()))
        b = a+box_width
        
        mask_x = np.all(self.__x >= a, axis=1) * np.all(self.__x <= b, axis=1)
        mask_y = np.all(self.__y >= constraint_lower, axis=1) * np.all(self.__y <= constraint_upper, axis=1)
        mask = mask_x * mask_y
        if mask.sum() < min(n, N_pool):
            raise ValueError("does not have enough data which are in the specified box and satisfy the constraint")
        
        idx = np.where(mask)[0]
        idx = np.random.choice(idx, min(n, N_pool), replace= self._with_replacement)

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y

    def get_box_bounds(self):
        a = self.__x.min(axis=0)
        b = self.__x.max(axis=0)
        return a, b
    
    def get_dimension(self, *args, **kwargs):
        return self.__dimension

    def generate_gaussian_noise(self, size: Sequence[int]):
        return np.random.normal(0, self.observation_noise, size=size) 

    def get_full_data(self):
        return self.__x.copy(), self.__y.copy()

    def possible_queries(self):
        return self.__x.copy()

