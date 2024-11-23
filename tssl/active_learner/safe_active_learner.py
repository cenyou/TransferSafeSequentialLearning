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
from typing import Union, Sequence
import os
import numpy as np
import pandas as pd
import time
from tssl.utils.utils import filter_nan
from scipy.stats import norm
from tssl.utils.safety_metrices import SafetyAreaMeasure
from tssl.enums.data_structure_enums import OutputType
from tssl.enums.active_learner_enums import ValidationType
from tssl.models.base_model import BaseModel
from tssl.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function import BaseSafeAcquisitionFunction
from tssl.pools.base_pool import BasePool
from tssl.pools.base_pool_with_safety import BasePoolWithSafety


class SafeActiveLearner:

    r"""
    Main class for safe active learning

    Main Attributes:
        acquisition_function : AcquisitionFunctionType - Enum which acquisiton function type should be performed e.g. SAFE_OPT, SAFE_GP_UCB,...
        validation_type : ValidationType - Enum which validation metric should be used e.g. Simple Regret, Cumm. Regret,...
        model_is_safety_model: bool - whether the safety is constrained directly on the main model or not
        save_results: bool - whether we save the plots/result or not
        experiment_path: str - path where we save files
    """

    def __init__(
        self,
        acquisition_function: BaseSafeAcquisitionFunction,
        validation_type: ValidationType,
        query_noisy: bool=True,
        model_is_safety_model: bool=False,
        run_ccl: bool=True,
        tolerance: Union[float, Sequence[float]]=0.01,
        save_results: bool=False,
        experiment_path: str=None,
        update_gp_hps_every_n_iter: int = 1,
        max_pool_size: int=None,
    ):
        self.acquisition_function = acquisition_function
        self.validation_type = validation_type
        self.observation_number = []
        self.validation_metrics = []
        self.infer_time = []
        self.validate_time = []
        self.kernel_scale = []
        self.kernel_lengthscales = []
        self.measure_safe_area = False
        self.safe_area = SafetyAreaMeasure(run_ccl=run_ccl)
        """SafetyAreaMeasure compute the areas of predictive safe regions and true safe regions,
                where each regions would be clustered."""
        self.query_noisy = query_noisy
        self.model_is_safety_model = model_is_safety_model
        self.num_of_models = 1 + self.acquisition_function.number_of_constraints - int(self.model_is_safety_model)  # the number of f & q_i, q are safety functions
        self.tolerance = tolerance
        self.save_results = save_results
        self.exp_path = experiment_path
        self.__save_model_pars = False
        self._update_gp_hps_in_this_iter = True
        self._update_gp_hps_every_n_iter = update_gp_hps_every_n_iter
        self.max_pool_size = max_pool_size

    def set_pool(self, pool: Union[BasePool, BasePoolWithSafety]):
        self.pool = pool

    def set_model(self, model: BaseModel, safety_models: Union[BaseModel, Sequence[BaseModel]] = None):
        """
        sets surrogate model

        Arguments:
        model - BaseModel - instance of some BaseModel child
        safety_models - BaseModel or list or BaseModel

        Currently, all the models should be producing 1D array, even if we want multiple safety constraint
            (single output or flattened multi output)
        The reason is that I also use this for transfer learning, and we already need MO for source and target,
            so it would be kind of complicated if we also make different safety constraint MO.
        """
        self.model = model
        if self.model_is_safety_model:
            self.safety_models = None
            assert self.acquisition_function.number_of_constraints == 1
            print("ATTENTION - model itself is safety model")
        else:
            if isinstance(safety_models, list):
                self.safety_models = safety_models
            elif isinstance(safety_models, BaseModel):
                self.safety_models = [safety_models]
            assert self.acquisition_function.number_of_constraints == len(self.safety_models)
    
    def set_train_data(self, x_train, y_train, z_train=None):
        """
        Method for setting the train set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n training datapoints with dimension d
        y_train - np.array - array of shape [n,p] containing the outputs of n training datapoints with dimension p
        z_train - np.array - array of shape [n,q] containing the safety outputs of n training datapoints with dimension q
        """
        self.x_data = np.atleast_2d(x_train)
        self.y_data = np.atleast_2d(y_train)
        if not self.model_is_safety_model:
            self.z_data = np.atleast_2d(z_train)
    
    def add_train_data(self, x, y, z=None):
        self.x_data = np.vstack((self.x_data, np.atleast_2d(x)))
        self.y_data = np.vstack((self.y_data, np.atleast_2d(y)))
        if not z is None:
            self.z_data = np.vstack((self.z_data, np.atleast_2d(z)))
    
    def set_test_data(self, x_test, y_test, z_test=None):
        """
        Method for setting the test set manually
        Arguments:
        x_test - np.array - array of shape [n,d] containing the inputs of n training datapoints with dimension d
        y_test - np.array - array of shape [n,p] containing the outputs of n training datapoints with dimension p
        z_test - np.array - array of shape [n,q] containing the safety outputs of n training datapoints with dimension q
        """
        self.x_test = np.atleast_2d(x_test)
        self.y_test = np.atleast_2d(y_test)
        if not self.model_is_safety_model:
            self.z_test = np.atleast_2d(z_test)

    def add_test_data(self, x, y, z=None):
        self.x_test = np.vstack((self.x_test, np.atleast_2d(x)))
        self.y_test = np.vstack((self.y_test, np.atleast_2d(y)))
        if not z is None:
            self.z_test = np.vstack((self.z_test, np.atleast_2d(z)))
    
    def _set_grid_data(self, x_grid, y_grid, z_grid=None):
        """
        Method for setting the safety check set manually
        Arguments:
        x_grid - np.array - array of shape [n,d] containing the inputs of n training datapoints with dimension d
        y_grid - np.array - array of shape [n,p] containing the outputs of n training datapoints with dimension p
        z_grid - np.array - array of shape [n,q] containing the safety outputs of n training datapoints with dimension q
        """
        self.x_grid = np.atleast_2d(x_grid)
        self.y_grid = np.atleast_2d(y_grid)
        if not self.model_is_safety_model:
            self.z_grid = np.atleast_2d(z_grid)

    def initialize_safe_area_measure(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray=None,
        label_grid: str=None,
        sheet_name: str=None
    ):
        self._set_grid_data(x_grid, y_grid, z_grid)
        self.measure_safe_area = True
        
        d = self.pool.get_dimension()
        self.safe_area.set_object_detector(self.pool.get_dimension())

        if not label_grid is None:
            self.safe_area.true_safe_lands_from_file(label_grid, sheet_name)
        else:
            Z = self.y_grid if self.model_is_safety_model else self.z_grid
            safe_bool = self.acquisition_function.compute_safe_data_set(Z).reshape([-1,1])
            self.safe_area.true_safe_lands(self.x_grid[..., :d], safe_bool.astype(int))

    def update(self, converge_check: bool):
        """
        Main update function - infers the model on the current dataset, optimizes the acquisition function and returns the query location -
        Acquisition function optimization is specified in self.acquisition_function_optimization_type
        """
        self._make_infer()

        x_pool = self.pool.possible_queries()
        if not self.max_pool_size is None and self.max_pool_size > 0:
            mask = np.random.choice(x_pool.shape[0], size=min(self.max_pool_size, x_pool.shape[0]), replace=False)
            x_pool = x_pool[mask]
        idx_dim = self._return_variable_idx()

        acq_score, S = self.acquisition_function.acquisition_score(
            x_pool[:, idx_dim],
            model = self.model,
            safety_models = self.safety_models,
            x_data = self.x_data[:, idx_dim],
            y_data = self.y_data,
            return_safe_set=True
        )

        if not np.any(S):
            raise StopIteration("There are no safe points to evaluate.")
        converge = 0#np.all( std[S] <= self.tolerance)
        if converge_check and converge:
            raise StopIteration("Converge.")
        
        x_safe = x_pool[S]
        new_query = x_safe[np.argmax(acq_score[S])]

        return new_query

    def learn(self, n_steps: int):
        """
        Main maximization loop - makes n_steps queries to oracle and returns collected validation metrics and query locations

        Arguments:
        n_steps : int - number of BO steps

        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
            int - true step we finish
        """
        self.validate(make_infer=True)
        true_steps = 0
        for i in range(0, n_steps):
            try:
                max_iter = 100
                for j in range(max_iter):
                    query = self.update(converge_check=(i>=4))
                    print("Query")
                    print(query)

                    if self.measure_safe_area:
                        D = self.pool.get_variable_dimension()
                        self.safe_area.label_points(np.atleast_2d(query)[:, :D])

                    if self.model_is_safety_model:
                        new_y = self.pool.query(query, noisy=self.query_noisy)
                        if np.isnan(new_y) and j < max_iter:
                            print('Qeury is nan, repeat')
                            continue
                        self.add_train_data(query, new_y)
                        break
                    else:
                        new_y, new_z = self.pool.query(query, noisy=self.query_noisy)
                        nan_check = np.isnan(
                            np.concatenate([np.reshape(new_y, -1), np.reshape(new_z, -1)])
                        )
                        if np.all(nan_check) and j < max_iter:
                            print('Qeury is nan, repeat')
                            continue
                        elif np.any(nan_check) and j < max_iter:
                            print('Qeurying output has some nan values, be careful')
                        self.add_train_data(query, new_y, new_z)
                        break

                self.validate(make_infer=False)
                true_steps += 1
                self._update_gp_hps_in_this_iter = (true_steps % self._update_gp_hps_every_n_iter == 0 )

            except StopIteration as e:
                print(f'Finish early: {e}')
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt

        return np.array(self.validation_metrics), self.x_data, true_steps

    def validate(self, make_infer: bool=False):
        """
        validation method - calculates validation metric (self.validation_type specifies which one) and stores it to self.validation_metrics list

        Arguments:
            query : np.array - selected query
        """
        idx_dim = self._return_variable_idx()
        if make_infer:
            self._make_infer()

        print('Validate')
        self.observation_number.append(self.x_data.shape[0])

        t_start = time.perf_counter()
        
        pred_mu, pred_sigma = self.model.predictive_dist(self.x_test[:, idx_dim])
        if self.validation_type == ValidationType.RMSE:
            rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
            self.validation_metrics.append(rmse)
        elif self.validation_type == ValidationType.NEG_LOG_LIKELI:
            neg_log_likeli = np.mean(-1 * norm.logpdf(np.squeeze(self.y_test), pred_mu, pred_sigma))
            self.validation_metrics.append(neg_log_likeli)
        
        t_end = time.perf_counter()
        self.validate_time.append(t_end - t_start)

        if self.measure_safe_area:
            print('Measure safety quality')
            
            if self.model_is_safety_model:
                S = self.acquisition_function.compute_safe_set(
                    self._get_variable_input(self.x_grid),
                    [self.model]
                )
            else:
                S = self.acquisition_function.compute_safe_set(
                    self._get_variable_input(self.x_grid),
                    self.safety_models
                )
            self.safe_area.true_positive_lands(S.astype(int))
            self.safe_area.false_positive_lands(S.astype(int))

    def save_experiment_summary(self, filename="SafeAL_result.csv"):
        columns = ['iter_idx']
        columns.extend( [f'x{i}' for i in range(self.x_data.shape[1])] )
        columns.extend( ['y', 'safe_bool', self.validation_type.name.lower()] )
        columns.extend( [f'infer_time_m{i}' for i in range(self.num_of_models)])
        columns.append('validate_time')
        
        N = self.x_data.shape[0]
        iter_idx = np.empty([N, 1]) * np.nan
        metric = np.empty([N, 1]) * np.nan
        infer_time = np.empty([N, self.num_of_models]) * np.nan
        validate_time = np.empty([N, 1]) * np.nan
        for i, n in enumerate(self.observation_number):
            iter_idx[n-1] = i
            metric[n-1] = self.validation_metrics[i]
            infer_time[n-1,:] = self.infer_time[i]
            validate_time[n-1] = self.validate_time[i]

        safety_observations = self.y_data if self.model_is_safety_model else self.z_data
        safe_bool = self.acquisition_function.compute_safe_data_set(safety_observations).reshape([-1,1])

        if self.measure_safe_area:
            D = self.pool.get_variable_dimension()
            safe_area_label = np.empty([self.x_data.shape[0], 1])

            if self.pool.output_type == OutputType.SINGLE_OUTPUT:
                safe_area_label[:,0] = self.safe_area.label_points(self.x_data[:,:D])
            elif self.pool.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
                safe_area_label[self.x_data[:,-1]==self.pool.task_index,0] = self.safe_area.label_points(self.x_data[self.x_data[:,-1]==self.pool.task_index,:D])

            data = np.hstack((iter_idx, self.x_data, self.y_data, safe_bool, metric, infer_time, validate_time, safe_area_label))
            columns.append('safe_label_data')
        else:
            data = np.hstack((iter_idx, self.x_data, self.y_data, safe_bool, metric, infer_time, validate_time))

        dataframe = pd.DataFrame(data, columns=columns)
        if self.measure_safe_area:
            n = self.safe_area.get_total_iter_num_true_positive()
            safe_df = self.safe_area.export_df( dataframe[dataframe['iter_idx']>=iter_idx[-n,0]].index )
            dataframe = pd.concat([dataframe, safe_df], axis=1)

        # save model k0
        if self.__save_model_pars:
            k_scale = pd.DataFrame(
                columns=[f'model{i}' for i in self.kernel_scale[0].keys()],
                index = iter_idx[~np.isnan(iter_idx[:,0]),0]
            )
            for i, k0_dict in enumerate(self.kernel_scale):
                for key, v in k0_dict.items():
                    k_scale.loc[i, f'model{key}'] = v
        
        # then save models lengthscale
        if self.__save_model_pars:
            k_lengs = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(self.kernel_lengthscales[0].keys(), names=['model_idx','kernel_idx']),
                index = iter_idx[~np.isnan(iter_idx[:,0]),0]
            )
            for i, kl_dict in enumerate(self.kernel_lengthscales):
                for key, v in kl_dict.items():
                    k_lengs.loc[i, key] = v

        if self.save_results:
            dataframe.to_csv(os.path.join(self.exp_path, filename))
            if self.__save_model_pars:
                with pd.ExcelWriter(
                    os.path.join(self.exp_path, filename.split('.csv')[0]+'_model_parameters.xlsx'),
                    mode='w') as writer:
                    k_scale.to_excel(writer, sheet_name='kernel_scale')
                    k_lengs.to_excel(writer, sheet_name='kernel_lgths')
        else:
            if self.__save_model_pars:
                return dataframe, k_scale, k_lengs
            else:
                return dataframe

    def _make_no_train_infer(self):
        print('    No gp update in this iter')
        idx_dim = self._return_variable_idx()
        infer_time = []

        self.model.set_model_data(*filter_nan(self.x_data[..., idx_dim], self.y_data))        
        infer_time.append(0.0)
        if not self.model_is_safety_model:
            for i, model in enumerate(self.safety_models):
                model.set_model_data(*filter_nan(self.x_data[..., idx_dim], self.z_data[..., i, None]))
                infer_time.append(0.0)

        self.infer_time.append(tuple(infer_time))

        if self.__save_model_pars:
            self._track_model_parameters()

        return True

    def _make_infer(self):
        if not self._update_gp_hps_in_this_iter:
            return self._make_no_train_infer()

        print('    Update GPs')
        self.model.reset_model()
        idx_dim = self._return_variable_idx()
        infer_time = []

        t_opt = self.model.infer(*filter_nan(self.x_data[:, idx_dim], self.y_data))
        
        infer_time.append(t_opt)
        if not self.model_is_safety_model:
            for i, model in enumerate(self.safety_models):
                model.reset_model()
                
                t_opt = model.infer(*filter_nan(self.x_data[:, idx_dim], self.z_data[:, i, None]))
                
                infer_time.append(t_opt)

        self.infer_time.append(tuple(infer_time))

        if self.__save_model_pars:
            self._track_model_parameters()

    def _track_model_parameters(self):
        k0 = {0: self.model.model.kernel.prior_scale}
        if not self.model_is_safety_model:
            for j, sm in enumerate(self.safety_models):
                k0[j] = sm.model.kernel.prior_scale
        
        self.kernel_scale.append(k0)

        models = [self.model]
        if not self.model_is_safety_model:
            models.extend(self.safety_models)

        if self.pool.output_type == OutputType.SINGLE_OUTPUT:
            self.kernel_lengthscales.append({(i, 0): tuple(m.model.kernel.kernel.lengthscales.numpy(),) for i, m in enumerate(models)})
        elif self.pool.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
            kl_dict = {}
            for i, m in enumerate(models):
                for j, k in enumerate(m.model.kernel.latent_kernels):
                    kl_dict[(i, j)] = tuple(k.lengthscales.numpy())
            self.kernel_lengthscales.append(kl_dict)

    def _return_variable_idx(self):
        idx_dim = np.ones(self.x_data.shape[1], dtype=bool)
        
        dim = self.pool.get_dimension()
        var_dim = self.pool.get_variable_dimension()
        if var_dim < dim:
            _, idx = self.pool.get_context_status(return_idx=True)
            idx_dim[idx] = False
                
        return idx_dim

    def _get_variable_input(self, x: np.ndarray):
        idx_dim = self._return_variable_idx()
        return x[:, idx_dim]

    def _update_posterior(self, x: np.ndarray):
        mu = np.empty((x.shape[0], self.num_of_models), dtype=np.float)
        std = np.empty((x.shape[0], self.num_of_models), dtype=np.float)

        idx_dim = self._return_variable_idx()

        mu[:, 0], std[:, 0] = self.model.predictive_dist(x[:, idx_dim])
        for i in range(1, self.num_of_models):  # if self.model_is_safety_model, self.num_of_models will be 1 anyways
            mu[:, i], std[:, i] = self.safety_models[i - 1].predictive_dist(x[:, idx_dim])
        return mu, std


    

if __name__ == "__main__":
    pass
