# Transfer Safe Sequential Learning (TSSL)

This repo contains classes and scripts to actively learn Gaussian Processes under safety critical scenario.
This is attach to paper Global Safe Sequential Learning via Efficient Knowledge Transfer, which will be published on arXiv later.

Please reach out Cen-You.Li@de.bosch.com for any further information.

# Setup
We use conda to manage our environment ([see here for conda environment management](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)).

Install python and the required packages:
```bash
conda env create --file environment.yml
conda activate tssl
pip install -e .
```

Our test files cover all important functions/classes of the repo. Run pytest to see if the environment is set up correctly:

```bash
conda activate tssl
pytest
```

If you see the following error message
```bash
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```
try to remove ```path\to\user\.conda\envs\tssl\Lib\site-packages\torch\lib\libiomp5md.dll```

# Paper experiments

In the following, we describe how to reproduce the experiments in our paper.

## Simulated datasets
We first generate the synthetic data, and then run experiments on the generated data.
### Generate datasets:

```bash
python ./tssl/experiments/simulator/data_generator_safe_mogp.py --folder YOUR_DATA_PATH --dim 1
python ./tssl/experiments/simulator/data_generator_safe_mogp.py --folder YOUR_DATA_PATH --dim 2
python ./tssl/experiments/simulator/data_generator_safe_branin.py --folder YOUR_DATA_PATH
```

Now GP1D, GP2D and Branin data are generated.

### Run SAL
```bash
python ./tssl/experiments/safe_learning/main_safe_al_from_simulator.py \
 --experiment_data_dir YOUR_DATA_PATH \
 --experiment_output_dir YOUR_RESULT_PATH \
 --experiment_idx 1 \
 --simulator_config DESCRIBED_BELOW \
 --acquisition_function_config BasicSafePredEntropyAllConfig --label_safeland True --safe_lower 0 --safe_upper inf --n_data_initial 10 --n_steps 50 --n_data_test 200 --query_noisy True
```
`--experiment_idx` is the seed, we use [1-5].
`--simulator_config` specifies the dataset.

For **GP1D** data: use ` --simulator_config SingleTaskMOGP1Dz[0-19]Config`.

For **GP2D** data: use ` --simulator_config SingleTaskMOGP2Dz[0-19]Config`.

For **Branin** data: use ` --simulator_config SingleTaskBranin[0-4]Config`.

### Run EffTrans

```bash
python ./tssl/experiments/safe_learning/main_safe_transfer_al_from_simulator.py \
 --experiment_data_dir YOUR_DATA_PATH \
 --experiment_output_dir YOUR_RESULT_PATH \
 --experiment_idx 1 \
 --simulator_config DESCRIBED_BELOW \
 --kernel_config BasicMIAdditiveConfig --model_config BasicTransferGPModelConfig --acquisition_function_config BasicSafePredEntropyAllConfig --label_safeland True --safe_lower_source 0 --safe_upper_source inf --safe_lower 0 --safe_upper inf --n_data_s 100 --n_data_initial 10 --n_steps 50 --n_data_test 200 --query_noisy True
```
 `--experiment_idx` is the seed, we use [1-5].
`--simulator_config` specifies the dataset.

For **GP1D** data: use ` --simulator_config TransferTaskMOGP1Dz[0-19]Config`.

For **GP2D** data: use ` --simulator_config TransferTaskMOGP2Dz[0-19]Config`.

For **Branin** data: use ` --simulator_config TransferTaskBranin[0-4]Config`.

### Run FullTransHGP:
replace `--model_config BasicTransferGPModelConfig` by `--model_config BasicSOMOGPModelConfig`, the remaining arguments are the same as `EffTrans`

### Run FullTransLMC:
replace `--kernel_config BasicMIAdditiveConfig --model_config BasicTransferGPModelConfig` by `--kernel_config BasicCoregionalizationPLConfig --model_config BasicSOMOGPModelConfig`, the remaining arguments are the same as `EffTrans`

### Run Rothfuss et al. 2022
replace `--kernel_config BasicMIAdditiveConfig --model_config BasicTransferGPModelConfig` by `--kernel_config BasicFPACOHKernelConfig --model_config BasicMetaGPModelConfig`, the remaining arguments are the same as `EffTrans`


## engine datasets
The datasets are published on github ([link](https://github.com/boschresearch/Bosch-Engine-Datasets/tree/master/pengines)).

First move all the excel files into path ```YOUR_DATA_PATH/engine```

### Run SAL
```bash
python ./tssl/experiments/safe_learning/main_safe_al_from_simulator.py \
 --experiment_data_dir YOUR_DATA_PATH \
 --experiment_output_dir YOUR_RESULT_PATH \
 --experiment_idx 1 \
 --simulator_config SingleTaskEngineInterpolated_PI0s_Config --acquisition_function_config BasicSafePredEntropyAllConfig --safe_lower -100 --safe_upper 1.0 --n_data_initial 20 --n_steps 100 --n_data_test 1000  --label_safeland True --query_noisy False

```
`--experiment_idx` is the seed, we use [1-5].
`--simulator_config SingleTaskEngineInterpolated_TEx_Config` can also be used (supplementary experiment).

### Run EffTrans
```bash
python ./tssl/experiments/safe_learning/main_safe_transfer_al_from_simulator.py \
 --experiment_data_dir YOUR_DATA_PATH \
 --experiment_output_dir YOUR_RESULT_PATH \
 --experiment_idx 1 \
 --simulator_config TransferTaskEngineInterpolated_PI0s_Config --kernel_config BasicMIAdditiveConfig --model_config BasicTransferGPModelConfig --acquisition_function_config BasicSafePredEntropyAllConfig --safe_lower_source -100 --safe_upper_source 100 --safe_lower -100 --safe_upper 1.0 --n_data_s 500 --n_data_initial 20 --n_steps 100 --n_data_test 1000  --label_safeland True --query_noisy False

```
`--experiment_idx` is the seed, we use [1-5].
`--simulator_config TransferTaskEngineInterpolated_TEx_Config` can also be used (supplementary experiment).

### Run FullTransHGP
```bash
python ./tssl/experiments/safe_learning/main_safe_transfer_al_from_simulator.py \
 --experiment_data_dir YOUR_DATA_PATH \
 --experiment_output_dir YOUR_RESULT_PATH \
 --experiment_idx 1 \
 --simulator_config TransferTaskEngineInterpolated_PI0s_Config --kernel_config BasicMIAdditiveConfig --model_config BasicSOMOGPModelConfig --acquisition_function_config BasicSafePredEntropyAllConfig --safe_lower_source -100 --safe_upper_source 100 --safe_lower -100 --safe_upper 1.0 --n_data_s 500 --n_data_initial 20 --n_steps 100 --n_data_test 1000  --label_safeland True --query_noisy False

```
`--experiment_idx` is the seed, we use [1-5].
`--simulator_config TransferTaskEngineInterpolated_TEx_Config` can also be used (supplementary experiment).

### Run FullTransLMC
```bash
python ./tssl/experiments/safe_learning/main_safe_transfer_al_from_simulator.py \
 --experiment_data_dir YOUR_DATA_PATH \
 --experiment_output_dir YOUR_RESULT_PATH \
 --experiment_idx 1 \
 --simulator_config TransferTaskEngineInterpolated_PI0s_Config --kernel_config BasicCoregionalizationPLConfig --model_config BasicSOMOGPModelConfig --acquisition_function_config BasicSafePredEntropyAllConfig --safe_lower_source -100 --safe_upper_source 100 --safe_lower -100 --safe_upper 1.0 --n_data_s 500 --n_data_initial 20 --n_steps 100 --n_data_test 1000  --label_safeland True --query_noisy False

```
`--experiment_idx` is the seed, we use [1-5].
`--simulator_config TransferTaskEngineInterpolated_TEx_Config` can also be used (supplementary experiment).

### Run Rothfuss et al. 2022
```bash
python ./tssl/experiments/safe_learning/main_safe_transfer_al_from_simulator.py \
 --experiment_data_dir YOUR_DATA_PATH \
 --experiment_output_dir YOUR_RESULT_PATH \
 --experiment_idx 1 \
 --simulator_config TransferTaskEngineInterpolated_PI0s_Config --kernel_config BasicFPACOHKernelConfig --model_config BasicMetaGPModelConfig --acquisition_function_config BasicSafePredEntropyAllConfig --safe_lower_source -100 --safe_upper_source 100 --safe_lower -100 --safe_upper 1.0 --n_data_s 500 --n_data_initial 20 --n_steps 100 --n_data_test 1000 --label_safeland True --query_noisy False

```
`--experiment_idx` is the seed, we use [1-5].
`--simulator_config TransferTaskEngineInterpolated_TEx_Config` can also be used (supplementary experiment).