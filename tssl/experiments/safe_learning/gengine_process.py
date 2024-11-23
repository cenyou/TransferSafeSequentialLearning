"""
code taken (with slight modification) from
https://github.com/boschresearch/SALMOGP/blob/master/0_load_csv_addNXstructures.py
"""

#import glob
import argparse
import csv
import json
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path, PosixPath


def parse_args():
    parser = argparse.ArgumentParser(description="Process: GEngine data")
    parser.add_argument("--experiment_data_dir", default='./experiments/data', type=str)
    args = parser.parse_args()
    return args

### vvvv DO NOT CHANGE BELOW vvvv ###
NX_pos_str = ['r0c0', 'r1c0', 'r1c1', 'r1c2', 'r1c3', 'r2c1', 'r3c1', 'r3c3', 'r4c0', 'r4c2', 'r4c3', 'r7c1', 'r8c2'] # for HC & O2
measurement_id = {
    (1, 'train'): [10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,35,36,37,38,39],
    (1, 'test'): [53,54,55,56,57,58,59,60,61,62,63,64,65],
    (2, 'train'): [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    (2, 'test'): [16,17,18,19,20,21]
}
columns = {
    (1, 'X'): [ "Speed", "Load", "Lambda", "Ign_Act_ECU", "Fuel_Cutoff" ],
    (1, 'Y'): [ "PAR", "CO", "CO2", "HC", "NOx", "O2", "T_EXM", "T_CAT1" ],
    (2, 'X'): [ "Speed", "Load", "Lambda", "Ign_Act_ECU" ],
    (2, 'Y'): [ "PAR", "HC", "NOx", "T_EXM", "T_CAT1" ],
}

pt1s = {
    (1, 'X'): [298, 38, 18, 32, 28],
    (1, 'Y'): [1]*8,
    (2, 'X'): [298, 38, 18, 32],
    (2, 'Y'): [1]*5
}
### ^^^^ DO NOT CHANGE ABOVE ^^^^ ###


def NX_matrix_transformation(matrix_strings, max_col = 5):
    r"""
    input list = [..., 'rXcY', ...], where X, Y are int
    return binary matrix with matrix[X, Y]=1, 0 elsewhere
    """
    if matrix_strings is None or matrix_strings is [None]:
        return np.ones([1, max_col], dtype=int)
    
    row = []
    col = []
    for string in matrix_strings:
        _, string = string.split('r')
        ind1, ind2 = string.split('c')
        
        row.append(int(ind1))
        col.append(int(ind2))
    
    NX_matrix = np.zeros([max(row) + 1, max_col], dtype=int)
    
    for i in range(len(row)):
        NX_matrix[row[i], col[i]] = 1
    
    return NX_matrix

def give_U_name(X_name, NX_matrix):
    U_name = []
    
    for i in range(len(X_name)):
        t = -np.arange(NX_matrix.shape[0])[NX_matrix[:, i]==1]
        for tt in t:
            if tt == 0:
                U_name.append(X_name[i]+", t")
            else:
                U_name.append(X_name[i]+", t"+str(tt))
    
    return U_name

def add_NX_structure(data, NX_matrix):
    r"""
    input
    data: [N, D] array
    NX_matrix: [T, D] array, element = 0 or 1
    
    output: [N, P] array, P = sum(NX_matrix)
    """
    steps = np.shape(NX_matrix)[0] - 1
    U = np.empty([data.shape[0]-steps, 0])
    for j in range(NX_matrix.shape[1]):
        t = -np.arange(steps+1)[NX_matrix[:, j]==1]
        for tt in t:
            if tt==0:
                U = np.hstack((U, data[steps:, j, None]))
            else:
                U = np.hstack((U, data[steps+tt:tt, j, None]))
            
    return U

def pt1_filter(x, pt1, axis=-1):
    r"""
    this is a filter with b = [1/pt1], a = [1, 1/pt1 - 1]
    a[0]y[n] = b[0]x[n] - a[1]y[n-1]
    
    see scipy.signal.lfilter for more info
    """
    b = np.array([1/pt1])
    a = np.array([1, 1/pt1 - 1])
    output = signal.lfilter(b, a, x, axis=axis)
    return output



def main(data_dir, used_measurement, X_name, Y_name, NX_matrix_inputs, pt1_inputs, pt1_outputs, store_data: bool=False, store_name: PosixPath=None):
    
    NX_matrix = NX_matrix_transformation(NX_matrix_inputs, len(X_name))
    U_name = give_U_name(X_name, NX_matrix)
    
    X = np.empty([0, len(X_name)])
    filt_X = np.empty([0, len(X_name)])
    U = np.empty([0, len(U_name)])
    filt_U = np.empty([0, len(U_name)])
    Y = np.empty([0, len(Y_name)])
    filt_Y = np.empty([0, len(Y_name)])
    
    for m_idx, m_value in enumerate(used_measurement):
        # first load data from a file
        data = np.empty([0,len(X_name)+len(Y_name)])
        file = data_dir / ("raw_0000%02d_measurement_000000.csv"%(m_value) )
        print("### measurement %3d / %s"%(m_value, str(used_measurement[m_idx:])))
        
        with open(file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            print("###    loading", end='')
            
            for row in csvreader:
                data = np.vstack((data,row))
        
        print(", filtering", end='')
        filt_data = np.empty(np.shape(data))
        for i, pt1 in enumerate(np.append(pt1_inputs, pt1_outputs)):
            filt_data[:, i] = pt1_filter(data[:, i], pt1)
        
        print(", adding NX structure", end='')
        uu = add_NX_structure(data, NX_matrix)
        filt_uu = add_NX_structure(filt_data, NX_matrix)
        
        print(", finishing\n")
        steps = NX_matrix.shape[0] - 1
        
        X = np.vstack((X, data[steps:, :len(X_name)]))
        filt_X = np.vstack((filt_X, filt_data[steps:, :len(X_name)]))
        U = np.vstack((U, uu))
        filt_U = np.vstack((filt_U, filt_uu))
        Y = np.vstack((Y, data[steps:, len(X_name):]))
        filt_Y = np.vstack((filt_Y, filt_data[steps:, len(X_name):]))
    
    # save result
    if store_data:
        print("### storing processed data")
        with pd.ExcelWriter(store_name, mode='w') as writer:
            print("###    X_raw", end='')
            pd.DataFrame(X, columns=X_name).to_excel(writer, sheet_name='X_raw')
            print(", X_processed", end='')
            pd.DataFrame(filt_X, columns=X_name).to_excel(writer, sheet_name='X_processed')
            print(", U_raw", end='')
            pd.DataFrame(U, columns=U_name).to_excel(writer, sheet_name='U_raw')
            print(", U_processed", end='')
            pd.DataFrame(filt_U, columns=U_name).to_excel(writer, sheet_name='U_processed')
            print(", Y_raw", end='')
            pd.DataFrame(Y, columns=Y_name).to_excel(writer, sheet_name='Y_raw')
            print(", Y_processed")
            pd.DataFrame(filt_Y, columns=Y_name).to_excel(writer, sheet_name='Y_processed')


    return True


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.experiment_data_dir)
    NX_matrix_inputs = NX_pos_str
    
    for engine_idx in [1, 2]:
        X_name = columns[(engine_idx, 'X')]
        Y_name = columns[(engine_idx, 'Y')]
        pt1_inputs = np.array(pt1s[(engine_idx, 'X')])
        pt1_outputs = np.array(pt1s[(engine_idx, 'Y')])

        with open(data_dir / f'gengine{engine_idx}_config.json', 'w') as fp:
            json.dump({"NX_matrix": NX_matrix_inputs, "pt1_inputs": pt1_inputs.tolist(), "pt1_outputs": pt1_outputs.tolist()}, fp)

        # training data
        used_measurement = measurement_id[(engine_idx, 'train')]
        main(
            data_dir / f'gengine{engine_idx}',
            used_measurement,
            X_name,
            Y_name,
            NX_matrix_inputs,
            pt1_inputs,
            pt1_outputs,
            store_data= True,
            store_name= data_dir / f'gengine{engine_idx}_data_training.xlsx'
        )

        # test data
        used_measurement = measurement_id[(engine_idx, 'test')]
        main(
            data_dir / f'gengine{engine_idx}',
            used_measurement,
            X_name,
            Y_name,
            NX_matrix_inputs,
            pt1_inputs,
            pt1_outputs,
            store_data= True,
            store_name= data_dir / f'gengine{engine_idx}_data_test.xlsx'
        )








