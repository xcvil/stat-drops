# Copyright 2020 Xiaochen Zheng @ETHZ and Jörg Rieckermann @EAWAG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file includes necessary operations for accessing to the SQL database by Python as well as
# the format converting between .db and .csv. Most of these functions are originally developed by
# the authors otherwise the sources are mentioned.
# ==============================================================================

import pandas as pd
import numpy as np
import sqlite3


def filter_rain(path, file_in, file_out, min_intensity=1.2):

    npy_input = np.load(path+file_in, allow_pickle=True)
    n = 0
    for i in range(npy_input.shape[0]):
        if float(npy_input[i, 2]) - float(min_intensity) > 0.000000001:
            npy_input[n, 0:5] = npy_input[i, 0:5]
            npy_input[n, 5] = np.array(npy_input[i, 5].split(',')[:32])
            npy_input[n, 6] = np.array(npy_input[i, 6].split(',')[:1024])
            n += 1
    np.save(path+file_out, npy_input[0:n])
    print('-'*60)
    print('file {} save at {}'.format(file_out, path+file_out))

    return npy_input[0:n]


path = '/home/xiaochenzheng/Desktop/Master_Project/2019_Disdros_TBA/Data/'
file = 'disdrometers_basel_corrected.db'
out1 = 'rawdata_station40.npy'
out2 = 'rawdata_station41.npy'

con = sqlite3.connect(path+file)

table1 = 'station_40'
table2 = 'station_41'
query1 = 'SELECT * FROM {}'.format(table1)
query2 = 'SELECT * FROM {}'.format(table2)

sql1 = pd.read_sql(query1, con)
sql2 = pd.read_sql(query2, con)

columns = ['record_number', 'datetime', 'rain_intensity_32_bit', 'radar_reflectivity', 'temperature_in_the_sensor',
           'field_n_d_224_comma_separated_positions_matrix', 'raw_data_4096_comma_separated_positions_matrix']

temp1 = sql1[columns]
temp2 = sql2[columns]

src1 = temp1.to_numpy()
src2 = temp2.to_numpy()

# for station 40

st40 = filter_rain(path, out1, 'station40.npy')


# Get the raw matrix as np.ndarray using vstack

"""
path = '/home/xiaochenzheng/Desktop/Master_Project/2019_Disdros_TBA/Data/st40/'
event0 = np.load(path+'0.npy', allow_pickle=True)
event0_rawm = event0[:, 6] # shape = (-1,)
event0_rawm = np.vstack(event0_rawm).astype(int) # shape = (-1, 1024)
"""

