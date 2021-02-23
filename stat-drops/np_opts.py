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
from sqlalchemy import create_engine
from tqdm import tqdm
import time


class np_opts(object):
    """
    Load .npy file with
    np.load('/home/xiaochenzheng/Desktop/Master_Project/2019_Disdros_TBA/Data/station40.npy', allow_pickle=True)
    """

    def __init__(self, folder, file_in, file_out=None):
        super(np_opts, self).__init__()
        self.folder = folder
        self.file_in = file_in
        self.file_out = file_out  # file name that you wanna generate, default is None

    def combine_data(self, table_name):
        if type(table_name) is not str:
            raise ValueError('table_name should be a str!')
        data0, col0 = self.pull_item(table_name)
        data1, col1 = self.pull_rain(table_name)
        data2, col2 = self.pull_raw_matrix(table_name)
        data = np.concatenate([data0, data1, data2], axis=1)
        col = col0 + col1 + col2
        np.save(self.folder + self.file_out, data)
        print('\nThe basic info of the {} data in table {}:\nshape: {} | data type: {}'.format(col, table_name,
                                                                                               data.shape,
                                                                                               data.dtype))
        print('-' * 60 + '\nFile saved at {}'.format(self.folder + self.file_out))

        return data, col

    def save_to_npy(self, query, npy_name):  # TODO not final version
        con = sqlite3.connect(self.folder + self.file_in)
        pd_sql = pd.read_sql(query, con)
        pd_np = pd_sql.to_numpy()
        np.save(self.folder + npy_name, pd_np)
        con.close()
        print('Converting finished!')

    def get_data(self, table_name):
        '''
        Different from the other func.. Did not change the data type, directly inherit from the DataFrame
        '''
        record_col = ['record_number']
        date_col = ['datetime']
        rain_intensity_col = ['rain_intensity_32_bit', 'rain_accumulated_32_bit']
        dsd_col = ['field_n_d_224_comma_separated_positions_matrix']

        con = sqlite3.connect(self.folder + self.file_in)
        query = 'SELECT * FROM {}'.format(table_name)
        db_init = pd.read_sql(query, con)
        print('Finish reading SQL......')

        # get record number, datetime, rainfall
        record_idx = db_init[record_col]
        record_idx = record_idx.to_numpy()
        print('-'*60+'\nFinish extracting {}'.format(record_col))
        print(record_idx.shape, record_idx.dtype)  # TODO dtype=int32

        datetime = db_init[date_col]
        datetime = datetime.to_numpy()
        print('-'*60+'\nFinish extracting {}'.format(date_col))
        print(datetime.shape, datetime.dtype)  # TODO dtype=object

        rain_intensity = db_init[rain_intensity_col]
        rain_intensity = rain_intensity.to_numpy()
        print('-'*60+'\nFinish extracting {}'.format(rain_intensity_col))
        print(rain_intensity.shape, rain_intensity.dtype)  # TODO dtype=object

        # split dsd
        dsd = db_init[dsd_col[0]].str.split(',', expand=True)
        dsd = dsd.to_numpy()[:, 0:-1]
        # dsd = np.array(dsd, dtype=np.float)
        print('-'*60+'\nFinish extracting {}'.format(dsd_col))
        print(dsd.shape, dsd.dtype)

        data = np.concatenate([record_idx, datetime, rain_intensity, dsd], axis=1)
        col_name = record_col + date_col + rain_intensity_col + dsd_col

        np.save(self.folder+self.file_out, data)
        print('-'*60+'\nFinish writing .npy file at {}'.format(self.folder+self.file_out))
        print('-'*60+'\nOVERVIEW\nThe shape is {} | The dtype is {}'.format(data.shape, data.dtype))

        con.close()

        return data, col_name

    def pull_dsd(self, table_name):
        '''
        column_name = 'field_n_d_224_comma_separated_positions_matrix'
        table_name = 'station_40'
        '''
        column_name = ['field_n_d_224_comma_separated_positions_matrix']
        con = sqlite3.connect(self.folder + self.file_in)

        if type(column_name[0]) is not str:
            raise ValueError('column_name should be a str!')

        query = 'SELECT * FROM {}'.format(table_name)
        pd_sql = pd.read_sql(query, con)
        temp = pd_sql[column_name[0]].str.split(',', expand=True)
        # print(temp.info())
        pd_np = temp.to_numpy()[:, 0:-1]
        pd_np = np.array(pd_np, dtype=np.float)
        print('\nThe basic info of the {} data in table {}:\nshape: {} | data type: {}'.format(column_name, table_name,
                                                                                               pd_np.shape,
                                                                                               pd_np.dtype))
        np.save(self.folder + 'dsd_' + self.file_out, pd_np)
        print('-' * 60 + '\nFile saved at {}'.format(self.folder + 'dsd_' + self.file_out))

        con.close()

        return pd_np, column_name  # return a np.ndarray and column_name list

    def pull_raw_matrix(self, table_name):
        '''
        column_name = 'raw_data_4096_comma_separated_positions_matrix'
        table_name = 'station_40'
        '''
        column_name = ['raw_data_4096_comma_separated_positions_matrix']
        con = sqlite3.connect(self.folder + self.file_in)

        if type(column_name[0]) is not str:
            raise ValueError('column_name should be a str!')

        query = 'SELECT * FROM {} LIMIT 53600, 1000'.format(table_name)
        pd_sql = pd.read_sql(query, con)
        temp = pd_sql[column_name[0]].str.split(',', expand=True)
        # print(temp.info())
        pd_np = temp.to_numpy()[:, 0:-1]
        # pd_np = np.array(pd_np, dtype=np.float)
        print('\nThe basic info of the {} data in table {}:\nshape: {} | data type: {}'.format(column_name, table_name,
                                                                                               pd_np.shape,
                                                                                               pd_np.dtype))
        np.save(self.folder + 'dsd_' + self.file_out, pd_np)
        print('-' * 60 + '\nFile saved at {}'.format(self.folder + 'raw_matrix_' + self.file_out))

        con.close()

        return pd_np, column_name  # return a np.ndarray and column_name list

    def pull_rain(self, table_name):
        '''
        column_name: choose the target column name
        table_name = 'station_40'
        '''
        con = sqlite3.connect(self.folder + self.file_in)
        column_name = ['temperature_in_the_sensor', 'rain_intensity_32_bit']

        for i in column_name:
            if type(i) is not str:
                raise ValueError('column_name should be a str!')

        query = 'SELECT * FROM {}'.format(table_name)
        pd_sql = pd.read_sql(query, con)
        # print(temp.info())
        pd_np = pd_sql[column_name]
        pd_np = pd_np.to_numpy()
        print('\nThe basic info of the {} data in table {}:\nshape: {} | data type: {}'.format(column_name, table_name,
                                                                                               pd_np.shape,
                                                                                               pd_np.dtype))
        np.save(self.folder + 'rain_intensity_' + self.file_out, pd_np)
        print('-' * 60 + '\nFile saved at {}'.format(self.folder + 'rain_intensity_' + self.file_out))

        con.close()

        return pd_np, column_name  # return a np.ndarray and the column name list

    def pull_item(self, table_name):
        '''
        column_name: choose the target column name
        table_name = 'station_40'
        '''
        con = sqlite3.connect(self.folder + self.file_in)
        column_name = ['record_number', 'datetime']

        for i in column_name:
            if type(i) is not str:
                raise ValueError('column_name should be a str!')

        query = 'SELECT * FROM {}'.format(table_name)
        pd_sql = pd.read_sql(query, con)
        # print(temp.info())
        pd_np = pd_sql[column_name]
        pd_np = pd_np.to_numpy()
        print('\nThe basic info of the {} data in table {}:\nshape: {} | data type: {}'.format(column_name, table_name,
                                                                                               pd_np.shape,
                                                                                               pd_np.dtype))
        np.save(self.folder + 'time_info_' + self.file_out, pd_np)
        print('-' * 60 + '\nFile saved at {}'.format(self.folder + 'time_info_' + self.file_out))

        con.close()

        return pd_np, column_name  # return a np.ndarray and the column name list