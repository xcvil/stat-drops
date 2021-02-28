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


class DBOpts(object):
    """
    This class contains the operations for the .db file with sqlite3 and pandas
    CHOOSE argument of the self-defined functions correctly.
    """

    def __init__(self, folder, file_in, file_out=None):
        super(DBOpts, self).__init__()
        self.folder = folder
        self.file_in = file_in
        self.file_out = file_out  # file name that you wanna generate, default is None

    def chunker(self, seq, size):
        '''the chunker for tqdm processing bar from http://stackoverflow.com/a/434328'''
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def convert_db_to_csv(self, query, csv_name):
        con = sqlite3.connect(self.folder + self.file_in)
        pd_sql = pd.read_sql(query, con)
        pd_sql.to_csv(self.folder + csv_name + '.csv', index=False)
        con.close()
        print('Converting finished!')

    def extract_col_to_csv(self, query, table, col):

        if type(col) != str or type(table) != str:
            raise ValueError('col or table arg should be a str')

        con = sqlite3.connect(self.folder + self.file_in)
        pd_sql = pd.read_sql(query, con)
        pd_sql[col].to_csv(self.folder + table + '_' + col + '.csv', index=False)
        con.close()

    def edit_sql_col(self, query):
        '''ONLY specific with this type of error. Split OK, 0000.00 and fix them all'''
        starttime = time.time()

        con = sqlite3.connect(self.folder + self.file_in)
        col = pd.read_sql(query, con)

        temp = col['datalogger_voltage'].str.split(",", n=1, expand=True)

        col['datalogger_voltage'] = temp[0]
        col['rain_accumulated_32_bit'] = col['rain_intensity_32_bit']
        col['rain_intensity_32_bit'] = temp[1]

        con.close()
        endtime = time.time()
        print('Editing is finished! {} seconds used. Now conversion starting'.format(endtime - starttime))

        return col  # return DataFrame

    def convert_DataFrame_to_db(self, table, df):
        #########################################################################
        # This one seems safer than convert_DataFrame_to_db_with_processbar.    #
        # I cannot tell why exactly. It is jus a feeling.                      #
        #########################################################################
        if type(table) != str:
            raise ValueError('table arg should be a str')

        db_name = 'sqlite:///' + self.folder + self.file_out
        engine = create_engine(db_name, echo=False)
        connection = engine.raw_connection()
        df.to_sql(table, con=connection, index=False)

    def convert_DataFrame_to_db_with_processbar(self, table, df):
        '''df: DataFrame
        db_name: 'sqlite:///YourNewDatabaseName.db'

        With a processing bar, looks beautiful!
        '''
        if type(table) != str:
            raise ValueError('table arg should be a str')

        db_name = 'sqlite:///' + self.folder + self.file_out
        engine = create_engine(db_name, echo=False)
        connection = engine.raw_connection()

        chunksize = int(len(df) / 10)  # 10%

        with tqdm(total=len(df), desc="coverting to .db file") as pbar:
            for i, cdf in enumerate(self.chunker(df, chunksize)):  # how to call function inside the class
                replace = "replace" if i == 0 else "append"
                cdf.to_sql(table, con=connection, if_exists=replace, index=False)
                pbar.update(chunksize)

    def sql_query(self, query):
        '''SQL query operation with selecting query properly
        
        show tables' name: 'SELECT name FROM sqlite_master WHERE type = 'table''
        show headers of column of THE TABLE: 'SELECT * FROM {} LIMIT 1'.format(THE TABLE)
        show first/last entry of THE TABLE: 'SELECT * FROM {0} WHERE rowid = (SELECT MIN/MAX(rowid) FROM {0})'.format(THE TABLE)
        Without reading the data through pandas
        '''

        con = sqlite3.connect(self.folder + self.file_in)
        c = con.cursor()

        try:
            c.execute(query)
            response = c.fetchall()
            desc = c.description
        except Exception as e:
            raise e
        finally:
            c.close()
            con.close()

        return response, desc

    def brief_check(self, query):
        con = sqlite3.connect(self.folder + self.file_in)
        col = pd.read_sql(query, con)
        print(col.info())
        con.close()

        return col


if __name__ == '__main__':
    '''Aim to show the basic info of the database (tables, columns, and rows). 
    Fix the error and export the data as .db file'''

    folder = '/home/xiaochenzheng/Desktop/Master_Project/2019_Disdros_TBA/Data/'
    file_in = 'disdrometers_basel.db'
    file_out = 'a.db'
    opts = DBOpts(folder, file_in, file_out)

    # show all tables in the database
    print("Tables in database {}.".format(folder + file_in))
    query1 = 'SELECT name FROM sqlite_master WHERE type = "table"'
    tables, desc = opts.sql_query(query1)
    print(tables)

    # list all titles of columns
    print("-" * 60)
    for table in tables:
        print("\nColumns of table {}.".format(table[0]))
        query2 = 'SELECT * FROM {} LIMIT 1'.format(table[0])
        RE, desc = opts.sql_query(query2)
        print([d[0] for d in desc])

    # list the first and last five rows of each table
    print("-" * 60, "\n")
    for table in tables:
        print("\nLast entry of table {}.".format(table[0]))
        response_min, _ = opts.sql_query(
            'SELECT * FROM {0} WHERE rowid = (SELECT MIN(rowid) FROM {0})'.format(table[0]))
        print(response_min)
        response_max, _ = opts.sql_query(
            'SELECT * FROM {0} WHERE rowid = (SELECT MAX(rowid) FROM {0})'.format(table[0]))
        print(response_max)

    # Fix the error and save the DataFrame as .db file
    print("-" * 60)
    for table in tables:
        query3 = 'SELECT * FROM {}'.format(table[0])
        # table = ('station_40',) so use table[0] here otherwise table is a tuple not a string
        temp = opts.edit_sql_col(query3)
        opts.convert_DataFrame_to_db_with_processbar(table[0], temp)
