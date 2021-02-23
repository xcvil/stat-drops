## Project Structure

```shell script
stat-drops/
├── stat-drops/
│   ├── __init__.py
│   ├── common.py
│   ├── dbopts.py
│   ├── machine.py
│   ├── model.py
│   ├── npopts.py
|   ├── utils.py
│   └── __init__.py
├── data/
├── LICENSE
└── README.md
```

## Module Codebase

### core.model




### core.common
**Statistical measures**

core.common.kl_divergence 
* to calculate Kullback–Leibler divergence given expected and observed counts of drops for each diameter class
* probability is given by counts/total amount (of drops)

core.common.expectation_gamma
* to calculate the expectation of a gamma distribution

pearson_chi_square
* to calculate chi square

## Data generated from TBA database

* File format: .npy

* Data format: np.ndarray

* Data type: python object

* Data order

  array[Index, TEMP, Rainfall_Intensity, array[...]] 

  dtype='object'

| Index[0] | Date_time[1] | TEMP[2] | Rainfall Intensity[3] | Raw matrix[4:1028]       |
| -------- | ------------ | ------- | --------------------- | ------------------------ |
| 111111   |              | '035'   | '0000.000'            | array('000', ..., '000') |
| ...      |              | ...     | ...                   | ...                      |
| 999999   |              | '028'   | '0000.000'            | array('000', ..., '000') |