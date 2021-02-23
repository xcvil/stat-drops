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
import numpy as np
from scipy import stats


def expectation_gamma(mu_lambda):
    return mu_lambda[0] / mu_lambda[1]


def kl_divergence(px, qx, num_samples):
    kl = 0
    for i in range(num_samples):
        if px[i] == 0 or qx[i] == 0:
            pass
        else:
            kl += px[i] * np.log(px[i] / qx[i])
    return kl


def pearson_chi_square(dsd, para_estimated, diameter_interval):
    n = np.sum(dsd)
    p = stats.gamma.cdf(diameter_interval[1], para_estimated[0], loc=0, scale=para_estimated[1]**-1)-stats.gamma.cdf(diameter_interval[0], para_estimated[0], loc=0, scale=para_estimated[1]**-1)
    chi_square, i = 0, 0
    for _n, _p in zip(dsd[2:], p[2:]):
        if _p > 0.005:
            chi_square += (_n-n*_p)**2/(n*_p)
            i += 1
    return (chi_square, i-3)