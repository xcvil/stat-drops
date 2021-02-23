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
import math


def gamma_log_likelihood_func(mu_lambda, num_samples, diameter):
    return num_samples * (mu_lambda[0] * np.log(mu_lambda[1]) - np.log(math.gamma(mu_lambda[0]))) + \
           np.sum((mu_lambda[0]-1) * np.log(diameter) - mu_lambda[1] * diameter)


def gamma_neg_log_likelihood_func(mu_lambda, num_samples, diameter):
    return -gamma_log_likelihood_func(mu_lambda, num_samples, diameter)


def pdf(diameter, mu_lambda):
    return (1/(math.gamma(mu_lambda[0]))) * mu_lambda[1]**mu_lambda[0] * diameter**(mu_lambda[0]-1) * \
           np.exp(-mu_lambda[1]*diameter)


def n_zero(n_d, diameter, mu_lambda):
    return n_d/(diameter**mu_lambda[0]*np.exp(-mu_lambda[1]*diameter))


# def corrected_log_likelihood(mu_lambda, num_samples, diameter):
#     return num_samples * (mu_lambda[0] * np.log(mu_lambda[1]) - np.log(math.gamma(mu_lambda[0]))) + \
#            np.sum((mu_lambda[0]-1) * np.log(diameter+mu_lambda[2]) - mu_lambda[1] * (diameter+mu_lambda[2]))
#
#
# def corrected_neg_log_likelihood(mu_lambda, num_samples, diameter):
#     return -corrected_log_likelihood(mu_lambda, num_samples, diameter)


def binom_neg_log_likelihood_func(mu_lambda, num_drops, diameter_interval, total_num_drops):
    """args:
    num_drops: number of drops for different classes i;
    diameter_interval: 2x32 array where the diameter_interval[0] is the lower bound of interval; diameter_interval[0].shape = (32,)
    """
    P = stats.gamma.cdf(diameter_interval[1], mu_lambda[0], loc=0, scale=mu_lambda[1]**-1)\
        -stats.gamma.cdf(diameter_interval[0], mu_lambda[0], loc=0, scale=mu_lambda[1]**-1)
    neg_sum = 0
    for (n, p) in zip(num_drops, P):
        neg_sum -= stats.binom.logpmf(n, total_num_drops, p)
    return neg_sum


def num_drops_func(diameter, N0, mu, lam):
    return N0*diameter**(mu-1)*np.exp(-lam*diameter)


def num_drops_integrate_func(d_interval, ntot, mu, lam):
    """args:
    d_interval.shape = [32, 2]
    """
    P = stats.gamma.cdf(d_interval[:,0], mu, loc=0, scale=lam**-1)-stats.gamma.cdf(d_interval[:,1], mu, loc=0, scale=lam**-1)
    return ntot*P
