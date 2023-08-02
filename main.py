# load packages
import numpy as np
import random
import os
import sys

import pybind11 # this package is needed to bind the C++ function from src/Pycpp_functions.cpp to python.
os.system("g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/Pycpp_functions.cpp -o src/pycpp`python3-config --extension-suffix`")

# import cpp functions
sys.path.append('src') 
import pycpp 

# include plot functions
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# include additional functions
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import gaussian_kde


# "random" settings
reproducible = True
seeds = 2023

# MCMC settings
mcmc_sampling = True # should the MCMC sampling be executed?
post_process_saved = False # are the post process values already saved?
sample_length = 1000000
burn_in = 250000
thin_size = 75


# load functions
exec(open("src/Py_functions.py").read())

# load data and prepare the MCMC sampling
exec(open("src/load_data.py").read())
exec(open("src/mcmc_preparation.py").read())

# MCMC sampling (only needed once)
if mcmc_sampling:
    exec(open("src/mcmc_execution.py").read())

# post processing of the MCMC samples and plotting
exec(open("src/mcmc_postprocessing.py").read())
