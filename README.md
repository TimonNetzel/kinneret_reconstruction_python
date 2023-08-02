FUNCTIONALITY:

    The scripts perform the MCMC simulation described in the publication.
    Each script is written very simply and clearly to ensure a quick insight into how it works.
    The Python-based scripts contain some MCMC functions written in C++. 
    Only a rather rarely used Python package (pycpp) has to be installed at the beginning, which creates an 
    interface between Python and C++.


TO RUN RECONSTRUCTION:

    Open 'main.py'.
    
    Install the following packages if necessary: pycpp,... (e.g. pip install pycpp,..).
    
    After that, the C++ functions must be compiled and included into Python (for each individual Python version).
    For this purpose, the three commented out lines can be used under Linux.
    This only has to be done once, so that the C++ functions can then be imported with import pycpp.

    Import all other required packages and set the settings for reproducibility and MCMC.
    
    Load and prepare all data stored in npy files (data/in).

    Run and save the MCMC simulation (takes about 40 seconds on a standard CPU and is saved in data/out).

    Run the post-processing routine that calculates, stores and plots the most important posterior metrics 
    (data/out, plots).

CHANGES OF BASIC SETTINGS:

    If reproducibly is set to "False", each MCMC simulation will give a slightly different result.
    Since our MCMC simulation converges, the differences are minimal and the main features of the results are preserved.
    
    MCMC sampling only needs to be done once.
    That is why there is the possibility to switch it off.
    In addition, the calculation of the post-processing metrics takes some time.
    For this reason, it is also possible to save it after a one-time calculation so that it does not have 
    to be performed again.
    The convergence test described in the publication is based on the predefined parameters of sample length, 
    burn-in and thin size.
    A corresponding change should therefore be treated with caution.

