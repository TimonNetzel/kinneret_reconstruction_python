

##------------------------------------------------------------------------------
## MCMC preparation
##------------------------------------------------------------------------------ 

# If the MCMC output should be reproducible
if(reproducible): 
    random.seed(seeds)

##------------------------------------------------------------------------------ 
## define the parameters of the calibration density: proposed is a normal, gamma 
## and beta distribution of Pr(PP,C|A,P,Theta)
##------------------------------------------------------------------------------ 

# approximate the recent mean values of the precipitation with the parameters of the gamma distribution
r = np.linspace(0.001, 0.5, num=3000)
s1 = r*prior_recent[2]
s2 = (r * prior_recent[3])**2

shape = s1[np.argmin(np.abs(s1 - s2))]
rate = r[np.argmin(np.abs(s1 - s2))]


# these beta distribution parameters result in a good flexibility around the mean of 0.5
shape1 = 3 
shape2 = 3 


##------------------------------------------------------------------------------ 
## create the parameters of the proposal dirichlet distribution Pr(P|omega):
## the uncertaintie of each taxon is weighted with (num_taxa^2)
##------------------------------------------------------------------------------ 

dirichlet_spread = num_taxa**2 
jeffreys_taxa_prior = np.repeat(0.5, num_taxa)


##------------------------------------------------------------------------------ 
## create the parameters of the proposal distribution (truncated normal) from 
## which we want to sample from a higher resolution of the transfer function: Pr(P|C,psi)
##------------------------------------------------------------------------------ 

# data for the truncated normal proposal distributions
tf_quarts_temp = np.empty((num_biomes,3))
tf_quarts_pann = np.empty((num_biomes,3))
sds_temp = np.empty((num_biomes))
sds_pann = np.empty((num_biomes))

for i in np.arange(0,num_biomes):
    # quantiles (truncations)
    tf_quarts_temp[i,:] = my_quantiles(temp_range,weights=biome_tf_temp[i,:],probs = np.array([0.05,0.5,0.95]))
    tf_quarts_pann[i,:] = my_quantiles(pann_range,weights=biome_tf_pann[i,:],probs = np.array([0.05,0.5,0.95]))
    
    # although truncated normal dists are used (not full normal dists), the below adaption reveals a nice approximation of the best choice of the proposal sds
    expectation_temp = np.matmul(temp_range,biome_tf_temp[i,:])
    expectation_pann = np.matmul(pann_range,biome_tf_pann[i,:]) 
    sds_temp[i] = np.sqrt(np.matmul(((temp_range - expectation_temp)**2),biome_tf_temp[i,:]))
    sds_pann[i] = np.sqrt(np.matmul(((pann_range - expectation_pann)**2),biome_tf_pann[i,:]))
    # from Rosenthal 2010: "Optimal Proposal Distributions and Adaptive MCMC" (equation 5) 
    sds_temp[i] = sds_temp[i]*(2.38**2/(num_taxa))
    sds_pann[i] = sds_pann[i]*(2.38**2/(num_taxa))


# summary in vectors
sds_tfs = list(sds_temp)+list(sds_pann)
tfs_lower = list(tf_quarts_temp[:,0]) + list(tf_quarts_pann[:,0])
tfs_upper = list(tf_quarts_temp[:,2])+list(tf_quarts_pann[:,2])


##------------------------------------------------------------------------------ 
## create the prior list with the respective start values of the MCMC
##------------------------------------------------------------------------------ 

# start values of the taxa weights
taxa_weights = [list() for _ in range(sample_length)]
taxa_weights[0] = [1/num_taxa] * num_taxa

# start values of the climate samples from the transfer functions
tf_sample = [list() for _ in range(sample_length)]
tf_sample[0] = list(tf_quarts_temp[:,2]) + list(tf_quarts_pann[:,2] )


# prior
prior = []

# start values of the calibration proposal distribution:
expl_variance = [0] * sample_length
expl_variance[0] = random.gauss(0.2, 0.0005)  
prior.append(list(expl_variance))

recent_temp = [0] * sample_length
recent_temp[0] = random.gauss(4,0.005)
prior.append(list(recent_temp))

recent_pann = [0] * sample_length
recent_pann[0] = random.gauss(200,5)
prior.append(list(recent_pann))

acceptance = [0] * sample_length
acceptance[0] = 0
prior.append(list(acceptance))

