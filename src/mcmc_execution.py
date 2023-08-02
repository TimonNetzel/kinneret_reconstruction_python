 
 
##------------------------------------------------------------------------------
## MCMC execution
## Summary of the transfer function data, the information of the proposal 
## distributions, and the core data
##------------------------------------------------------------------------------  

# summary
tf_info = []
tf_info.append(wts_in_hidden)
tf_info.append(wts_hidden_out)
tf_info.append(wts_bias_hidden)
tf_info.append(wts_bias_out)
tf_info.append(normal_params)
tf_info.append(sds_tfs)
tf_info.append(tfs_lower)
tf_info.append(tfs_upper)

proposal_params = []
proposal_params.append([dirichlet_spread])
proposal_params.append(list(jeffreys_taxa_prior))
proposal_params.append([shape1])
proposal_params.append([shape2])
proposal_params.append([prior_recent[0]])
proposal_params.append([prior_recent[1]])
proposal_params.append([shape])
proposal_params.append([rate])

core_info = []
core_info.append([num_biomes])
core_info.append([num_taxa])
core_info.append([len(age)])
core_info.append(list(ap_age))
core_info.append(list(taxa_spectrum_age.T.flatten())) 
core_info.append(list(biomes_assign.T.flatten()))

sampling_info = []
sampling_info.append([sample_length])
seed_samples = [round(random.uniform(1, 127773)) for _ in range(sample_length)] # upper limit is prescribed by the corresponding C++ truncated normal function
sampling_info.append(seed_samples)


# MCMC execution
posterior = pycpp.mcmc_execution(taxa_weights,tf_sample,prior, core_info, proposal_params, tf_info, sampling_info)

# acceptance rate 
accept_cumsum = np.cumsum(posterior[0])
acc_rate = np.empty(sample_length-1)
for i in np.arange(1,sample_length):
    acc_rate[i-1] = accept_cumsum[i] / i
    

# save the posterior output
np.savez("data/out/posterior.npz",acceptance=posterior[0],post_tf_sample=posterior[1],post_taxa_weights=posterior[2],post_expl_variance=posterior[3],post_temp=posterior[4],post_pann=posterior[5],acc_rate=acc_rate)




