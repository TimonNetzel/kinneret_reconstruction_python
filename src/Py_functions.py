
##------------------------------------------------------------------------------ 
## function which calculates quantiles without the need of samples like 
## base::sample
##------------------------------------------------------------------------------ 

def my_quantiles(x, weights=None, probs=None):
    n = len(x)
    rw = np.cumsum(weights)/np.sum(weights)
    q = np.zeros(len(probs))
    for i in range(len(probs)):
        p = probs[i]
        if p == 0:
            q[i] = x[0]
        elif p == 1:
            q[i] = x[n-1]
        else:
            select = np.min(np.where(rw >= p))
            if rw[select] == p:
                q[i] = np.mean(x[select:(select + 2)])
            else:
                q[i] = x[select]
    return q


##------------------------------------------------------------------------------ 
## function which calculates provides the most important data from the posterior
## samples
##------------------------------------------------------------------------------ 
 
def post_processing(posterior,age,depth,burn_in,sample_length,thin_size,nnet_fit_params):


    ##------------------------------------------------------------------------------
    ## posterior climate: summarize all posterior reconstruction 
    ## and the respective biome weights
    ##------------------------------------------------------------------------------

    # thinning
    which_samples = np.linspace(burn_in,sample_length-1,(sample_length-burn_in)/thin_size).astype(int)

    ct1 = -1
    ct2 = -1
    
    all_tmp_mean_age = np.empty((len(which_samples),len(age)))
    all_pann_mean_age = np.empty((len(which_samples),len(age)))
    
    all_tmp_mean_depth = np.empty((len(which_samples),len(depth)))
    all_pann_mean_depth = np.empty((len(which_samples),len(depth)))
    
    all_biomes_age = np.empty((len(which_samples),len(age),num_biomes))
    all_biomes_depth = np.empty((len(which_samples),len(depth),num_biomes))
    
    post_tf_sample = posterior["post_tf_sample"]
    post_taxa_weights = posterior["post_taxa_weights"]
    
    for i in which_samples:

        # biome probability: depth
        biome_ratios_depth_temp = np.array(pycpp.spectrum_to_biome_assign(list(taxa_spectrum_depth.T.flatten()), post_taxa_weights[(num_taxa*i):(num_taxa*i+num_taxa)], list(biomes_assign.T.flatten()), len(depth), num_taxa, num_biomes))
        biome_ratios_depth = biome_ratios_depth_temp.reshape(num_biomes,len(depth)).T
  
        
        # reconst: depth
        tf_samples_matrix = np.array(post_tf_sample[(2*num_biomes*i):(2*num_biomes*i+(2*num_biomes))]).reshape(2, num_biomes).T 
        reconst_mean_depth = np.matmul( biome_ratios_depth,tf_samples_matrix)
        ct1 = ct1 + 1
        all_tmp_mean_depth[ct1,:] = reconst_mean_depth[:,0]
        all_pann_mean_depth[ct1,:] = reconst_mean_depth[:,1]
        all_biomes_depth[ct1,:,:] = biome_ratios_depth

        
        # biome probability: age
        biome_ratios_age_temp = np.array(pycpp.spectrum_to_biome_assign(list(taxa_spectrum_age.T.flatten()), post_taxa_weights[(num_taxa*i):(num_taxa*i+num_taxa)], list(biomes_assign.T.flatten()), len(age), num_taxa, num_biomes))
        biome_ratios_age = biome_ratios_age_temp.reshape(num_biomes,len(age)).T

        # reconst: age
        reconst_mean_age = np.matmul(biome_ratios_age,tf_samples_matrix)
        ct2 = ct2 + 1
        all_tmp_mean_age[ct2,:] = reconst_mean_age[:,0]
        all_pann_mean_age[ct2,:] = reconst_mean_age[:,1]
        all_biomes_age[ct2,:,:] = biome_ratios_age
    

    ##------------------------------------------------------------------------------
    ## create arrays which contain the densities of the posterior reconstructions
    ## and biome percentages
    ##------------------------------------------------------------------------------

    
    temp_range = nnet_fit_params["temp_range"]
    pann_range = nnet_fit_params["pann_range"] 
    linear_pann_range = np.linspace(0, 6000, dims)

    # depth
    post_temp_array_depth = np.empty((len(depth), dims))
    post_pann_array_depth = np.empty((len(depth), dims))
    post_biomes_array_depth = np.empty((num_biomes,len(depth), dims))
    for i in np.arange(0,len(depth)):

        # densities per time step: reconst
        post_temp_dens_temp = gaussian_kde(all_tmp_mean_depth[:,i])
        post_temp_dens_temp = post_temp_dens_temp.evaluate(np.linspace(min(temp_range), max(temp_range), dims))
        post_temp_dens_temp = post_temp_dens_temp/sum(post_temp_dens_temp)
        
        
        post_pann_dens_temp = gaussian_kde(all_pann_mean_depth[:,i])
        post_pann_dens_temp = post_pann_dens_temp.evaluate(linear_pann_range)
        post_pann_dens_temp = post_pann_dens_temp/sum(post_pann_dens_temp)

        
        # densities per time step: biomes
        for j in np.arange(0,num_biomes):
            post_biomes_temp = gaussian_kde(all_biomes_depth[:,i,j])
            post_biomes_temp = post_biomes_temp.evaluate(np.linspace(0, 1, dims))   
            post_biomes_array_depth[j,i,:] = post_biomes_temp/sum(post_biomes_temp)    
        

        # posterior densities summary
        post_temp_array_depth[i,:] = post_temp_dens_temp
        post_pann_array_depth[i,:] = post_pann_dens_temp

  
  
    # age
    post_temp_array_age = np.empty((len(age), dims)) 
    post_pann_array_age = np.empty((len(age), dims))
    post_biomes_array_age = np.empty((num_biomes,len(age), dims))
    for i in np.arange(0,len(age)):

        # densities per time step: reconst
        post_temp_dens_temp = gaussian_kde(all_tmp_mean_age[:,i])
        post_temp_dens_temp = post_temp_dens_temp.evaluate(np.linspace(min(temp_range), max(temp_range), dims))
        post_temp_dens_temp = post_temp_dens_temp/sum(post_temp_dens_temp)
        
        post_pann_dens_temp = gaussian_kde(all_pann_mean_age[:,i])
        post_pann_dens_temp = post_pann_dens_temp.evaluate(linear_pann_range)
        post_pann_dens_temp = post_pann_dens_temp/sum(post_pann_dens_temp)

        
        # densities per time step: biomes
        for j in np.arange(0,num_biomes):
            post_biomes_temp = gaussian_kde(all_biomes_age[:,i,j])
            post_biomes_temp = post_biomes_temp.evaluate(np.linspace(0, 1, dims))   
            post_biomes_array_age[j,i,:] = post_biomes_temp/sum(post_biomes_temp)    
        

        # posterior densities summary
        post_temp_array_age[i,:] = post_temp_dens_temp
        post_pann_array_age[i,:] = post_pann_dens_temp
        
        
        
    ##---------------------------------------------------------------------------------------------------------
    ## calculate the quartiles of the posterior reconstructions and biome percentages
    ##---------------------------------------------------------------------------------------------------------
    
    
    temp_int = np.linspace(min(temp_range), max(temp_range), 10000)
    pann_int = np.linspace(min(linear_pann_range), max(linear_pann_range), 10000)

    # depth
    temp_depth_25 = np.empty(len(depth))
    temp_depth_50 = np.empty(len(depth))
    temp_depth_75 = np.empty(len(depth))
    pann_depth_25 = np.empty(len(depth))
    pann_depth_50 = np.empty(len(depth))
    pann_depth_75 = np.empty(len(depth))
    for i in np.arange(0,len(depth)):
        f_temp = interp1d(temp_range, post_temp_array_depth[i,:], kind="linear", fill_value="extrapolate")
        probs_temp_int = f_temp(temp_int)
        probs_temp_int[probs_temp_int < 0] = 1e-6
        probs_temp_int = probs_temp_int/sum(probs_temp_int)
        
        f_pann = interp1d(linear_pann_range, post_pann_array_depth[i,:], kind="linear", fill_value="extrapolate")
        probs_pann_int = f_pann(pann_int)
        probs_pann_int[probs_pann_int < 0] = 1e-6
        probs_pann_int = probs_pann_int/sum(probs_pann_int)

        quantiles_temp_depth = my_quantiles(temp_int,weights=probs_temp_int,probs = np.array([0.25,0.5,0.75]))
        temp_depth_25[i] = quantiles_temp_depth[0]
        temp_depth_50[i] = quantiles_temp_depth[1]
        temp_depth_75[i] = quantiles_temp_depth[2]
        quantiles_pann_depth = my_quantiles(pann_int,weights=probs_pann_int,probs = np.array([0.25,0.5,0.75]))
        pann_depth_25[i] = quantiles_pann_depth[0]
        pann_depth_50[i] = quantiles_pann_depth[1]
        pann_depth_75[i] = quantiles_pann_depth[2]
    
     
    # age
    temp_age_25 = np.empty(len(age))
    temp_age_50 = np.empty(len(age))
    temp_age_75 = np.empty(len(age))
    pann_age_25 = np.empty(len(age))
    pann_age_50 = np.empty(len(age))
    pann_age_75 = np.empty(len(age))
    for i in np.arange(0,len(age)):
        f_temp = interp1d(temp_range, post_temp_array_age[i,:], kind="linear", fill_value="extrapolate")
        probs_temp_int = f_temp(temp_int)
        probs_temp_int[probs_temp_int < 0] = 1e-6
        probs_temp_int = probs_temp_int/sum(probs_temp_int)
        
        f_pann = interp1d(linear_pann_range, post_pann_array_age[i,:], kind="linear", fill_value="extrapolate")
        probs_pann_int = f_pann(pann_int)
        probs_pann_int[probs_pann_int < 0] = 1e-6
        probs_pann_int = probs_pann_int/sum(probs_pann_int)

        quantiles_temp_age = my_quantiles(temp_int,weights=probs_temp_int,probs = np.array([0.25,0.5,0.75]))
        temp_age_25[i] = quantiles_temp_age[0]
        temp_age_50[i] = quantiles_temp_age[1]
        temp_age_75[i] = quantiles_temp_age[2]
        quantiles_pann_age = my_quantiles(pann_int,weights=probs_pann_int,probs = np.array([0.25,0.5,0.75]))
        pann_age_25[i] = quantiles_pann_age[0]
        pann_age_50[i] = quantiles_pann_age[1]
        pann_age_75[i] = quantiles_pann_age[2]


    # biomes
    biomes_int = np.linspace(0,1,10000)
    biome_depth_25 = np.empty((num_biomes, len(depth)))
    biome_depth_50 = np.empty((num_biomes, len(depth)))
    biome_depth_75 = np.empty((num_biomes, len(depth)))
    biome_age_25 = np.empty((num_biomes, len(age)))
    biome_age_50 = np.empty((num_biomes, len(age)))
    biome_age_75 = np.empty((num_biomes, len(age)))
    for i in np.arange(0,num_biomes):
        # depth
        for j in np.arange(0,len(depth)):
            f_temp = interp1d(np.linspace(0,1,dims), post_biomes_array_depth[i,j,:], kind="linear", fill_value="extrapolate")
            probs_biome_depth_int = f_temp(biomes_int)
            probs_biome_depth_int[probs_biome_depth_int < 0] = 1e-6
            probs_biome_depth_int = probs_biome_depth_int/sum(probs_biome_depth_int)
            quantiles_temp_depth = my_quantiles(biomes_int,weights=probs_biome_depth_int,probs = np.array([0.25,0.5,0.75]))
            biome_depth_25[i,j] = quantiles_temp_depth[0]
            biome_depth_50[i,j] = quantiles_temp_depth[1]
            biome_depth_75[i,j] = quantiles_temp_depth[2]
            
    
        # age
        for j in np.arange(0,len(age)):
            f_temp = interp1d(np.linspace(0,1,dims), post_biomes_array_age[i,j,:], kind="linear", fill_value="extrapolate")
            probs_biome_age_int = f_temp(biomes_int)
            probs_biome_age_int[probs_biome_age_int < 0] = 1e-6
            probs_biome_age_int = probs_biome_age_int/sum(probs_biome_age_int)
            quantiles_temp_age = my_quantiles(biomes_int,weights=probs_biome_age_int,probs = np.array([0.25,0.5,0.75]))
            biome_age_25[i,j] = quantiles_temp_age[0]
            biome_age_50[i,j] = quantiles_temp_age[1]
            biome_age_75[i,j] = quantiles_temp_age[2]        


    ##---------------------------------------------------------------------------------------------------------
    ## Posterior parameter: posterior transfer functions and taxa weights
    ##---------------------------------------------------------------------------------------------------------

    # densities of the prior tansfer functions
    temp_int = interp1d(np.linspace(0,1,dims), temp_range, kind="linear", fill_value="extrapolate")(np.linspace(0,1,len(which_samples)))
    pann_int = interp1d(np.linspace(0,1,dims), pann_range, kind="linear", fill_value="extrapolate")(np.linspace(0,1,len(which_samples)))
    
    tf_prior_temp_sample = np.empty((len(which_samples), num_biomes))
    tf_prior_pann_sample = np.empty((len(which_samples), num_biomes))
    for j in np.arange(0,num_biomes):
        tf_prior_temp_sample[:,j] = np.random.choice(temp_range, size=len(which_samples), p=biome_tf_temp[j,], replace=True)
        tf_prior_pann_sample[:,j] = np.random.choice(pann_range, size=len(which_samples), p=biome_tf_pann[j,], replace=True)
      
    # posterior tansfer functions samples
    tf_post_temp = np.empty((len(which_samples), num_biomes))
    tf_post_pann = np.empty((len(which_samples), num_biomes)) 
    ct = -1
    for i in which_samples:
        ct  = ct + 1
        tf_samples_temp = post_tf_sample[(2*num_biomes*i):(2*num_biomes*i+(2*num_biomes))]
        tf_post_temp[ct,:] = tf_samples_temp[0:3]
        tf_post_pann[ct,:] = tf_samples_temp[3:6]


    # ratio of posterior TFs to prior TFs CI sizes 
    prior_temp_ci_size = np.empty(num_biomes)
    post_temp_ci_size = np.empty(num_biomes)
    prior_pann_ci_size = np.empty(num_biomes)
    post_pann_ci_size = np.empty(num_biomes)
    for i in np.arange(0,num_biomes):
        prior_temp_ci_size[i] = np.diff(np.quantile(tf_prior_temp_sample[:,i], q=[0.025, 0.975]))
        post_temp_ci_size[i] = np.diff(np.quantile(tf_post_temp[:,i], q=[0.025, 0.975]))
        prior_pann_ci_size[i] = np.diff(np.quantile(tf_prior_pann_sample[:,i], q=[0.025, 0.975]))
        post_pann_ci_size[i] = np.diff(np.quantile(tf_post_pann[:,i], q=[0.025, 0.975]))

    
    # array with the posterior taxa weights
    post_weights = np.empty((len(which_samples),num_taxa ))
    for j in np.arange(0,num_taxa):
        ct = -1
        for i in which_samples:
            ct = ct + 1
            taxa_weights_temp = post_taxa_weights[(num_taxa*i):(num_taxa*i+num_taxa)]
            post_weights[ct,j] = taxa_weights_temp[j]
            
       
    # posterior tf sample prediction on a 2D climate grid
    post_tf_probs = np.zeros((num_biomes,dims,dims))
    for i in np.arange(num_biomes):
        
        x=tf_post_pann[:,i] 
        y=tf_post_temp[:,i]   
        xy = np.vstack((x, y))  
        
        dens = gaussian_kde(xy)  
        
        gx, gy = np.mgrid[pann_range.min():pann_range.max():dims*1j, temp_range.min():temp_range.max():dims*1j]
        gxy = np.dstack((gx, gy)) 
        
        z = np.apply_along_axis(dens, 2, gxy)   
        
        post_tf_probs[i,:,:] = z.reshape(dims, dims) 
        post_tf_probs[i,:,:] = post_tf_probs[i,:,:]/np.max(post_tf_probs[i,:,:])
              

    ##---------------------------------------------------------------------------------------------------------
    ## summary of the post processing output
    ##---------------------------------------------------------------------------------------------------------

    
    ## save the output
    np.savez("data/out/post_process.npz",which_samples=which_samples,linear_pann_range=linear_pann_range,post_weights=post_weights,prior_temp_ci_size=prior_temp_ci_size,post_temp_ci_size=post_temp_ci_size,prior_pann_ci_size=prior_pann_ci_size,post_pann_ci_size=post_pann_ci_size,tf_prior_temp_sample=tf_prior_temp_sample,tf_post_temp=tf_post_temp,tf_prior_pann_sample=tf_prior_pann_sample,tf_post_pann=tf_post_pann,biome_depth_25=biome_depth_25,biome_depth_50=biome_depth_50,biome_depth_75=biome_depth_75,biome_age_25=biome_age_25,biome_age_50=biome_age_50,biome_age_75=biome_age_75,temp_depth_25=temp_depth_25,temp_depth_50=temp_depth_50,temp_depth_75=temp_depth_75,temp_age_25=temp_age_25,temp_age_50=temp_age_50,temp_age_75=temp_age_75,pann_depth_25=pann_depth_25,pann_depth_50=pann_depth_50,pann_depth_75=pann_depth_75,pann_age_25=pann_age_25,pann_age_50=pann_age_50,pann_age_75=pann_age_75,post_temp_array_depth=post_temp_array_depth,post_temp_array_age=post_temp_array_age,post_pann_array_depth=post_pann_array_depth,post_pann_array_age=post_pann_array_age,post_biomes_array_depth=post_biomes_array_depth,post_biomes_array_age=post_biomes_array_age,post_tf_probs=post_tf_probs)
 
