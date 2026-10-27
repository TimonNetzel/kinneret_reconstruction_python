
##------------------------------------------------------------------------------
## age depth relationship
##------------------------------------------------------------------------------

# three different age resolutions:
# 1: for each depth
# 2: 50 year steps
# we choose the regular 50 year steps:
age_depth_relationship = np.load("data/in/age_depth_relationship.npz")
depth_given_age = age_depth_relationship["depth_given_age"] # 50 years resolution
age = age_depth_relationship["age_regular"] # 50 years resolution
depth = age_depth_relationship["depth"]

##------------------------------------------------------------------------------
## transfer functions and recent climate data of Lake Kinneret
##------------------------------------------------------------------------------

# load the data of the machine learning competition (nnet is the winner)
nnet_fit_params = np.load("data/in/nnet_fit_params.npz")

# nnet parameters
wts_in_hidden = nnet_fit_params["wts_in_hidden"]
wts_hidden_out = nnet_fit_params["wts_hidden_out"]
wts_bias_hidden = nnet_fit_params["wts_bias_hidden"]
wts_bias_out = nnet_fit_params["wts_bias_out"]
num_biomes = len(wts_bias_out) - 1

# climate ranges
temp_range = nnet_fit_params["temp_range"]
pann_range = nnet_fit_params["pann_range"]
dims = len(temp_range)


# normalization (due to the machine learning competition)
normal_params = nnet_fit_params["normal_params"]
temp_range_norm = (temp_range - normal_params[0]) / normal_params[1]
pann_range_norm = norm.ppf(gamma.cdf(np.sqrt(pann_range), normal_params[2], scale=1/normal_params[3]))

# prediction on a 2D normalized climate grid
prediction_grid = np.empty((dims**2, 2))
prediction_grid[:,0] = np.tile(temp_range_norm, dims) # wrong
prediction_grid[:,1] = np.repeat(pann_range_norm, dims) # maybe correct

biome_probs = np.zeros((len(wts_bias_out),prediction_grid.shape[0])) 
for i in np.arange(prediction_grid.shape[0]):
    biome_probs[:,i] = pycpp.my_nnet_prediction(prediction_grid[i,0], prediction_grid[i,1],wts_in_hidden, wts_hidden_out, wts_bias_hidden, wts_bias_out)

b1_give_c = np.reshape( biome_probs[0,:], (dims, dims) )
b2_give_c = np.reshape( biome_probs[1,:], (dims, dims) )
b3_give_c = np.reshape( biome_probs[2,:], (dims, dims) )



# marginal distribution of each biome and climate value
biome_tf_temp = np.zeros((num_biomes, dims))
biome_tf_temp[0,:] = np.sum(b1_give_c,axis=0)
biome_tf_temp[1,:] = np.sum(b2_give_c,axis=0)
biome_tf_temp[2,:] = np.sum(b3_give_c,axis=0)
biome_tf_temp = biome_tf_temp/np.sum(biome_tf_temp,axis=1,keepdims=True)

biome_tf_pann = np.zeros((num_biomes, dims))
biome_tf_pann[0,:] = np.sum(b1_give_c,axis=1)
biome_tf_pann[1,:] = np.sum(b2_give_c,axis=1)
biome_tf_pann[2,:] = np.sum(b3_give_c,axis=1)
biome_tf_pann = biome_tf_pann/np.sum(biome_tf_pann,axis=1,keepdims=True)

# recent climate data based on CRU
prior_recent = nnet_fit_params["recent_climate"]


##------------------------------------------------------------------------------
## choose those core depths from the age depth model which are investigated
##------------------------------------------------------------------------------

depth_age_trans = np.empty((len(age), len(depth)))
depth_offset = np.min(depth)

for i in np.arange(0,len(age)):
    for j in np.arange(0,len(depth)):
        depth_age_trans[i, j] = depth_given_age[i, int((depth[j]-depth_offset))]
    depth_age_trans[i,:] = depth_age_trans[i,:] / np.sum(depth_age_trans[i,:])


##------------------------------------------------------------------------------
## taxa information from the core: pollen percentages, taxa and biome assignment,
## arboreal taxa data
##------------------------------------------------------------------------------


core_data = np.load("data/in/core_data.npz")
    
# taxa spectrum
taxa_spectrum_depth = core_data["taxa_spectrum_depth"]
num_taxa = len(taxa_spectrum_depth[0])

# weight the taxa spectrum to densities
taxa_spectrum_depth = taxa_spectrum_depth/np.sum(taxa_spectrum_depth, axis=1, keepdims=True)

# taxa and biome assignment
biomes_assign = core_data["biomes_assign"]
taxa_names = core_data["taxa_names"]

# age depth transformations: taxa spectrum 
taxa_spectrum_age = np.empty((len(age),num_taxa))
for i in np.arange(0,num_taxa):
    taxa_spectrum_age[:,i] = np.matmul(depth_age_trans,taxa_spectrum_depth[:,i])


# age depth transformations: arboreal taxa data 
# (this is the reference curve for the annual precipitation)
ap_depth = core_data["ap_depth"]
ap_age = np.matmul(depth_age_trans,ap_depth)
