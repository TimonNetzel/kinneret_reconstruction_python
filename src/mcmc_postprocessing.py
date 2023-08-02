

##------------------------------------------------------------------------------------------------------ 
## MCMC post processing 
##------------------------------------------------------------------------------------------------------  

# load the posterior data
posterior = np.load("data/out/posterior.npz")


# post processing
if not post_process_saved:
    post_processing(posterior,age,depth,burn_in,sample_length,thin_size,nnet_fit_params)

post_process = np.load("data/out/post_process.npz")


##------------------------------------------------------------------------------------------------------ 
## plots of the posterior climate reconstruction
##------------------------------------------------------------------------------------------------------ 


# define the same colormap as the terrain colors in R
terrain_colors_rgb = np.load("data/in/terrain_colors.npz")
terrain_colors = cm.get_cmap(lut=500)
terrain_colors.colors = terrain_colors_rgb["terrain_colors_rgb"].T


# define ylims of both climate variables
temp_range_template = np.arange(-30, 31, 5)
precip_range_template = np.arange(0, 4001, 200)

temp_range_temp = [min(post_process["temp_age_50"])-5 , max(post_process["temp_age_50"])+5]  
lower_ylim = temp_range_template[np.abs(temp_range_template - temp_range_temp[0]).argmin()]
upper_ylim = temp_range_template[np.abs(temp_range_template - temp_range_temp[1]).argmin()]
lower_temp_id = np.abs(temp_range - lower_ylim).argmin()
upper_temp_id = np.abs(temp_range - upper_ylim).argmin()
temp_array = post_process["post_temp_array_age"].T[np.arange(dims-1, -1,-1),:]
temp_array_cutted = temp_array[np.arange(dims-upper_temp_id, dims-lower_temp_id,1),:]


precip_range_temp = [0,round(max(post_process["pann_age_50"])+300)]
lower_ylim = 0
upper_ylim = precip_range_template[np.abs(precip_range_template - precip_range_temp[1]).argmin()]
lower_pann_id = np.abs(post_process["linear_pann_range"] - lower_ylim).argmin()
upper_pann_id = np.abs(post_process["linear_pann_range"] - upper_ylim).argmin()
pann_array = post_process["post_pann_array_age"].T[np.arange(dims-1, -1,-1),:]
pann_array_cutted = pann_array[np.arange(dims-upper_pann_id, dims-lower_pann_id,1),:]



# plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8)) 
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)

im1 = axs[0].imshow(X=temp_array_cutted,cmap=terrain_colors,extent=[min(age), max(age), temp_range[lower_temp_id], temp_range[upper_temp_id]],interpolation="none", aspect="auto")
plt.colorbar(im1,ax=axs[0])
axs[0].plot(age,post_process["temp_age_25"], color="black", linewidth=1, linestyle="dashed")
axs[0].plot(age,post_process["temp_age_50"], color="black", linewidth=2)
axs[0].plot(age,post_process["temp_age_75"], color="black", linewidth=1, linestyle="dashed")
axs[0].set_title("(a) Temperature",loc = "left", fontweight="bold")
axs[0].set_ylabel("$\mathregular{T_{DJF}}$ [°C]")

im2 = axs[1].imshow(X=pann_array_cutted,cmap=terrain_colors,extent=[min(age), max(age), post_process["linear_pann_range"][lower_pann_id], post_process["linear_pann_range"][upper_pann_id]],interpolation="none", aspect="auto")
plt.colorbar(im2,ax=axs[1])
axs[1].plot(age,post_process["pann_age_25"], color="black", linewidth=1, linestyle="dashed")
axs[1].plot(age,post_process["pann_age_50"], color="black", linewidth=2)
axs[1].plot(age,post_process["pann_age_75"], color="black", linewidth=1, linestyle="dashed")
axs[1].set_title("(b) Precipitation",loc = "left", fontweight="bold")
axs[1].set_xlabel("Age [cal a BP]")
axs[1].set_ylabel("$\mathregular{P_{ANN}}$ [mm]")

plt.tight_layout()
plt.savefig("plots/reconstruction.pdf")
plt.close(fig)



##------------------------------------------------------------------------------------------------------ 
## plots of the posterior biome percentages with respect to time
##------------------------------------------------------------------------------------------------------ 


biome_names = ["Mediterranean biome","Irano-Turanian biome","Saharo-Arabian biome"]

biomes_dists = post_process["post_biomes_array_age"]
med_biome_dist = biomes_dists[0,:,:].T[np.arange(dims-1, -1,-1),:]
irano_biome_dist = biomes_dists[1,:,:].T[np.arange(dims-1, -1,-1),:]
saharo_biome_dist = biomes_dists[2,:,:].T[np.arange(dims-1, -1,-1),:]


# plot
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8)) 

im1 = axs[0].imshow(X=med_biome_dist,cmap=terrain_colors,extent=[min(age), max(age), 0, 1],interpolation="none", aspect="auto")
plt.colorbar(im1,ax=axs[0])
axs[0].plot(age,post_process["biome_age_25"][0,:], color="black", linewidth=1, linestyle="dashed")
axs[0].plot(age,post_process["biome_age_50"][0,:], color="black", linewidth=2)
axs[0].plot(age,post_process["biome_age_75"][0,:], color="black", linewidth=1, linestyle="dashed")
axs[0].set_title("(a) "+ biome_names[0],loc = "left", fontweight="bold")


im2 = axs[1].imshow(X=irano_biome_dist,cmap=terrain_colors,extent=[min(age), max(age), 0, 1],interpolation="none", aspect="auto")
plt.colorbar(im2,ax=axs[1])
axs[1].plot(age,post_process["biome_age_25"][1,:], color="black", linewidth=1, linestyle="dashed")
axs[1].plot(age,post_process["biome_age_50"][1,:], color="black", linewidth=2)
axs[1].plot(age,post_process["biome_age_75"][1,:], color="black", linewidth=1, linestyle="dashed")
axs[1].set_title("(b) "+ biome_names[1],loc = "left", fontweight="bold")


im2 = axs[2].imshow(X=saharo_biome_dist,cmap=terrain_colors,extent=[min(age), max(age), 0, 1],interpolation="none", aspect="auto")
plt.colorbar(im2,ax=axs[2])
axs[2].plot(age,post_process["biome_age_25"][2,:], color="black", linewidth=1, linestyle="dashed")
axs[2].plot(age,post_process["biome_age_50"][2,:], color="black", linewidth=2)
axs[2].plot(age,post_process["biome_age_75"][2,:], color="black", linewidth=1, linestyle="dashed")
axs[2].set_title("(c) "+ biome_names[2],loc = "left", fontweight="bold")
axs[2].set_xlabel("Age [cal a BP]")

plt.tight_layout()
plt.savefig("plots/biomes.pdf")
plt.close(fig)


##------------------------------------------------------------------------------------------------------ 
## plots of the posterior and prior taxa weights
##------------------------------------------------------------------------------------------------------ 

# taxa colors in terms of assigned biomes
biome_id = np.where(biomes_assign == 1)[1]
biome_cols = ["#A2CD5A", "#EE8262", "#FFB90F"]
taxa_g_biomes_cols = ["" for _ in range(len(taxa_names))]
for i in np.arange(0,len(taxa_names)):
    taxa_g_biomes_cols[i] = biome_cols[biome_id[i]]

post_weights = post_process["post_weights"]


# plot
plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.05)

bplot = plt.boxplot(post_weights, vert=False, labels=taxa_names, notch=True, patch_artist=True, medianprops=dict(color="black"),whis=None)
plt.xlim([0, 0.25])
plt.title("Posterior and prior taxa weights",loc = "left", fontweight="bold")
# fill with colors
for patch, color in zip(bplot["boxes"], taxa_g_biomes_cols):
    patch.set_facecolor(color)
# prior weights
plt.axvline(x=1/num_taxa, color="black", zorder=3)

plt.savefig("plots/taxa_weights.pdf")
plt.close()

##------------------------------------------------------------------------------------------------------ 
## plots of the posterior and prior transfer functions (image plots)
##------------------------------------------------------------------------------------------------------


# smoothed posterior transfer function samples
post_tf_probs = post_process["post_tf_probs"]

# nonlinear precipitation axis projected onto a linear axis
y = np.linspace(0, dims, dims)
x = pann_range
spl = splrep(x, y)
x2 =  np.linspace(pann_range.min(), pann_range.max(), dims)
y2 = splev(x2, spl)

pann_interpol_ids = (np.round(y2)).astype(int)-1   
pann_interpol_ids = np.where(pann_interpol_ids >= 0, pann_interpol_ids,0)

b1_give_c_interpol = np.zeros((dims,dims))
b2_give_c_interpol = np.zeros((dims,dims))
b3_give_c_interpol = np.zeros((dims,dims))

for i in np.arange(dims):
    for j in np.arange(dims):
        b1_give_c_interpol[j,i] = b1_give_c[pann_interpol_ids[j],i]
        b2_give_c_interpol[j,i] = b2_give_c[pann_interpol_ids[j],i]
        b3_give_c_interpol[j,i] = b3_give_c[pann_interpol_ids[j],i]


# plot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) 

cs1 = axs[0].contour(temp_range,x2, b1_give_c_interpol, levels=np.arange(0.5,1), colors='k',antialiased = True,linestyles = 'dashed')
cs2 = axs[0].contour(temp_range,x2, post_tf_probs[0,:,:], levels=np.arange(0.5,1), colors='k',antialiased = True)
cs1.collections[0].set_label("Prior")
cs2.collections[0].set_label("Posterior")
axs[0].clabel(cs1, inline=0, fontsize=11, fmt='%1.1f')
axs[0].clabel(cs2, inline=0, fontsize=11, fmt='%1.1f')
axs[0].set_xlim(-10, 30)
axs[0].set_ylim(0, 1500)
axs[0].set_ylabel("$\mathregular{P_{ANN}}$ [mm]")
axs[0].set_xlabel("$\mathregular{T_{DJF}}$ [°C]")
axs[0].set_title("(a) "+ biome_names[0]+" probabilites",loc = "left", fontweight="bold")
axs[0].legend()

cs1 = axs[1].contour(temp_range,x2, b2_give_c_interpol, levels=np.arange(0.5,1), colors='k',antialiased = True,linestyles = 'dashed')
cs2 = axs[1].contour(temp_range,x2, post_tf_probs[1,:,:], levels=np.arange(0.5,1), colors='k',antialiased = True)
cs1.collections[0].set_label("Prior")
cs2.collections[0].set_label("Posterior")
axs[1].clabel(cs1, inline=0, fontsize=11, fmt='%1.1f')
axs[1].clabel(cs2, inline=0, fontsize=11, fmt='%1.1f')
axs[1].set_xlim(-10, 30)
axs[1].set_ylim(0, 1500)
axs[1].set_xlabel("$\mathregular{T_{DJF}}$ [°C]")
axs[1].set_title("(b) "+ biome_names[1]+" probabilites",loc = "left", fontweight="bold")
axs[1].legend()

cs1 = axs[2].contour(temp_range,x2, b3_give_c_interpol, levels=np.arange(0.5,1), colors='k',antialiased = True,linestyles = 'dashed')
cs2 = axs[2].contour(temp_range,x2, post_tf_probs[2,:,:], levels=np.arange(0.5,1), colors='k',antialiased = True)
cs1.collections[0].set_label("Prior")
cs2.collections[0].set_label("Posterior")
axs[2].clabel(cs1, inline=0, fontsize=11, fmt='%1.1f')
axs[2].clabel(cs2, inline=0, fontsize=11, fmt='%1.1f')
axs[2].set_xlim(-10, 30)
axs[2].set_ylim(0, 1500)
axs[2].set_xlabel("$\mathregular{T_{DJF}}$ [°C]")
axs[2].set_title("(c) "+ biome_names[2]+" probabilites",loc = "left", fontweight="bold")
axs[2].legend()

plt.tight_layout()
plt.savefig("plots/transfer_functions_2D_dists.pdf")
plt.close(fig)


 
##------------------------------------------------------------------------------------------------------ 
## plots of the ratio of posterior TFs to prior TFs CI sizes
##------------------------------------------------------------------------------------------------------  


ratios = np.reshape(np.array([post_process["post_temp_ci_size"]/post_process["prior_temp_ci_size"],post_process["post_pann_ci_size"]/post_process["prior_pann_ci_size"]]), (2,num_biomes))
bar_width = 0.35 # set the width of each bar

# plot
fig, ax = plt.subplots(figsize=(8, 8)) # create a figure and axes object
plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.05)

bar1 = ax.barh(np.arange(len(biome_names)), ratios[0], height=bar_width, color="orange", label="$\mathregular{T_{DJF}}$") # create the first set of bars
bar2 = ax.barh(np.arange(len(biome_names))+bar_width, ratios[1], height=bar_width, color="lightblue", label="$\mathregular{P_{ANN}}$") # create the second set of bars
ax.set_yticks(np.arange(len(biome_names))+bar_width/2) # set the y-ticks to the center of each group of bars
ax.set_yticklabels(biome_names) # set the tick labels to the biome names
ax.set_xlabel("") # set the label for the x-axis
ax.set_xlim([0, max(ratios.flatten())+0.2]) # set the x-axis limits
ax.legend(loc="lower right") # add a legend to the plot
ax.set_title("Ratio of posterior to prior CI sizes",loc = "left", fontweight="bold")
plt.savefig("plots/transfer_functions_ratio.pdf")
plt.close(fig)


##------------------------------------------------------------------------------------------------------ 
## plots of the posterior and prior climate and variance module (indipendent MH)
##------------------------------------------------------------------------------------------------------ 


temp_axes = np.linspace(min(temp_range), max(temp_range), dims)
pann_axes = np.linspace(min(pann_range), max(pann_range), dims)
expl_axes = np.linspace(0, 1, dims)

which_samples = post_process["which_samples"]

# temp
post_temp = np.empty(len(which_samples))
post_temp_all = posterior["post_temp"]
for i in np.arange(0,len(which_samples)):
    post_temp[i] = post_temp_all[which_samples[i]]
    
post_temp_dens_temp = gaussian_kde(post_temp)
post_dens_temp = post_temp_dens_temp.evaluate(temp_axes)
post_dens_temp = post_dens_temp/max(post_dens_temp)

norm_x = np.linspace(min(temp_range), max(temp_range), 1000)
prior_dens_temp = norm.pdf(norm_x,prior_recent[0],prior_recent[1])
prior_dens_temp = prior_dens_temp/max(prior_dens_temp)


# pann
post_pann = np.empty(len(which_samples))
post_pann_all = posterior["post_pann"]
for i in np.arange(0,len(which_samples)):
    post_pann[i] = post_pann_all[which_samples[i]]
    
post_temp_dens_pann = gaussian_kde(post_pann)
post_dens_pann = post_temp_dens_pann.evaluate(pann_axes)
post_dens_pann = post_dens_pann/max(post_dens_pann)


gamma_x = np.linspace(min(pann_range), max(pann_range), 1000)
prior_dens_pann = gamma.pdf(gamma_x,shape,scale=1/rate)
prior_dens_pann = prior_dens_pann/max(prior_dens_pann)


# AP/NAP
post_expl_variance = np.empty(len(which_samples))
post_expl_variance_all = posterior["post_expl_variance"]
for i in np.arange(0,len(which_samples)):
    post_expl_variance[i] = post_expl_variance_all[which_samples[i]]
    
post_temp_dens_pollen = gaussian_kde(post_expl_variance)
post_dens_pollen = post_temp_dens_pollen.evaluate(expl_axes)
post_dens_pollen = post_dens_pollen/max(post_dens_pollen)

beta_x = np.linspace(0, 1, 1000)
prior_dens_pollen = beta.pdf(beta_x,shape1,shape2)
prior_dens_pollen = prior_dens_pollen/max(prior_dens_pollen)


# plot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5)) 
plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.10)

axs[0].plot(temp_axes,post_dens_temp, color="black")
axs[0].fill_between(np.hstack([np.min(temp_axes), temp_axes]), np.hstack([0, post_dens_temp]), color="lightgrey")
axs[0].plot(norm_x, prior_dens_temp, color="orange")
axs[0].set_ylabel("Normalized density")
axs[0].set_xlabel("$\mathregular{T_{DJF}}$ [°C]")
axs[0].set_xlim(0, 20)
axs[0].set_ylim(0,1)
axs[0].set_yticks([])  
axs[0].legend(("Posterior","Prior"))
axs[0].set_title("(a) Temperature",loc = "left", fontweight="bold")


axs[1].plot(pann_axes,post_dens_pann, color="black")
axs[1].fill_between(np.hstack([np.min(pann_axes), pann_axes]), np.hstack([0, post_dens_pann]), color="lightgrey")
axs[1].plot(gamma_x, prior_dens_pann, color="orange")
axs[1].set_xlabel("$\mathregular{P_{ANN}}$ [mm]")
axs[1].set_xlim(0,1000)
axs[1].set_ylim(0,1)
axs[1].set_yticks([])  
axs[1].set_title("(b) Precipitation",loc = "left", fontweight="bold")


axs[2].plot(expl_axes,post_dens_pollen, color="black")
axs[2].fill_between(np.hstack([np.min(expl_axes), expl_axes]), np.hstack([0, post_dens_pollen]), color="lightgrey")
axs[2].plot(beta_x, prior_dens_pollen, color="orange")
axs[2].set_xlabel("$\mathregular{R^2}$")
axs[2].set_xlim(0,1)
axs[2].set_ylim(0,1)
axs[2].set_yticks([])  
axs[2].set_title("(c) Arboreal pollen",loc = "left", fontweight="bold")


plt.savefig("plots/independent_proposal_dists.pdf")
plt.close(fig)


##------------------------------------------------------------------------------------------------------ 
## age-depth transformation of arboreal pollen 
##------------------------------------------------------------------------------------------------------ 


ages_no_transfrom = age_depth_relationship["age_mean"]
ages_transfrom = age_depth_relationship["age_regular"]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11, 12)) 
plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.10)


axs[0].plot(depth,ap_depth, color="black")
axs[0].fill_between(np.hstack([np.min(depth), depth]), np.hstack([0, ap_depth]), color="lightgrey")
axs[0].set_ylabel("%")
axs[0].set_xlabel("Depth [cm]")
axs[0].set_xlim(min(depth),max(depth)) 
axs[0].set_ylim(0, 100)  
axs[0].set_title("(a) Arboreal pollen from Lake Kinneret with respect to depth",loc = "left", fontweight="bold")


axs[1].plot(ages_transfrom,ap_age, color="black")
axs[1].fill_between(np.hstack([np.min(ages_transfrom), ages_transfrom]), np.hstack([0, ap_age]), color="lightgrey")
axs[1].plot(ages_no_transfrom, ap_depth, color="orange", linewidth = 2)
axs[1].set_ylabel("%")
axs[1].set_xlabel("Age [cal a BP]")
axs[1].set_xlim(min(ages_transfrom),max(ages_transfrom)) 
axs[1].set_ylim(0, 100)  
axs[1].legend(("New age-depth-transformation","Old age-depth-transformation"))
axs[1].set_title("(b) As (a), but with respect to age",loc = "left", fontweight="bold")

plt.savefig("plots/age_depth_transform.pdf")
plt.close(fig)

