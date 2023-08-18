 

// include some libraries
# include <vector>
# include <random>

// include some additional pybind libraries
# include <pybind11/stl.h>

// use specific namespaces to improve the clarity of the code
using namespace std;
namespace py = pybind11;


//--------------------------------------------------------------------------------------------------
// CLASSES
//--------------------------------------------------------------------------------------------------

class trunc_normal {
    private:
        double r8poly_value_horner( const int &m, double c[], const double &x );
        double normal_01_cdf( const double &x );
        double normal_01_cdf_inv( const double &p );
        double r8_uniform_01( int &seed );
        double normal_01_pdf( const double &x );
    public:
        double truncnorm_pdf( const double &x, const double &mu, const double &sigma, const double &a, const double &b );
        double truncnorm_sample( const double &mu, const double &sigma, const double &a, const double &b, int &seed );
};

 
class gamma_functions {
    private:
        double kf_lgamma( double z);
        double _kf_gammap( double s, double z);
        double _kf_gammaq(double s, double z);
        float my_logf( float a);
    public:
        double kf_gammap( double s, double z);
        double kf_gammaq( double s, double z);
        float my_erfinvf( float a);
};



class Vector_operations{
    
    public:
        // function which multiplies a double to all elements of a vector
        vector<double> vector_weight_number(
            const vector<double>& input, 
            const double& weight
        );
        // function which multiplies a double to all elements of a vector
        vector<double> vector_weight_vector(
            const vector<double>& input, 
            const vector<double>& weight
        );
        // function which subtacted a double of all elements of a vector
        vector<double> vector_subtract(
            const vector<double>& input, 
            const double& subtractor
        );
        // function for the sum of a vector
        double sum_vector(const vector<double>& input);
        // function which handles input vectors of the gamma function
        vector<double> lgamma_vector(const vector<double>& input);
        // function which handles input vectors of the log function
        vector<double> log_vector(const vector<double>& input);
};
  

class Dists : public gamma_functions {
    
    public:
        // function which simulates draws from a dirichlet distribution 
        vector<double> dirichlet_sample(
            const vector<double>& alpha, 
            default_random_engine& rng
        );
        // functions for fixed proposal pdfs and for normalizing of the sampled precipitation values
        double gamma_pdf(
            const double& value,
            const double& alpha,
            const double& beta
        );     
        double gamma_cdf(
            const double& x,
            const double& shape,
            const double& rate
        ); 
        
        double beta_pdf(
            const double& value,
            const double& alpha,
            const double& beta
        ); 
        
        double norm_pdf(
            const double& value,
            const double& mean,
            const double& sigma
        ); 
        
        double norm_cdf(const double& value);
        double norm_quantiles(const double& value);
        
};



class  taxa_weights : public Dists, public Vector_operations{
    public:
        
        // samples from the dirichlet proposal density for the taxa weights
        vector<double> new_taxa_weights(
            const vector<double>& old_weights, 
            const int& dirichlet_spread, 
            default_random_engine& rng
        );
        
        // function for the calculation of the taxa weight ratios (rwMH)
        double taxa_weights_ratios(
            const vector<double>& new_weights, 
            const vector<double>& old_weights, 
            const vector<double>& jeffreys_taxa_prior
        );

};


class transfer_function : public Dists, public trunc_normal {
    public:
        
        int num_biomes;
        
        // function for sampling new climate values from the TFs and calculation of the respective ratios (rwMH)
        vector<double> tf_sampling(
            const vector<double>& old_tf_sample,
            const vector<double>& sds_tfs,
            const vector<double>& tfs_lower,
            const vector<double>& tfs_upper,
            int& seed
        );
        
        // function for calculating the probabilites of the nnet based transfer function (Pr(B|C,theta)). The climate values C should be normalized.
        vector<double> my_nnet_prediction(
            const double& tf_sample_temp,
            const double& tf_sample_pann,
            const vector<double>& wts_in_hidden,
            const vector<double>& wts_hidden_out,
            const vector<double>& wts_bias_hidden,
            const vector<double>& wts_bias_out
        );
        
        // function for sampling new climate values from the TFs and calculation of the respective ratios (rwMH)
        double tf_samples_ratios(
            const vector<double>& new_tf_sample, 
            const vector<double>& old_tf_sample,
            const vector<double>& new_tf_sample_norm, 
            const vector<double>& old_tf_sample_norm, 
            const vector<double>& sds_tfs, 
            const vector<double>& tfs_lower, 
            const vector<double>& tfs_upper, 
            const vector<double>& wts_in_hidden, 
            const vector<double>& wts_hidden_out, 
            const vector<double>& wts_bias_hidden, 
            const vector<double>& wts_bias_out
        );
        
        // normalize the tf samples for checking their probability in ML model space 
        vector<double> normalize_tf_samples(
            const vector<double>& tf_samples,
            const vector<double>& normal_params
        );        
};



class reconstruction {
    private:
        
        // explained variance: for the comparison of the reconstruction with a reference curve
        double expl_variance_cpp(
            const int& n, 
            const vector<double>& reconst, 
            const vector<double>& reference
        );
        
    public:
        
        int length_age;
        
        // caluclation of the climate reconstruction and select those data for the calibration step: recent climate and explained variance with the reference curve
        vector<double> reconst_ap_pann(
            const vector<double>& biome_ratios_age, 
            const vector<double>& reference, 
            const vector<double>& tf_sample, 
            const int& num_biomes
        );
};



//--------------------------------------------------------------------------------------------------
// RCPP EXPORT FUNCTIONS
//--------------------------------------------------------------------------------------------------



// function for calculating the probabilites of the nnet based transfer function (Pr(B|C,theta)). The climate values C should be normalized.
vector<double> my_nnet_prediction(
    const double& tf_sample_temp,
    const double& tf_sample_pann,
    const vector<double>& wts_in_hidden,
    const vector<double>& wts_hidden_out,
    const vector<double>& wts_bias_hidden,
    const vector<double>& wts_bias_out
) {
    const int num_target = 4; // three known biomes and one unspecified "biome"
    const int num_hidden = 6;
    vector<double> output_sum_scaled(num_target, 0.0);
    double output_sum_norm = 0.0;

    for (int i = 0; i < num_target; ++i) {
        double output_sum = 0.0;
        for (int j = 0; j < num_hidden; ++j) {
            // calculate the weighted sum of inputs for each hidden unit and shift them according to the bias
            double hidden_sums = wts_in_hidden[j] * tf_sample_temp + wts_in_hidden[j + num_hidden] * tf_sample_pann + wts_bias_hidden[j];
            // apply the activation function (sigmoid) to the hidden unit sums
            double hidden_outputs = 1 / (1 + exp(-hidden_sums));
            // calculate the weighted sum of hidden weights for the output units
            output_sum += hidden_outputs * wts_hidden_out[i * num_hidden + j];
        }
        // preparation of softmax output
        output_sum_scaled[i] = exp(output_sum + wts_bias_out[i]);
        output_sum_norm += output_sum_scaled[i];
    }
    // normalization
    vector<double> softmax(num_target, 0.0);
    for (int i = 0; i < num_target; ++i) {
        softmax[i] = output_sum_scaled[i] / output_sum_norm;
    }

    return softmax;
}



// SOME MATRIX OPERATIONS
// transform the weighted taxa spectrum to biome probability densities
vector<double> spectrum_to_biome_assign(
    const vector<double>& spectrum, 
    const vector<double>& weights, 
    const vector<double>& biome_assign, 
    const int& length_age, 
    const int& num_taxa, 
    const int& num_biomes
){
    
    // weight the taxa spectrum (each taxa with the specific weight)
    vector<double> weighted_spectrum(length_age*num_taxa);
    for (int j = 0; j < num_taxa; j++) {
        for (int i = 0; i < length_age; i++) {
            weighted_spectrum[(length_age*j)+i] = spectrum[(length_age*j)+i]*weights[j];
        }
    }

    // convolution of the weighted spectrum with the biome assigns
    vector<double> biome_given_age(length_age*num_biomes);
    for (int i = 0; i < length_age; i++) {
        for (int j = 0; j < num_biomes; j++) {
            for (int k = 0; k < num_taxa; k++) {
                biome_given_age[(length_age*j)+i] += weighted_spectrum[(length_age*k)+i] * biome_assign[(num_taxa*j)+k];
            }
        }
    }
    
    // norm the biomes given age (pdf)
    vector<double> biome_given_age_norm(length_age*num_biomes);
    vector<double> row_sums(length_age,0.0);
    for (int i = 0; i < length_age; i++) {
        for (int j = 0; j < num_biomes; j++) {
            row_sums[i] += biome_given_age[(length_age*j)+i];
        }
        for (int j = 0; j < num_biomes; j++) {
            biome_given_age_norm[(length_age*j)+i] = biome_given_age[(length_age*j)+i] / row_sums[i];
        }
    } 
    
    return biome_given_age_norm;
}


// calculates the columnwise sums of a matrix
vector<double> my_colSums(
    const vector<double>& x, 
    const int& nr, 
    const int& nc
) {
    vector<double> out(nc);
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < nr; j++) {
            out[i] += x[(nr*i)+j];
        }
    }
    return out;
}

// calculates the rowwise sums of a matrix
vector<double> my_rowSums(
    const vector<double>& x, 
    const int& nr, 
    const int& nc
) {
    vector<double> out(nr);
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            out[i] += x[(nr*j)+i];
        }
    }
    return out;
}


// function for the MCMC execution
vector<vector<double>> mcmc_execution( 
    vector<vector<double>> taxa_weights_container,
    vector<vector<double>> tf_sample,
    vector<vector<double>> prior,
    vector<vector<double>> core_info,
    vector<vector<double>> proposal_params, 
    vector<vector<double>> tf_info, 
    vector<vector<int>> sampling_info
){ 
    
    // Basic declaration:
    
    // create class objects
    Dists dists;
    taxa_weights tw;
    transfer_function tf;
    reconstruction rec;
    
    // core data
    tf.num_biomes = static_cast<int>(core_info[0][0]);
    int num_taxa = static_cast<int>(core_info[1][0]);
    rec.length_age = static_cast<int>(core_info[2][0]);
    vector<double> ap_age = core_info[3]; 
    vector<double> taxa_spectrum_age = core_info[4];
    vector<double> biomes_assign = core_info[5]; 
    
    // starting MCMC values
    vector<vector<double>> out_taxa_weights = taxa_weights_container;
    vector<vector<double>> out_tf_sample = tf_sample; 
    vector<double> out_expl_variance = prior[0];
    vector<double> out_recent_temp = prior[1];
    vector<double> out_recent_pann = prior[2];
    vector<double> acceptance = prior[3];
     
    vector<double> old_taxa_weights(num_taxa);
    vector<double> old_tf_sample(2*tf.num_biomes);
    double old_expl_variance;
    double old_recent_temp;
    double old_recent_pann;

    // for the weight updates
    int dirichlet_spread = static_cast<int>(proposal_params[0][0]);
    vector<double> jeffreys_taxa_prior = proposal_params[1];

    
    // for the TF updates
    vector<double> wts_in_hidden = tf_info[0];
    vector<double> wts_hidden_out = tf_info[1];
    vector<double> wts_bias_hidden = tf_info[2];
    vector<double> wts_bias_out = tf_info[3];
    vector<double> normal_params = tf_info[4];
    vector<double> sds_tfs = tf_info[5];
    vector<double> tfs_lower = tf_info[6];
    vector<double> tfs_upper = tf_info[7];


    // for the calibration updates
    double shape1 = proposal_params[2][0];
    double shape2 = proposal_params[3][0];
    
    double mean_temp = proposal_params[4][0];
    double sd_temp = proposal_params[5][0];
    
    double shape_pann = proposal_params[6][0];
    double rate_pann = proposal_params[7][0];

    // sampling information
    int sample_length = static_cast<int>(sampling_info[0][0]);
    vector<int> seed_samples = sampling_info[1];

    // for random sample generation 
    default_random_engine rng(seed_samples[0]);

    // draw from uniform distribution within the acceptance-rejection step
    uniform_real_distribution<double> uniform_sample(0.0, 1.0);
    
    //--------------------------------------------------------------------------------------------------
    // MCMC loop
    //--------------------------------------------------------------------------------------------------
    
    // declaration of objects inside MCMC loop
    vector<double> new_weights; 
    vector<double> biome_ratios_age;
    vector<double> new_tf_sample;
    vector<double> new_tf_sample_norm;
    vector<double> old_tf_sample_norm;
    vector<double> new_reconst_params;
      
    double taxa_weigths_ratio,
    transfer_ratio,
    propose_variance,
    start_variance,
    variance_ratio,
    propose_temp,
    start_temp,
    temp_ratio,
    propose_pann,
    start_pann,
    pann_ratio,
    climate_ratio,
    acc_prob;
    
     

    
    for(int i=1; i<sample_length; i++) {
        
        //--------------------------------------------------------------------------------------------------
        // initialization
        //--------------------------------------------------------------------------------------------------
        
        old_taxa_weights = out_taxa_weights[i-1]; 
        old_tf_sample = out_tf_sample[i-1];
        old_expl_variance = out_expl_variance[i-1];
        old_recent_temp = out_recent_temp[i-1];
        old_recent_pann = out_recent_pann[i-1];
        
        //--------------------------------------------------------------------------------------------------
        // taxa weight module: rwMH
        //--------------------------------------------------------------------------------------------------
        
        // sample new taxa weights and calculate the respective taxa weight ratios for the rwMH
        new_weights = tw.new_taxa_weights(old_taxa_weights,dirichlet_spread,rng);
        taxa_weigths_ratio = tw.taxa_weights_ratios(new_weights,old_taxa_weights,jeffreys_taxa_prior); 
        
        // create new taxa spectrum based on the updated weights and new biome probs based on the updated taxa spectrum and norm them to densities
        biome_ratios_age = spectrum_to_biome_assign(taxa_spectrum_age,new_weights,biomes_assign,rec.length_age,num_taxa,tf.num_biomes);

        //--------------------------------------------------------------------------------------------------
        // transfer function module: rwMH
        //--------------------------------------------------------------------------------------------------

        // sample new climate values from the TFs 
        new_tf_sample = tf.tf_sampling(old_tf_sample,sds_tfs,tfs_lower,tfs_upper,seed_samples[i]);
        
        // normalize the tf samples for the nnet model
        new_tf_sample_norm = tf.normalize_tf_samples(new_tf_sample,normal_params);
        old_tf_sample_norm = tf.normalize_tf_samples(old_tf_sample,normal_params);
        
        // calculate the respective ratios for the rwMH
        transfer_ratio = tf.tf_samples_ratios(new_tf_sample,
                                           old_tf_sample,
                                           new_tf_sample_norm,
                                           old_tf_sample_norm,
                                           sds_tfs,tfs_lower,
                                           tfs_upper,
                                           wts_in_hidden,
                                           wts_hidden_out,
                                           wts_bias_hidden,
                                           wts_bias_out);

        //--------------------------------------------------------------------------------------------------
        // reconstruction module: independent MH 
        //--------------------------------------------------------------------------------------------------
        
        // reconstruction
        new_reconst_params = rec.reconst_ap_pann(biome_ratios_age,ap_age,new_tf_sample,tf.num_biomes);

        // ratios
        // proxy pool module: independent MH
        propose_variance = dists.beta_pdf(new_reconst_params[0], shape1,shape2);
        start_variance = dists.beta_pdf(old_expl_variance, shape1,shape2);
        variance_ratio = propose_variance/start_variance;
        
        // climate module: independent MH
        propose_temp = dists.norm_pdf(new_reconst_params[1], mean_temp,sd_temp);
        start_temp = dists.norm_pdf(old_recent_temp, mean_temp,sd_temp);
        temp_ratio = propose_temp/start_temp;
        
        propose_pann = dists.gamma_pdf(new_reconst_params[2], shape_pann,rate_pann);
        start_pann = dists.gamma_pdf(old_recent_pann, shape_pann,rate_pann);
        pann_ratio = propose_pann/start_pann;

        climate_ratio = temp_ratio * pann_ratio;

        //--------------------------------------------------------------------------------------------------
        // acceptance 
        //--------------------------------------------------------------------------------------------------
    
        acc_prob = variance_ratio * climate_ratio * taxa_weigths_ratio * transfer_ratio; 

        
        if(uniform_sample(rng) < acc_prob){
            out_taxa_weights[i] = new_weights; 
            out_tf_sample[i] = new_tf_sample;
            out_expl_variance[i] = new_reconst_params[0];
            out_recent_temp[i] = new_reconst_params[1];
            out_recent_pann[i] = new_reconst_params[2];
            acceptance[i] = 1;
        }else{
            out_taxa_weights[i] = old_taxa_weights;
            out_tf_sample[i] = old_tf_sample;
            out_expl_variance[i]  = old_expl_variance;
            out_recent_temp[i] =  old_recent_temp;
            out_recent_pann[i] =  old_recent_pann;
            acceptance[i] = 0;
        }
    }
  
    //--------------------------------------------------------------------------------------------------
    // output
    //--------------------------------------------------------------------------------------------------

    
    // transform the vectors of vectors to vectors
    int num_tfs = 2*tf.num_biomes;
    vector<double> out_tf_sample_vec(sample_length*num_tfs);
    for(int i=0; i<sample_length; i++) {
        for(int j=0; j<num_tfs; j++) {
            out_tf_sample_vec[num_tfs*i+j] = out_tf_sample[i][j];
        }
    }
    
    vector<double> out_taxa_weights_vec(sample_length*num_taxa);
    for(int i=0; i<sample_length; i++) {
        for(int j=0; j<num_taxa; j++) {
            out_taxa_weights_vec[num_taxa*i+j] = out_taxa_weights[i][j];
        }
    }
    
    // summarize the output 
    vector<vector<double>> posterior(6);
    posterior[0] = acceptance;
    posterior[1] = out_tf_sample_vec;
    posterior[2] = out_taxa_weights_vec;
    posterior[3] = out_expl_variance;
    posterior[4] = out_recent_temp;
    posterior[5] = out_recent_pann;
    
    return posterior;  
 
}



// bind the needed functions
PYBIND11_MODULE(pycpp, m) {
    m.doc() = "pybind11 pycpp plugin"; 
    m.def("my_nnet_prediction", &my_nnet_prediction, "Function which predicts the biomes given climate values");
    m.def("spectrum_to_biome_assign", &spectrum_to_biome_assign, "Function which includes matrix normalization and multipication");
    m.def("mcmc_execution", &mcmc_execution, "Function which conducts the MCMC sampling");
}



//--------------------------------------------------------------------------------------------------
// TRUNCATED NORMAL
//--------------------------------------------------------------------------------------------------

// some functions for the dirichlet distribution from: https://people.math.sc.edu/Burkardt/ 

double trunc_normal::r8poly_value_horner ( const int &m, double c[], const double &x ){
    int i;
    double value;
    value = c[m];
    for ( i = m - 1; 0 <= i; i-- ){
        value = value * x + c[i];
    }
    return value;
}

double trunc_normal::normal_01_cdf ( const double &x ){
    double a1 = 0.398942280444;
    double a2 = 0.399903438504;
    double a3 = 5.75885480458;
    double a4 = 29.8213557808;
    double a5 = 2.62433121679;
    double a6 = 48.6959930692;
    double a7 = 5.92885724438;
    double b0 = 0.398942280385;
    double b1 = 3.8052E-08;
    double b2 = 1.00000615302;
    double b3 = 3.98064794E-04;
    double b4 = 1.98615381364;
    double b5 = 0.151679116635;
    double b6 = 5.29330324926;
    double b7 = 4.8385912808;
    double b8 = 15.1508972451;
    double b9 = 0.742380924027;
    double b10 = 30.789933034;
    double b11 = 3.99019417011;
    double cdf;
    double q;
    double y;
    
    //  |X| <= 1.28.
    if ( fabs ( x ) <= 1.28 ){
        y = 0.5 * x * x;

        q = 0.5 - fabs ( x ) * ( a1 - a2 * y / ( y + a3 - a4 / ( y + a5+ a6 / ( y + a7 ) ) ) );
        
    //  1.28 < |X| <= 12.7
    }else if ( fabs ( x ) <= 12.7 ){
        y = 0.5 * x * x;
        q = exp ( - y ) * b0 / ( fabs ( x ) - b1 + b2  / ( fabs ( x ) + b3 + b4  / ( fabs ( x ) - b5+ b6  / ( fabs ( x ) + b7 - b8  / ( fabs ( x ) + b9 + b10 / ( fabs ( x ) + b11 ) ) ) ) ) );

    //  12.7 < |X|
    }else{
        q = 0.0;
    }
    
    //  Take account of negative X.
    if ( x < 0.0 ){
        cdf = q;
    }else{
        cdf = 1.0 - q;
    }

    return cdf;
}

double trunc_normal::normal_01_cdf_inv ( const double &p ){
    
    double a[8] = {
        3.3871328727963666080,     1.3314166789178437745E+2,
        1.9715909503065514427E+3,  1.3731693765509461125E+4,
        4.5921953931549871457E+4,  6.7265770927008700853E+4,
        3.3430575583588128105E+4,  2.5090809287301226727E+3 };
    double b[8] = {
        1.0,                       4.2313330701600911252E+1,
        6.8718700749205790830E+2,  5.3941960214247511077E+3,
        2.1213794301586595867E+4,  3.9307895800092710610E+4,
        2.8729085735721942674E+4,  5.2264952788528545610E+3 };
    double c[8] = {
        1.42343711074968357734,     4.63033784615654529590,
        5.76949722146069140550,     3.64784832476320460504,
        1.27045825245236838258,     2.41780725177450611770E-1,
        2.27238449892691845833E-2,  7.74545014278341407640E-4 };
    double const1 = 0.180625;
    double const2 = 1.6;
    double d[8] = {
        1.0,                        2.05319162663775882187,
        1.67638483018380384940,     6.89767334985100004550E-1,
        1.48103976427480074590E-1,  1.51986665636164571966E-2,
        5.47593808499534494600E-4,  1.05075007164441684324E-9 };
    double e[8] = {
        6.65790464350110377720,     5.46378491116411436990,
        1.78482653991729133580,     2.96560571828504891230E-1,
        2.65321895265761230930E-2,  1.24266094738807843860E-3,
        2.71155556874348757815E-5,  2.01033439929228813265E-7 };
    double f[8] = {
        1.0,                        5.99832206555887937690E-1,
        1.36929880922735805310E-1,  1.48753612908506148525E-2,
        7.86869131145613259100E-4,  1.84631831751005468180E-5,
        1.42151175831644588870E-7,  2.04426310338993978564E-15 };
    double q;
    double r;
    double split1 = 0.425;
    double split2 = 5.0;
    double value;

    if ( p <= 0.0 ){
        value = - HUGE_VAL;
        return value;
    }

    if ( 1.0 <= p ){
        value = HUGE_VAL;
        return value;
    }

    q = p - 0.5;

    if ( fabs ( q ) <= split1 ){
        r = const1 - q * q;
        value = q * r8poly_value_horner ( 7, a, r ) / r8poly_value_horner ( 7, b, r );
    }else{
        if ( q < 0.0 ){
            r = p;
        }else{
            r = 1.0 - p;
        }

        if ( r <= 0.0 ){
            value = HUGE_VAL;
        }else{
            r = sqrt ( - log ( r ) );

            if ( r <= split2 ){
                r = r - const2;
                value = r8poly_value_horner ( 7, c, r ) / r8poly_value_horner ( 7, d, r );
            }else{
                r = r - split2;
                value = r8poly_value_horner ( 7, e, r ) / r8poly_value_horner ( 7, f, r );
            }
        }

        if ( q < 0.0 ){
            value = - value;
        }

    }

    return value;
}

double trunc_normal::r8_uniform_01 ( int &seed ){
    int k;
    double r;
    k = seed / 127773;
    seed = 16807 * ( seed - k * 127773 ) - k * 2836;
    if ( seed < 0 ){
        seed = seed + 2147483647;
    }
    r = ( double ) ( seed ) * 4.656612875E-10;

    return r;
}

double trunc_normal::normal_01_pdf ( const double &x ){
    double pdf;
    const double r8_pi = 3.14159265358979323;
    pdf = exp ( -0.5 * x * x ) / sqrt ( 2.0 * r8_pi );
    return pdf;
} 

double trunc_normal::truncnorm_pdf ( const double &x, const double &mu, const double &sigma, const double &a, const double &b ){
    double alpha;
    double alpha_cdf;
    double beta;
    double beta_cdf;
    double pdf;
    double xi;
    double xi_pdf;

    if ( x < a ){
        pdf = 0.0;
    }else if ( x <= b ){
        alpha = ( a - mu ) / sigma;
        beta = ( b - mu ) / sigma;
        xi = ( x - mu ) / sigma;

        alpha_cdf = normal_01_cdf ( alpha );
        beta_cdf = normal_01_cdf ( beta );
        xi_pdf = normal_01_pdf ( xi );

        pdf = xi_pdf / ( beta_cdf - alpha_cdf ) / sigma;
    }
    else{
        pdf = 0.0;
    }
    
    return pdf;
}


double trunc_normal::truncnorm_sample ( const double &mu, const double &sigma, const double &a, const double &b, int &seed ){
    double alpha;
    double alpha_cdf;
    double beta;
    double beta_cdf;
    double u;
    double x;
    double xi;
    double xi_cdf;

    alpha = ( a - mu ) / sigma;
    beta = ( b - mu ) / sigma;

    alpha_cdf = normal_01_cdf ( alpha );
    beta_cdf = normal_01_cdf ( beta );

    u = r8_uniform_01 ( seed );
    xi_cdf = alpha_cdf + u * ( beta_cdf - alpha_cdf );
    xi = normal_01_cdf_inv ( xi_cdf );

    x = mu + sigma * xi;

    return x;
}



//--------------------------------------------------------------------------------------------------
// GAMMA FUNCTIONS
//--------------------------------------------------------------------------------------------------

// some gamma functions from:  https://github.com/lh3/samtools/blob/master/bcftools/kfunc.c   
        
/* Log gamma function
* \log{\Gamma(z)}
* AS245, 2nd algorithm, http://lib.stat.cmu.edu/apstat/245
*/

double gamma_functions::kf_lgamma( double z){
    double x = 0;
    x += 0.1659470187408462e-06 / (z+7);
    x += 0.9934937113930748e-05 / (z+6);
    x -= 0.1385710331296526     / (z+5);
    x += 12.50734324009056      / (z+4);
    x -= 176.6150291498386      / (z+3);
    x += 771.3234287757674      / (z+2);
    x -= 1259.139216722289      / (z+1);
    x += 676.5203681218835      / z;
    x += 0.9999999999995183;
    return log(x) - 5.58106146679532777 - z + (z-0.5) * log(z+6.5);
}



/* The following computes regularized incomplete gamma functions.
* Formulas are taken from Wiki, with additional input from Numerical
* Recipes in C (for modified Lentz's algorithm) and AS245
* (http://lib.stat.cmu.edu/apstat/245).
*
* A good online calculator is available at:
*
*   http://www.danielsoper.com/statcalc/calc23.aspx
*
* It calculates upper incomplete gamma function, which equals
* kf_gammaq(s,z)*tgamma(s).
*/

#define KF_GAMMA_EPS 1e-14
#define KF_TINY 1e-290

// regularized lower incomplete gamma function, by series expansion
double gamma_functions::_kf_gammap( double s, double z){
    double sum, x;
    int k;
    for (k = 1, sum = x = 1.; k < 100; ++k) {
        sum += (x *= z / (s + k));
        if (x / sum < KF_GAMMA_EPS) break;
    }
    return exp(s * log(z) - z - kf_lgamma(s + 1.) + log(sum));
}

// regularized upper incomplete gamma function, by continued fraction
double gamma_functions::_kf_gammaq(double s, double z){
    int j;
    double C, D, f;
    f = 1. + z - s; C = f; D = 0.;
    // Modified Lentz's algorithm for computing continued fraction
    // See Numerical Recipes in C, 2nd edition, section 5.2
    for (j = 1; j < 100; ++j) {
        double a = j * (s - j), b = (j<<1) + 1 + z - s, d;
        D = b + a * D;
        if (D < KF_TINY) D = KF_TINY;
        C = b + a / C;
        if (C < KF_TINY) C = KF_TINY;
        D = 1. / D;
        d = C * D;
        f *= d;
        if (fabs(d - 1.) < KF_GAMMA_EPS) break;
    }
    return exp(s * log(z) - z - kf_lgamma(s) - log(f));
}

double gamma_functions::kf_gammap( double s, double z){
    return z <= 1. || z < s? gamma_functions::_kf_gammap(s, z) : 1. - gamma_functions::_kf_gammaq(s, z);
}

double gamma_functions::kf_gammaq( double s, double z){
    return z <= 1. || z < s? 1. - gamma_functions::_kf_gammap(s, z) : gamma_functions::_kf_gammaq(s, z);
}



// inverse error function from: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c

// #include <math.h>
//         float my_logf (float);

/* compute natural logarithm with a maximum error of 0.85089 ulp */
float gamma_functions::my_logf ( float a){
    float i, m, r, s, t;
    int e;

    m = frexpf (a, &e);
    if (m < 0.666666667f) { // 0x1.555556p-1
        m = m + m;
        e = e - 1;
    }
    i = (float)e;
    /* m in [2/3, 4/3] */
    m = m - 1.0f;
    s = m * m;
    /* Compute log1p(m) for m in [-1/3, 1/3] */
    r =             -0.130310059f;  // -0x1.0ae000p-3
    t =              0.140869141f;  //  0x1.208000p-3
    r = fmaf (r, s, -0.121484190f); // -0x1.f19968p-4
    t = fmaf (t, s,  0.139814854f); //  0x1.1e5740p-3
    r = fmaf (r, s, -0.166846052f); // -0x1.55b362p-3
    t = fmaf (t, s,  0.200120345f); //  0x1.99d8b2p-3
    r = fmaf (r, s, -0.249996200f); // -0x1.fffe02p-3
    r = fmaf (t, m, r);
    r = fmaf (r, m,  0.333331972f); //  0x1.5554fap-2
    r = fmaf (r, m, -0.500000000f); // -0x1.000000p-1
    r = fmaf (r, s, m);
    r = fmaf (i,  0.693147182f, r); //  0x1.62e430p-1 // log(2)
    if (!((a > 0.0f) && (a <= 3.40282346e+38f))) { // 0x1.fffffep+127
        r = a + a;  // silence NaNs if necessary
        if (a  < 0.0f) r = ( 0.0f / 0.0f); //  NaN
        if (a == 0.0f) r = (-1.0f / 0.0f); // -Inf
    }
    return r;
}


/* compute inverse error functions with maximum error of 2.35793 ulp */
float gamma_functions::my_erfinvf ( float a){
    float p, r, t;
    t = fmaf (a, 0.0f - a, 1.0f);
    t = my_logf (t);
    if (fabsf(t) > 6.125f) { // maximum ulp error = 2.35793
        p =              3.03697567e-10f; //  0x1.4deb44p-32 
        p = fmaf (p, t,  2.93243101e-8f); //  0x1.f7c9aep-26 
        p = fmaf (p, t,  1.22150334e-6f); //  0x1.47e512p-20 
        p = fmaf (p, t,  2.84108955e-5f); //  0x1.dca7dep-16 
        p = fmaf (p, t,  3.93552968e-4f); //  0x1.9cab92p-12 
        p = fmaf (p, t,  3.02698812e-3f); //  0x1.8cc0dep-9 
        p = fmaf (p, t,  4.83185798e-3f); //  0x1.3ca920p-8 
        p = fmaf (p, t, -2.64646143e-1f); // -0x1.0eff66p-2 
        p = fmaf (p, t,  8.40016484e-1f); //  0x1.ae16a4p-1 
    } else { // maximum ulp error = 2.35002
        p =              5.43877832e-9f;  //  0x1.75c000p-28 
        p = fmaf (p, t,  1.43285448e-7f); //  0x1.33b402p-23 
        p = fmaf (p, t,  1.22774793e-6f); //  0x1.499232p-20 
        p = fmaf (p, t,  1.12963626e-7f); //  0x1.e52cd2p-24 
        p = fmaf (p, t, -5.61530760e-5f); // -0x1.d70bd0p-15 
        p = fmaf (p, t, -1.47697632e-4f); // -0x1.35be90p-13 
        p = fmaf (p, t,  2.31468678e-3f); //  0x1.2f6400p-9 
        p = fmaf (p, t,  1.15392581e-2f); //  0x1.7a1e50p-7 
        p = fmaf (p, t, -2.32015476e-1f); // -0x1.db2aeep-3 
        p = fmaf (p, t,  8.86226892e-1f); //  0x1.c5bf88p-1 
    }
    r = a * p;
    return r;
}



//--------------------------------------------------------------------------------------------------
// VECTOR OPERATIONS
//--------------------------------------------------------------------------------------------------

        
// function which multiplies a double to all elements of a vector
vector<double> Vector_operations::vector_weight_number(
    const vector<double>& input, 
    const double& weight
){
    int k = input.size();
    vector<double> output_vec(k);
    for (int i = 0; i < k; i++) {
        output_vec[i] = input[i] * weight;
    }
    return output_vec;
}
    
// function which multiplies a double to all elements of a vector
vector<double> Vector_operations::vector_weight_vector(
    const vector<double>& input, 
    const vector<double>& weight
){
    int k = input.size();
    vector<double> output_vec(k);
    for (int i = 0; i < k; i++) {
        output_vec[i] = input[i] * weight[i];
    }
    return output_vec;
}
    
// function which subtacted a double of all elements of a vector
vector<double> Vector_operations::vector_subtract(
    const vector<double>& input, 
    const double& subtractor
){
    int k = input.size();
    vector<double> output_vec(k);
    for (int i = 0; i < k; i++) {
        output_vec[i] = input[i] - subtractor;
    }
    return output_vec;
}

// function for the sum of a vector
double Vector_operations::sum_vector(const vector<double>& input){
    
    int k = input.size();
    double output = 0;
    for (int i = 0; i < k; i++) {
        output += input[i];
    }
    return output;
}

// function which handles input vectors of the gamma function
vector<double> Vector_operations::lgamma_vector(const vector<double>& input){
    int k = input.size();
    vector<double> output(k);
    for (int i = 0; i < k; i++) {
        output[i] = lgamma(input[i]);
    }
    return output;
}

// function which handles input vectors of the log function
vector<double> Vector_operations::log_vector(const vector<double>& input){
    int k = input.size();
    vector<double> output(k);
    for (int i = 0; i < k; i++) {
        output[i] = log(input[i]);
    }
    return output;
}





//--------------------------------------------------------------------------------------------------
// DISTS
//--------------------------------------------------------------------------------------------------



// function which simulates draws from a dirichlet distribution 
vector<double> Dists::dirichlet_sample(
    const vector<double>& alpha, 
    default_random_engine& rng
) {
    int k = alpha.size();
    vector<double> y(k);
    
    double sum = 0;
    for (int i = 0; i < k; i++) {
        double alpha_i = alpha[i];
        gamma_distribution<double> gamma_dist(alpha_i,1);
        y[i] = gamma_dist(rng);
        sum += y[i];
    }

    for (int i = 0; i < k; i++) {
        y[i] /= sum;
    }

    return y;
}


// functions for fixed proposal pdfs and for normalizing of the sampled precipitation values
double Dists::gamma_pdf(
    const double& value,
    const double& alpha,
    const double& beta
) {
    return( pow(beta, alpha)*pow(value, (alpha-1))*pow(M_E, (-1*beta*value))/tgamma(alpha) );
}

double Dists::gamma_cdf(
    const double& x,
    const double& shape,
    const double& rate
) {
    return kf_gammap(shape,x*rate)/ kf_gammaq(shape,0) ;
}

double Dists::beta_pdf(
    const double& value,
    const double& alpha,
    const double& beta
) {
    return( pow(value, alpha-1)*pow(1-value, beta-1) / (tgamma(alpha)*tgamma(beta)/tgamma(alpha+beta)) );
}

double Dists::norm_pdf(
    const double& value,
    const double& mean,
    const double& sigma
) {
    return( 1/(sigma*sqrt(2*M_PI))*pow(M_E, -((value - mean)*(value - mean))/(2*sigma*sigma)) );
}

// considers standard normal distribution
double Dists::norm_cdf(const double& value){
    return 0.5 * erfc(-value * M_SQRT1_2);
}

// considers standard normal distribution
double Dists::norm_quantiles(const double& value){
    return M_SQRT2 * my_erfinvf(2*value-1);
}



//--------------------------------------------------------------------------------------------------
// TAXA WEIGHTS
//--------------------------------------------------------------------------------------------------


// samples from the dirichlet proposal density for the taxa weights
vector<double> taxa_weights::new_taxa_weights(
    const vector<double>& old_weights, 
    const int& dirichlet_spread, 
    default_random_engine& rng
) {
    double min_weight;
    int k = old_weights.size();
    vector<double> new_weights(k);
    vector<double> dirichlet_input(k);
    dirichlet_input = vector_weight_number(old_weights,dirichlet_spread);
    
    int check = 0;
    while (check == 0) {
        // call the dirichlet_sample function to sample new weights.
        new_weights = dirichlet_sample(dirichlet_input,rng);
        // check if all values of the new weights are greater than 0.
        min_weight = *min_element(new_weights.begin(), new_weights.end());
        if (min_weight > 0) {
            check = 1;
        }
    }

    return new_weights;
}


// function for the calculation of the taxa weight ratios (rwMH)
double taxa_weights::taxa_weights_ratios(
    const vector<double>& new_weights, 
    const vector<double>& old_weights, 
    const vector<double>& jeffreys_taxa_prior
){

    // calculate weight ratios
    double treat_jeffrey_prior = lgamma(sum_vector(jeffreys_taxa_prior)) - sum_vector(lgamma_vector(jeffreys_taxa_prior));
    vector<double> log_new_w = log_vector(new_weights);
    vector<double> log_old_w = log_vector(old_weights);
    
    double weights_ratio1 = treat_jeffrey_prior + sum_vector(vector_weight_vector(vector_subtract(jeffreys_taxa_prior,1),log_new_w));
    weights_ratio1 = exp(weights_ratio1);
    
    double weights_ratio2 = treat_jeffrey_prior + sum_vector(vector_weight_vector(vector_subtract(jeffreys_taxa_prior,1),log_old_w));
    weights_ratio2 = exp(weights_ratio2);
    
    double proposal_ratio1 = lgamma(sum_vector(new_weights)) - sum_vector(lgamma_vector(new_weights)) + sum_vector(vector_weight_vector(vector_subtract(new_weights,1),log_old_w));
    proposal_ratio1 = exp(proposal_ratio1);
    
    double proposal_ratio2 = lgamma(sum_vector(old_weights)) - sum_vector(lgamma_vector(old_weights)) + sum_vector(vector_weight_vector(vector_subtract(old_weights,1),log_new_w));
    proposal_ratio2 = exp(proposal_ratio2);
    
    double new_weights_ratios = (weights_ratio1/weights_ratio2) * (proposal_ratio1/proposal_ratio2);
    
    return new_weights_ratios;
} 



//--------------------------------------------------------------------------------------------------
// TRANSFER FUNCTION
//--------------------------------------------------------------------------------------------------


// function for sampling new climate values from the TFs and calculation of the respective ratios (rwMH)
vector<double> transfer_function::tf_sampling(
    const vector<double>& old_tf_sample,
    const vector<double>& sds_tfs,
    const vector<double>& tfs_lower,
    const vector<double>& tfs_upper,
    int& seed
){ 

    // sample new tf data: proposals are truncated normal densities with parameters based on the TFs
    int n = old_tf_sample.size();
    vector<double> new_tf_sample(n);
    for(int j=0; j<n; j++) {
        new_tf_sample[j] = truncnorm_sample(old_tf_sample[j], sds_tfs[j], tfs_lower[j], tfs_upper[j], seed);
    }
    
    return new_tf_sample;
} 


// function for calculating the probabilites of the nnet based transfer function (Pr(B|C,theta)). The climate values C should be normalized.
vector<double> transfer_function::my_nnet_prediction(
    const double& tf_sample_temp,
    const double& tf_sample_pann,
    const vector<double>& wts_in_hidden,
    const vector<double>& wts_hidden_out,
    const vector<double>& wts_bias_hidden,
    const vector<double>& wts_bias_out
) {
    const int num_target = 4; // three known biomes and one unspecified "biome"
    const int num_hidden = 6;
    vector<double> output_sum_scaled(num_target, 0.0);
    double output_sum_norm = 0.0;

    for (int i = 0; i < num_target; ++i) {
        double output_sum = 0.0;
        for (int j = 0; j < num_hidden; ++j) {
            // calculate the weighted sum of inputs for each hidden unit and shift them according to the bias
            double hidden_sums = wts_in_hidden[j] * tf_sample_temp + wts_in_hidden[j + num_hidden] * tf_sample_pann + wts_bias_hidden[j];
            // apply the activation function (sigmoid) to the hidden unit sums
            double hidden_outputs = 1 / (1 + exp(-hidden_sums));
            // calculate the weighted sum of hidden weights for the output units
            output_sum += hidden_outputs * wts_hidden_out[i * num_hidden + j];
        }
        // preparation of softmax output
        output_sum_scaled[i] = exp(output_sum + wts_bias_out[i]);
        output_sum_norm += output_sum_scaled[i];
    }
    // normalization
    vector<double> softmax(num_target, 0.0);
    for (int i = 0; i < num_target; ++i) {
        softmax[i] = output_sum_scaled[i] / output_sum_norm;
    }

    return softmax;
}

// function for sampling new climate values from the TFs and calculation of the respective ratios (rwMH)
double transfer_function::tf_samples_ratios(
    const vector<double>& new_tf_sample, 
    const vector<double>& old_tf_sample,
    const vector<double>& new_tf_sample_norm, 
    const vector<double>& old_tf_sample_norm, 
    const vector<double>& sds_tfs, 
    const vector<double>& tfs_lower, 
    const vector<double>& tfs_upper, 
    const vector<double>& wts_in_hidden, 
    const vector<double>& wts_hidden_out, 
    const vector<double>& wts_bias_hidden, 
    const vector<double>& wts_bias_out
){ 

    // calculate tf sample ratios (rwMH)
    double tf_ratio,temp_dists_tf_ratio,pann_dists_tf_ratio; 
    double transfer_ratio = 1;
    for(int j=0; j<num_biomes; j++) {
        int k = j+num_biomes;
        
        tf_ratio = my_nnet_prediction(new_tf_sample_norm[j],new_tf_sample_norm[k], wts_in_hidden, wts_hidden_out,wts_bias_hidden,wts_bias_out)[j] / my_nnet_prediction(old_tf_sample_norm[j], old_tf_sample_norm[k], wts_in_hidden,wts_hidden_out,wts_bias_hidden,wts_bias_out)[j];
            
        temp_dists_tf_ratio = truncnorm_pdf(old_tf_sample[j],new_tf_sample[j], sds_tfs[j], tfs_lower[j], tfs_upper[j])/truncnorm_pdf(new_tf_sample[j],old_tf_sample[j], sds_tfs[j], tfs_lower[j], tfs_upper[j]); 

        pann_dists_tf_ratio = truncnorm_pdf(old_tf_sample[k],new_tf_sample[k], sds_tfs[k], tfs_lower[k], tfs_upper[k])/truncnorm_pdf(new_tf_sample[k],old_tf_sample[k], sds_tfs[k], tfs_lower[k], tfs_upper[k]);
        
        transfer_ratio = transfer_ratio * tf_ratio * temp_dists_tf_ratio * pann_dists_tf_ratio;
    }

    return transfer_ratio;
    
}  


// normalize the tf samples for checking their probability in ML model space 
vector<double> transfer_function::normalize_tf_samples(
    const vector<double>& tf_samples,
    const vector<double>& normal_params
){

    vector<double> tf_samples_norm(num_biomes*2);
    
    for (int i = 0; i < num_biomes; i++) {
        tf_samples_norm[i] = (tf_samples[i] - normal_params[0]) /normal_params[1];
        tf_samples_norm[i+3] = norm_quantiles(gamma_cdf(sqrt(tf_samples[i+3]), normal_params[2], normal_params[3]));
    }
    return tf_samples_norm;
}



//--------------------------------------------------------------------------------------------------
// RECONSTRUCTION
//--------------------------------------------------------------------------------------------------

// explained variance: for the comparison of the reconstruction with a reference curve
double reconstruction::expl_variance_cpp(
    const int& n, 
    const vector<double>& reconst, 
    const vector<double>& reference
){
    double expl_variance = 0;
    double sum_xy = 0, sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0;

    for (int i = 0; i < n; i++) {
        double x = reconst[i];
        double y = reference[i];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    double correlation = numerator / denominator;
    if (denominator != 0 && correlation > 0) {
        expl_variance = correlation * correlation;
    }

    return expl_variance;
}


// caluclation of the climate reconstruction and select those data for the calibration step: recent climate and explained variance with the reference curve
vector<double> reconstruction::reconst_ap_pann(
    const vector<double>& biome_ratios_age, 
    const vector<double>& reference, 
    const vector<double>& tf_sample, 
    const int& num_biomes
) {

    // current reconstruction: Pr(C|A) = integral_B Pr(B|C) * Pr(B|A)
    vector<double> reconst_temp(length_age);
    vector<double> reconst_pann(length_age);

    for (int i = 0; i < length_age; i++) {
        for (int j = 0; j < num_biomes; j++) {
            int row_id = (length_age*j)+i; 
            reconst_temp[i] += biome_ratios_age[row_id] * tf_sample[j];
            reconst_pann[i] += biome_ratios_age[row_id] * tf_sample[j+num_biomes]; 
        }
    } 

    // recent climate values
    double recent_temp = reconst_temp[0];
    double recent_pann = reconst_pann[0];

    // explained variance with AP/NAP
    double expl_variance = expl_variance_cpp(length_age, reconst_pann, reference);

    // output
    vector<double> output = {expl_variance, recent_temp, recent_pann};
    return output;
}







