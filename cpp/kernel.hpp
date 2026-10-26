// Ueberarbeiteter MCMC-Sampler fuer das hierarchische Modell aus Netzel et al. (2025).
//
// Zielverteilung (unnormiert), Zustand x = (w, c):
//   pi(w, c) = Beta(R2(w,c); s1, s2) * N(T0(w,c); mu_T, sd_T) * Gamma(P0(w,c); k, theta)
//            * Dir(w; jeffreys) * prod_j p(B=j | T_j, P_j)
// Vorschlaege:
//   w' ~ Dir(kappa * w)                  (Random-Walk auf dem Simplex)
//   c' ~ TN(c, sd, [lower, upper])       (trunkierte Normalverteilung je Komponente)
//
// Gegenueber dem Original (legacy_kernel.hpp):
//   * alle Dichten und das Akzeptanzverhaeltnis im Log-Raum,
//   * Hastings-Korrektur des Dirichlet-Vorschlags wahlweise mit kappa (korrekt) oder
//     ohne kappa (Verhalten des Originals),
//   * portabler Zufallszahlengenerator (std::mt19937_64 mit eigenen Verteilungen),
//     d. h. identische Ketten mit jedem standardkonformen Compiler,
//   * Phi^-1 in doppelter statt einfacher Genauigkeit,
//   * Anzahl der Biome und Groesse des neuronalen Netzes nicht mehr fest verdrahtet,
//   * nur ausgewaehlte Iterationen werden gespeichert.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace kr {

using std::vector;

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
constexpr double LOG_2PI = 1.8378770664093454836;

// ---------------------------------------------------------------------------------------------
// Spezielle Funktionen
// ---------------------------------------------------------------------------------------------

inline double horner(const double* c, int m, double x) {
    double v = c[m];
    for (int i = m - 1; i >= 0; --i) v = v * x + c[i];
    return v;
}

// Standardnormal-Quantil, Algorithmus AS241 (Wichura 1988), relative Genauigkeit ~1e-16
inline double norm_ppf(double p) {
    static const double a[8] = {3.3871328727963666080, 1.3314166789178437745E+2, 1.9715909503065514427E+3,
                                1.3731693765509461125E+4, 4.5921953931549871457E+4, 6.7265770927008700853E+4,
                                3.3430575583588128105E+4, 2.5090809287301226727E+3};
    static const double b[8] = {1.0, 4.2313330701600911252E+1, 6.8718700749205790830E+2, 5.3941960214247511077E+3,
                                2.1213794301586595867E+4, 3.9307895800092710610E+4, 2.8729085735721942674E+4,
                                5.2264952788528545610E+3};
    static const double c[8] = {1.42343711074968357734, 4.63033784615654529590, 5.76949722146069140550,
                                3.64784832476320460504, 1.27045825245236838258, 2.41780725177450611770E-1,
                                2.27238449892691845833E-2, 7.74545014278341407640E-4};
    static const double d[8] = {1.0, 2.05319162663775882187, 1.67638483018380384940, 6.89767334985100004550E-1,
                                1.48103976427480074590E-1, 1.51986665636164571966E-2, 5.47593808499534494600E-4,
                                1.05075007164441684324E-9};
    static const double e[8] = {6.65790464350110377720, 5.46378491116411436990, 1.78482653991729133580,
                                2.96560571828504891230E-1, 2.65321895265761230930E-2, 1.24266094738807843860E-3,
                                2.71155556874348757815E-5, 2.01033439929228813265E-7};
    static const double f[8] = {1.0, 5.99832206555887937690E-1, 1.36929880922735805310E-1, 1.48753612908506148525E-2,
                                7.86869131145613259100E-4, 1.84631831751005468180E-5, 1.42151175831644588870E-7,
                                2.04426310338993978564E-15};
    if (p <= 0.0) return -HUGE_VAL;
    if (p >= 1.0) return HUGE_VAL;
    const double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q * horner(a, 7, r) / horner(b, 7, r);
    }
    double r = std::sqrt(-std::log(q < 0.0 ? p : 1.0 - p));
    double v;
    if (r <= 5.0) {
        r -= 1.6;
        v = horner(c, 7, r) / horner(d, 7, r);
    } else {
        r -= 5.0;
        v = horner(e, 7, r) / horner(f, 7, r);
    }
    return q < 0.0 ? -v : v;
}

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

// regularisierte untere unvollstaendige Gammafunktion P(s, z)
// (Reihe bzw. Kettenbruch wie im Original, aber mit std::lgamma)
inline double gamma_p(double s, double z) {
    if (z <= 0.0) return 0.0;
    if (z <= 1.0 || z < s) {
        double sum = 1.0, x = 1.0;
        for (int k = 1; k < 1000; ++k) {
            sum += (x *= z / (s + k));
            if (x / sum < 1e-16) break;
        }
        return std::exp(s * std::log(z) - z - std::lgamma(s + 1.0) + std::log(sum));
    }
    const double tiny = 1e-290;
    double f = 1.0 + z - s, C = f, D = 0.0;
    for (int j = 1; j < 1000; ++j) {
        const double a = j * (s - j), b = 2 * j + 1 + z - s;
        D = b + a * D;
        if (std::fabs(D) < tiny) D = tiny;
        C = b + a / C;
        if (std::fabs(C) < tiny) C = tiny;
        D = 1.0 / D;
        const double dd = C * D;
        f *= dd;
        if (std::fabs(dd - 1.0) < 1e-16) break;
    }
    return 1.0 - std::exp(s * std::log(z) - z - std::lgamma(s) - std::log(f));
}

inline double logsumexp(const vector<double>& x) {
    double m = *std::max_element(x.begin(), x.end());
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (double v : x) s += std::exp(v - m);
    return m + std::log(s);
}

// ---------------------------------------------------------------------------------------------
// Portabler Zufallszahlengenerator
// ---------------------------------------------------------------------------------------------

class Rng {
public:
    explicit Rng(uint64_t seed) : eng_(seed) {}

    // gleichverteilt auf (0, 1), 53 Bit
    double uniform() { return (static_cast<double>(eng_() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

    double normal() { return norm_ppf(uniform()); }

    // log einer Gamma(a, 1)-Zufallszahl; fuer a < 1 im Log-Raum, damit kleine Formparameter
    // nicht auf 0 unterlaufen (Marsaglia & Tsang 2000, inkl. Boost fuer a < 1)
    double log_gamma(double a) {
        if (a < 1.0) {
            return log_gamma(a + 1.0) + std::log(uniform()) / a;
        }
        const double d = a - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x, v;
            do {
                x = normal();
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            const double u = uniform();
            if (std::log(u) < 0.5 * x * x + d - d * v + d * std::log(v)) return std::log(d * v);
        }
    }

private:
    std::mt19937_64 eng_;
};

// ---------------------------------------------------------------------------------------------
// Log-Dichten
// ---------------------------------------------------------------------------------------------

inline double log_dirichlet(const vector<double>& log_x, const vector<double>& alpha) {
    double sa = 0.0, s = 0.0;
    for (size_t i = 0; i < alpha.size(); ++i) {
        sa += alpha[i];
        s += (alpha[i] - 1.0) * log_x[i] - std::lgamma(alpha[i]);
    }
    return std::lgamma(sa) + s;
}

inline double log_beta_pdf(double x, double a, double b) {
    if (x <= 0.0 || x >= 1.0) return NEG_INF;
    return (a - 1.0) * std::log(x) + (b - 1.0) * std::log1p(-x) + std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
}

inline double log_norm_pdf(double x, double mu, double sd) {
    const double z = (x - mu) / sd;
    return -0.5 * z * z - std::log(sd) - 0.5 * LOG_2PI;
}

inline double log_gamma_pdf(double x, double shape, double rate) {
    if (x <= 0.0) return NEG_INF;
    return shape * std::log(rate) + (shape - 1.0) * std::log(x) - rate * x - std::lgamma(shape);
}

// Masse der Standardnormalverteilung auf [alpha, beta], stabil auch im oberen Ausläufer
inline double norm_mass(double alpha, double beta) {
    if (alpha > 0.0) return norm_cdf(-alpha) - norm_cdf(-beta);
    return norm_cdf(beta) - norm_cdf(alpha);
}

inline double log_truncnorm_pdf(double x, double mu, double sd, double lo, double hi) {
    if (x < lo || x > hi) return NEG_INF;
    const double z = (x - mu) / sd;
    return -0.5 * z * z - 0.5 * LOG_2PI - std::log(sd) - std::log(norm_mass((lo - mu) / sd, (hi - mu) / sd));
}

inline double truncnorm_sample(double mu, double sd, double lo, double hi, Rng& rng) {
    const double alpha = (lo - mu) / sd, beta = (hi - mu) / sd;
    const double u = rng.uniform();
    double z;
    if (alpha > 0.0) {  // Spiegelung fuer Genauigkeit im oberen Ausläufer
        const double pa = norm_cdf(-alpha), pb = norm_cdf(-beta);
        z = -norm_ppf(pa - u * (pa - pb));
    } else {
        const double pa = norm_cdf(alpha), pb = norm_cdf(beta);
        z = norm_ppf(pa + u * (pb - pa));
    }
    return std::min(hi, std::max(lo, mu + sd * z));
}

// ---------------------------------------------------------------------------------------------
// Modellbausteine
// ---------------------------------------------------------------------------------------------

struct Nnet {
    vector<double> in_hidden, hidden_out, bias_hidden, bias_out;
    size_t n_hidden() const { return bias_hidden.size(); }
    size_t n_out() const { return bias_out.size(); }

    // log softmax-Wahrscheinlichkeit der Klasse k fuer normierte Eingaben (t, p)
    double log_prob(double t, double p, size_t k) const {
        const size_t nh = n_hidden(), no = n_out();
        double hidden[64];
        for (size_t j = 0; j < nh; ++j) {
            const double s = in_hidden[j] * t + in_hidden[j + nh] * p + bias_hidden[j];
            hidden[j] = 1.0 / (1.0 + std::exp(-s));
        }
        double out[64];
        double m = NEG_INF;
        for (size_t i = 0; i < no; ++i) {
            double s = bias_out[i];
            for (size_t j = 0; j < nh; ++j) s += hidden[j] * hidden_out[i * nh + j];
            out[i] = s;
            m = std::max(m, s);
        }
        double z = 0.0;
        for (size_t i = 0; i < no; ++i) z += std::exp(out[i] - m);
        return out[k] - m - std::log(z);
    }

    vector<double> probs(double t, double p) const {
        vector<double> r(n_out());
        for (size_t k = 0; k < n_out(); ++k) r[k] = std::exp(log_prob(t, p, k));
        return r;
    }
};

struct Model {
    // Kerndaten (zeilenweise): spectrum A x K, assign K x B, ap A
    int n_age = 0, n_taxa = 0, n_biomes = 0;
    vector<double> spectrum, assign, ap;
    // Transferfunktion und Normierung (mu_T, sd_T, Gamma-Form, Gamma-Rate fuer sqrt(P))
    Nnet nnet;
    vector<double> normal_params;
    // Vorschlagsverteilungen
    double dirichlet_spread = 0.0;
    vector<double> jeffreys;
    vector<double> tf_sd, tf_lower, tf_upper;  // Laenge 2B: erst T je Biom, dann P je Biom
    // Kalibrierung
    double beta_s1 = 3, beta_s2 = 3, temp_mean = 0, temp_sd = 1, pann_shape = 1, pann_rate = 1;

    void check() const {
        const size_t A = n_age, K = n_taxa, B = n_biomes;
        if (spectrum.size() != A * K || assign.size() != K * B || ap.size() != A || jeffreys.size() != K ||
            tf_sd.size() != 2 * B || tf_lower.size() != 2 * B || tf_upper.size() != 2 * B || normal_params.size() != 4)
            throw std::invalid_argument("inkonsistente Modelldimensionen");
        if (nnet.n_hidden() > 64 || nnet.n_out() > 64 || nnet.n_out() < B ||
            nnet.in_hidden.size() != 2 * nnet.n_hidden() || nnet.hidden_out.size() != nnet.n_hidden() * nnet.n_out())
            throw std::invalid_argument("inkonsistente Netzgewichte");
    }

    double norm_temp(double t) const { return (t - normal_params[0]) / normal_params[1]; }
    double norm_pann(double p) const { return norm_ppf(gamma_p(normal_params[2], std::sqrt(p) * normal_params[3])); }

    // Biomanteile je Alter (A x B, zeilenweise auf 1 normiert)
    void biome_ratios(const vector<double>& w, vector<double>& out) const {
        out.assign(static_cast<size_t>(n_age) * n_biomes, 0.0);
        for (int a = 0; a < n_age; ++a) {
            double* row = &out[static_cast<size_t>(a) * n_biomes];
            const double* sp = &spectrum[static_cast<size_t>(a) * n_taxa];
            for (int k = 0; k < n_taxa; ++k) {
                const double sw = sp[k] * w[k];
                if (sw == 0.0) continue;
                const double* as = &assign[static_cast<size_t>(k) * n_biomes];
                for (int b = 0; b < n_biomes; ++b) row[b] += sw * as[b];
            }
            double s = 0.0;
            for (int b = 0; b < n_biomes; ++b) s += row[b];
            for (int b = 0; b < n_biomes; ++b) row[b] /= s;
        }
    }

    // abgeleitete Groessen: R2 (P-Rekonstruktion vs. AP), T und P der juengsten Altersstufe
    void derived(const vector<double>& w, const vector<double>& c, vector<double>& ratios, double& r2, double& t0,
                 double& p0) const {
        biome_ratios(w, ratios);
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        t0 = p0 = 0.0;
        for (int a = 0; a < n_age; ++a) {
            double rt = 0.0, rp = 0.0;
            for (int b = 0; b < n_biomes; ++b) {
                rt += ratios[static_cast<size_t>(a) * n_biomes + b] * c[b];
                rp += ratios[static_cast<size_t>(a) * n_biomes + b] * c[b + n_biomes];
            }
            if (a == 0) {
                t0 = rt;
                p0 = rp;
            }
            const double y = ap[a];
            sx += rp; sy += y; sxx += rp * rp; syy += y * y; sxy += rp * y;
        }
        const double n = n_age;
        const double num = n * sxy - sx * sy;
        const double den = std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
        const double corr = num / den;
        r2 = (den != 0.0 && corr > 0.0) ? corr * corr : 0.0;
    }
};

struct Options {
    bool dirichlet_hastings_fix = true;  // kappa in der Hastings-Korrektur des Dirichlet-Vorschlags
    // Terme der Zielverteilung einzeln abschaltbar (fuer Validierungslaeufe, z. B. "nur Prior")
    bool use_calibration = true;
    bool use_tf_likelihood = true;
    bool use_taxa_prior = true;
};

struct State {
    vector<double> log_w, w, c;
    double r2 = 0, t0 = 0, p0 = 0;
    double log_target = NEG_INF;
};

struct ChainOutput {
    vector<double> taxa_weights, tf_sample, expl_variance, recent_temp, recent_pann, log_target;
    vector<int64_t> n_accepted;  // kumulierte Anzahl akzeptierter Vorschlaege bis zur gespeicherten Iteration
    double min_log_weight = 0.0;  // kleinstes je erreichtes log(w_k) (Diagnose fuer Randnaehe)
};

class Sampler {
public:
    Sampler(const Model& m, const Options& o) : m_(m), o_(o) { m_.check(); }

    // Log-Zielverteilung eines Zustands; berechnet die abgeleiteten Groessen mit
    double log_target(State& s) const {
        m_.derived(s.w, s.c, ratios_, s.r2, s.t0, s.p0);
        double lt = 0.0;
        if (o_.use_calibration) {
            lt += log_beta_pdf(s.r2, m_.beta_s1, m_.beta_s2);
            lt += log_norm_pdf(s.t0, m_.temp_mean, m_.temp_sd);
            lt += log_gamma_pdf(s.p0, m_.pann_shape, m_.pann_rate);
        }
        if (o_.use_taxa_prior) lt += log_dirichlet(s.log_w, m_.jeffreys);
        if (o_.use_tf_likelihood) {
            const int B = m_.n_biomes;
            for (int b = 0; b < B; ++b)
                lt += m_.nnet.log_prob(m_.norm_temp(s.c[b]), m_.norm_pann(s.c[b + B]), b);
        }
        for (size_t j = 0; j < s.c.size(); ++j)
            if (s.c[j] < m_.tf_lower[j] || s.c[j] > m_.tf_upper[j]) lt = NEG_INF;
        return lt;
    }

    State make_state(const vector<double>& w, const vector<double>& c) const {
        State s;
        double sum = 0.0;
        for (double v : w) sum += v;
        s.w.resize(w.size());
        s.log_w.resize(w.size());
        for (size_t k = 0; k < w.size(); ++k) {
            s.w[k] = w[k] / sum;
            s.log_w[k] = std::log(s.w[k]);
        }
        s.c = c;
        s.log_target = log_target(s);
        return s;
    }

    // log q(to | from) des Dirichlet-Vorschlags Dir(kappa * from)
    double log_q_weights(const State& to, const State& from) const {
        const double kappa = o_.dirichlet_hastings_fix ? m_.dirichlet_spread : 1.0;
        alpha_.resize(from.w.size());
        for (size_t k = 0; k < from.w.size(); ++k) alpha_[k] = std::max(kappa * from.w[k], 1e-300);
        return log_dirichlet(to.log_w, alpha_);
    }

    ChainOutput run(State cur, int64_t n_iter, const vector<int64_t>& keep, uint64_t seed) const {
        Rng rng(seed);
        ChainOutput out;
        const size_t K = m_.n_taxa, C = 2 * static_cast<size_t>(m_.n_biomes);
        out.taxa_weights.reserve(keep.size() * K);
        out.tf_sample.reserve(keep.size() * C);
        out.min_log_weight = *std::min_element(cur.log_w.begin(), cur.log_w.end());

        int64_t n_acc = 0;
        size_t next_keep = 0;
        auto store = [&](int64_t i) {
            while (next_keep < keep.size() && keep[next_keep] == i) {
                out.taxa_weights.insert(out.taxa_weights.end(), cur.w.begin(), cur.w.end());
                out.tf_sample.insert(out.tf_sample.end(), cur.c.begin(), cur.c.end());
                out.expl_variance.push_back(cur.r2);
                out.recent_temp.push_back(cur.t0);
                out.recent_pann.push_back(cur.p0);
                out.log_target.push_back(cur.log_target);
                out.n_accepted.push_back(n_acc);
                ++next_keep;
            }
        };
        store(0);

        State prop;
        prop.log_w.resize(K);
        prop.w.resize(K);
        prop.c.resize(C);
        vector<double> log_y(K);

        for (int64_t i = 1; i < n_iter; ++i) {
            // Taxa-Gewichte: w' ~ Dir(kappa * w), im Log-Raum gezogen
            for (size_t k = 0; k < K; ++k) log_y[k] = rng.log_gamma(std::max(m_.dirichlet_spread * cur.w[k], 1e-300));
            const double lse = logsumexp(log_y);
            for (size_t k = 0; k < K; ++k) {
                prop.log_w[k] = log_y[k] - lse;
                prop.w[k] = std::exp(prop.log_w[k]);
            }
            // Klimawerte der Transferfunktionen: c' ~ TN(c, sd, [lower, upper])
            double log_hastings_c = 0.0;
            for (size_t j = 0; j < C; ++j) {
                prop.c[j] = truncnorm_sample(cur.c[j], m_.tf_sd[j], m_.tf_lower[j], m_.tf_upper[j], rng);
                log_hastings_c += log_truncnorm_pdf(cur.c[j], prop.c[j], m_.tf_sd[j], m_.tf_lower[j], m_.tf_upper[j]) -
                                  log_truncnorm_pdf(prop.c[j], cur.c[j], m_.tf_sd[j], m_.tf_lower[j], m_.tf_upper[j]);
            }
            prop.log_target = log_target(prop);

            const double log_hastings_w = log_q_weights(cur, prop) - log_q_weights(prop, cur);
            const double log_alpha = prop.log_target - cur.log_target + log_hastings_w + log_hastings_c;

            if (std::log(rng.uniform()) < log_alpha) {
                std::swap(cur, prop);
                ++n_acc;
                const double mlw = *std::min_element(cur.log_w.begin(), cur.log_w.end());
                out.min_log_weight = std::min(out.min_log_weight, mlw);
            }
            store(i);
        }
        return out;
    }

private:
    Model m_;
    Options o_;
    mutable vector<double> ratios_, alpha_;
};

}  // namespace kr
