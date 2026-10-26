// Python-Anbindung (pybind11) des Original-Samplers und des ueberarbeiteten Samplers.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kde.hpp"
#include "kernel.hpp"
#include "legacy_kernel.hpp"

namespace py = pybind11;
using std::vector;

using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using larr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

template <typename T, typename A>
static vector<T> to_vec(const A& a) {
    const T* p = a.data();
    return vector<T>(p, p + a.size());
}

template <typename T>
static py::array_t<T> to_array(const vector<T>& v, std::vector<py::ssize_t> shape) {
    py::array_t<T> a(shape);
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

static kr::Model make_model(py::dict d) {
    kr::Model m;
    darr spectrum = d["spectrum"].cast<darr>();
    darr assign = d["assign"].cast<darr>();
    if (spectrum.ndim() != 2 || assign.ndim() != 2) throw std::invalid_argument("spectrum/assign muessen 2D sein");
    m.n_age = static_cast<int>(spectrum.shape(0));
    m.n_taxa = static_cast<int>(spectrum.shape(1));
    m.n_biomes = static_cast<int>(assign.shape(1));
    m.spectrum = to_vec<double>(spectrum);
    m.assign = to_vec<double>(assign);
    m.ap = to_vec<double>(d["ap"].cast<darr>());
    m.nnet.in_hidden = to_vec<double>(d["wts_in_hidden"].cast<darr>());
    m.nnet.hidden_out = to_vec<double>(d["wts_hidden_out"].cast<darr>());
    m.nnet.bias_hidden = to_vec<double>(d["wts_bias_hidden"].cast<darr>());
    m.nnet.bias_out = to_vec<double>(d["wts_bias_out"].cast<darr>());
    m.normal_params = to_vec<double>(d["normal_params"].cast<darr>());
    m.dirichlet_spread = d["dirichlet_spread"].cast<double>();
    m.jeffreys = to_vec<double>(d["jeffreys"].cast<darr>());
    m.tf_sd = to_vec<double>(d["tf_sd"].cast<darr>());
    m.tf_lower = to_vec<double>(d["tf_lower"].cast<darr>());
    m.tf_upper = to_vec<double>(d["tf_upper"].cast<darr>());
    m.beta_s1 = d["beta_shape1"].cast<double>();
    m.beta_s2 = d["beta_shape2"].cast<double>();
    m.temp_mean = d["temp_mean"].cast<double>();
    m.temp_sd = d["temp_sd"].cast<double>();
    m.pann_shape = d["pann_shape"].cast<double>();
    m.pann_rate = d["pann_rate"].cast<double>();
    m.check();
    return m;
}

static kr::Options make_options(py::dict d) {
    kr::Options o;
    if (d.contains("dirichlet_hastings_fix")) o.dirichlet_hastings_fix = d["dirichlet_hastings_fix"].cast<bool>();
    if (d.contains("use_calibration")) o.use_calibration = d["use_calibration"].cast<bool>();
    if (d.contains("use_tf_likelihood")) o.use_tf_likelihood = d["use_tf_likelihood"].cast<bool>();
    if (d.contains("use_taxa_prior")) o.use_taxa_prior = d["use_taxa_prior"].cast<bool>();
    return o;
}

// Ueberarbeiteter Sampler: eine Kette
static py::dict run_chain(py::dict model, py::dict options, darr init_w, darr init_c, int64_t n_iter, larr keep,
                          uint64_t seed) {
    const kr::Model m = make_model(model);
    const kr::Options o = make_options(options);
    const vector<double> w0 = to_vec<double>(init_w), c0 = to_vec<double>(init_c);
    const vector<int64_t> kp = to_vec<int64_t>(keep);
    if (static_cast<int>(w0.size()) != m.n_taxa || static_cast<int>(c0.size()) != 2 * m.n_biomes)
        throw std::invalid_argument("Startwerte passen nicht zu den Modelldimensionen");
    if (!std::is_sorted(kp.begin(), kp.end())) throw std::invalid_argument("keep muss aufsteigend sortiert sein");

    kr::ChainOutput out;
    {
        py::gil_scoped_release release;
        kr::Sampler s(m, o);
        out = s.run(s.make_state(w0, c0), n_iter, kp, seed);
    }
    const py::ssize_t n = static_cast<py::ssize_t>(out.expl_variance.size());
    py::dict r;
    r["taxa_weights"] = to_array(out.taxa_weights, {n, m.n_taxa});
    r["tf_sample"] = to_array(out.tf_sample, {n, 2 * m.n_biomes});
    r["expl_variance"] = to_array(out.expl_variance, {n});
    r["recent_temp"] = to_array(out.recent_temp, {n});
    r["recent_pann"] = to_array(out.recent_pann, {n});
    r["log_target"] = to_array(out.log_target, {n});
    r["n_accepted"] = to_array(out.n_accepted, {n});
    r["min_log_weight"] = out.min_log_weight;
    return r;
}

// Zielverteilung und abgeleitete Groessen eines Zustands (fuer Tests und Startwerte)
static py::dict evaluate_state(py::dict model, py::dict options, darr w, darr c) {
    const kr::Model m = make_model(model);
    kr::Sampler s(m, make_options(options));
    kr::State st = s.make_state(to_vec<double>(w), to_vec<double>(c));
    py::dict r;
    r["log_target"] = st.log_target;
    r["expl_variance"] = st.r2;
    r["recent_temp"] = st.t0;
    r["recent_pann"] = st.p0;
    return r;
}

static darr biome_ratios(py::dict model, darr w) {
    const kr::Model m = make_model(model);
    vector<double> out;
    m.biome_ratios(to_vec<double>(w), out);
    return to_array(out, {m.n_age, m.n_biomes});
}

static darr nnet_probs(py::dict model, darr temp_norm, darr pann_norm) {
    const kr::Model m = make_model(model);
    auto t = temp_norm.unchecked<1>();
    auto p = pann_norm.unchecked<1>();
    const py::ssize_t n = t.shape(0), k = static_cast<py::ssize_t>(m.nnet.n_out());
    darr out({n, k});
    auto o = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        const vector<double> pr = m.nnet.probs(t(i), p(i));
        for (py::ssize_t j = 0; j < k; ++j) o(i, j) = pr[j];
    }
    return out;
}

// Original-Sampler: eine Kette, bitidentisch zu mcmc_execution() des Originals
static py::dict run_chain_legacy(py::dict model, darr init_w, darr init_c, double init_expl_variance,
                                 double init_recent_temp, double init_recent_pann, iarr seed_samples, larr keep) {
    const kr::Model m = make_model(model);
    const int A = m.n_age, K = m.n_taxa, B = m.n_biomes;
    // spaltenweise Ablage wie im Original (taxa_spectrum_age.T.flatten(), biomes_assign.T.flatten())
    vector<double> spec_cm(static_cast<size_t>(A) * K), assign_cm(static_cast<size_t>(K) * B);
    for (int a = 0; a < A; ++a)
        for (int k = 0; k < K; ++k) spec_cm[static_cast<size_t>(A) * k + a] = m.spectrum[static_cast<size_t>(a) * K + k];
    for (int k = 0; k < K; ++k)
        for (int b = 0; b < B; ++b) assign_cm[static_cast<size_t>(K) * b + k] = m.assign[static_cast<size_t>(k) * B + b];
    const vector<int> seeds = to_vec<int32_t>(seed_samples);
    const vector<int64_t> kp64 = to_vec<int64_t>(keep);
    const vector<long> kp(kp64.begin(), kp64.end());
    const vector<double> w0 = to_vec<double>(init_w), c0 = to_vec<double>(init_c);
    if (B != 3) throw std::invalid_argument("der Original-Sampler unterstuetzt nur drei Biome");

    legacy::ChainOutput out;
    {
        py::gil_scoped_release release;
        out = legacy::run_chain(w0, c0, init_expl_variance, init_recent_temp, init_recent_pann, B, K, A, m.ap, spec_cm,
                                assign_cm, static_cast<int>(m.dirichlet_spread), m.jeffreys, m.beta_s1, m.beta_s2,
                                m.temp_mean, m.temp_sd, m.pann_shape, m.pann_rate, m.nnet.in_hidden, m.nnet.hidden_out,
                                m.nnet.bias_hidden, m.nnet.bias_out, m.normal_params, m.tf_sd, m.tf_lower, m.tf_upper,
                                static_cast<int>(seeds.size()), seeds, kp);
    }
    const py::ssize_t n = static_cast<py::ssize_t>(out.expl_variance.size());
    py::dict r;
    r["taxa_weights"] = to_array(out.taxa_weights, {n, K});
    r["tf_sample"] = to_array(out.tf_sample, {n, 2 * B});
    r["expl_variance"] = to_array(out.expl_variance, {n});
    r["recent_temp"] = to_array(out.recent_temp, {n});
    r["recent_pann"] = to_array(out.recent_pann, {n});
    r["accepted"] = to_array(out.accepted, {static_cast<py::ssize_t>(out.accepted.size())});
    return r;
}

// Netzvorhersage mit der Originalfunktion (fuer das bitidentische Gitter im Legacy-Modus)
static darr legacy_nnet_probs(darr temp_norm, darr pann_norm, darr a, darr b, darr c, darr d) {
    const vector<double> wa = to_vec<double>(a), wb = to_vec<double>(b), wc = to_vec<double>(c), wd = to_vec<double>(d);
    auto t = temp_norm.unchecked<1>();
    auto p = pann_norm.unchecked<1>();
    const py::ssize_t n = t.shape(0), k = static_cast<py::ssize_t>(wd.size());
    darr out({n, k});
    auto o = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        const vector<double> r = legacy::my_nnet_prediction(t(i), p(i), wa, wb, wc, wd);
        for (py::ssize_t j = 0; j < k; ++j) o(i, j) = r[j];
    }
    return out;
}

// Kerndichteschaetzung je Zeile von x (n_series x n) auf dem Gitter g0 + j*dg, j < m
static darr kde1d_rows(darr x, double g0, double dg, int64_t m, unsigned n_threads) {
    if (x.ndim() != 2) throw std::invalid_argument("x muss 2D sein");
    const py::ssize_t ns = x.shape(0), n = x.shape(1);
    darr out({ns, static_cast<py::ssize_t>(m)});
    const double* px = x.data();
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        kde::kde1d_many(px, ns, n, g0, dg, m, po, n_threads ? n_threads : std::thread::hardware_concurrency());
    }
    return out;
}

static darr kde2d_grid(darr a, darr b, double a0, double da, int64_t ma, double b0, double db, int64_t mb) {
    if (a.size() != b.size()) throw std::invalid_argument("a und b muessen gleich lang sein");
    darr out({static_cast<py::ssize_t>(ma), static_cast<py::ssize_t>(mb)});
    const double *pa = a.data(), *pb = b.data();
    double* po = out.mutable_data();
    const size_t n = a.size();
    {
        py::gil_scoped_release release;
        kde::kde2d(pa, pb, n, a0, da, ma, b0, db, mb, po);
    }
    return out;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++-Kern der Kinneret-Rekonstruktion";
    m.def("run_chain", &run_chain, py::arg("model"), py::arg("options"), py::arg("init_w"), py::arg("init_c"),
          py::arg("n_iter"), py::arg("keep"), py::arg("seed"));
    m.def("run_chain_legacy", &run_chain_legacy, py::arg("model"), py::arg("init_w"), py::arg("init_c"),
          py::arg("init_expl_variance"), py::arg("init_recent_temp"), py::arg("init_recent_pann"),
          py::arg("seed_samples"), py::arg("keep"));
    m.def("evaluate_state", &evaluate_state, py::arg("model"), py::arg("options"), py::arg("w"), py::arg("c"));
    m.def("biome_ratios", &biome_ratios, py::arg("model"), py::arg("w"));
    m.def("nnet_probs", &nnet_probs, py::arg("model"), py::arg("temp_norm"), py::arg("pann_norm"));
    m.def("legacy_nnet_probs", &legacy_nnet_probs);
    m.def("kde1d_rows", &kde1d_rows, py::arg("x"), py::arg("g0"), py::arg("dg"), py::arg("m"), py::arg("n_threads") = 0);
    m.def("kde2d_grid", &kde2d_grid, py::arg("a"), py::arg("b"), py::arg("a0"), py::arg("da"), py::arg("ma"),
          py::arg("b0"), py::arg("db"), py::arg("mb"));
    m.def("norm_ppf", &kr::norm_ppf);
    m.def("gamma_p", &kr::gamma_p);
}
