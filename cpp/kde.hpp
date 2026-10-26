// Gauss-Kerndichteschaetzer auf regelmaessigen Gittern.
//
// Ergebnis wie scipy.stats.gaussian_kde(...).evaluate(grid) (Bandbreite nach Scott), jedoch
// werden nur Gitterpunkte innerhalb von CUT Bandbreiten um jede Stichprobe ausgewertet.
// Der dadurch vernachlaessigte Beitrag ist < exp(-CUT^2 / 2) = exp(-50) relativ zum Kernmaximum.
#pragma once

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace kde {

using std::vector;
constexpr double CUT = 10.0;
constexpr double INV_SQRT_2PI = 0.39894228040143267794;

// Scott-Bandbreite (Standardabweichung des Kerns) wie in scipy: sd(ddof=1) * n^(-1/5)
inline double scott_bandwidth(const double* x, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m += x[i];
    m /= static_cast<double>(n);
    double v = 0.0;
    for (size_t i = 0; i < n; ++i) v += (x[i] - m) * (x[i] - m);
    v /= static_cast<double>(n - 1);
    return std::sqrt(v) * std::pow(static_cast<double>(n), -0.2);
}

// 1D-Dichte einer Stichprobe x (Laenge n) auf dem regelmaessigen Gitter g0 + j*dg, j = 0..m-1
inline void kde1d(const double* x, size_t n, double g0, double dg, size_t m, double* out) {
    std::fill(out, out + m, 0.0);
    const double h = scott_bandwidth(x, n);
    if (!(h > 0.0)) {  // entartete Stichprobe (alle Werte gleich): wie scipy nicht definiert
        std::fill(out, out + m, std::nan(""));
        return;
    }
    const double norm = INV_SQRT_2PI / (h * static_cast<double>(n));
    for (size_t i = 0; i < n; ++i) {
        const double lo = std::ceil((x[i] - CUT * h - g0) / dg), hi = std::floor((x[i] + CUT * h - g0) / dg);
        const long j0 = std::max(0L, static_cast<long>(lo));
        const long j1 = std::min(static_cast<long>(m) - 1, static_cast<long>(hi));
        for (long j = j0; j <= j1; ++j) {
            const double z = (g0 + j * dg - x[i]) / h;
            out[j] += std::exp(-0.5 * z * z);
        }
    }
    for (size_t j = 0; j < m; ++j) out[j] *= norm;
}

// viele Reihen (z. B. ein Zeitschritt je Reihe) parallel: x ist (n_series x n) zeilenweise
inline void kde1d_many(const double* x, size_t n_series, size_t n, double g0, double dg, size_t m, double* out,
                       unsigned n_threads) {
    n_threads = std::max(1u, std::min<unsigned>(n_threads, static_cast<unsigned>(n_series)));
    vector<std::thread> pool;
    for (unsigned t = 0; t < n_threads; ++t) {
        pool.emplace_back([=]() {
            for (size_t s = t; s < n_series; s += n_threads) kde1d(x + s * n, n, g0, dg, m, out + s * m);
        });
    }
    for (auto& th : pool) th.join();
}

// 2D-Dichte (volle Kovarianz wie scipy) auf dem Gitter (a0 + i*da, b0 + j*db), Ausgabe (ma x mb)
inline void kde2d(const double* a, const double* b, size_t n, double a0, double da, size_t ma, double b0, double db,
                  size_t mb, double* out) {
    std::fill(out, out + ma * mb, 0.0);
    double mean_a = 0, mean_b = 0;
    for (size_t i = 0; i < n; ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= n;
    mean_b /= n;
    double saa = 0, sbb = 0, sab = 0;
    for (size_t i = 0; i < n; ++i) {
        saa += (a[i] - mean_a) * (a[i] - mean_a);
        sbb += (b[i] - mean_b) * (b[i] - mean_b);
        sab += (a[i] - mean_a) * (b[i] - mean_b);
    }
    const double f2 = std::pow(static_cast<double>(n), -2.0 / 6.0);  // Scott: n^(-1/(d+4)), d = 2
    const double caa = saa / (n - 1) * f2, cbb = sbb / (n - 1) * f2, cab = sab / (n - 1) * f2;
    const double det = caa * cbb - cab * cab;
    if (!(det > 0.0)) {
        std::fill(out, out + ma * mb, std::nan(""));
        return;
    }
    const double iaa = cbb / det, ibb = caa / det, iab = -cab / det;
    const double norm = 1.0 / (2.0 * M_PI * std::sqrt(det) * static_cast<double>(n));
    const double ra = CUT * std::sqrt(caa), rb = CUT * std::sqrt(cbb);
    for (size_t k = 0; k < n; ++k) {
        const long i0 = std::max(0L, static_cast<long>(std::ceil((a[k] - ra - a0) / da)));
        const long i1 = std::min(static_cast<long>(ma) - 1, static_cast<long>(std::floor((a[k] + ra - a0) / da)));
        const long j0 = std::max(0L, static_cast<long>(std::ceil((b[k] - rb - b0) / db)));
        const long j1 = std::min(static_cast<long>(mb) - 1, static_cast<long>(std::floor((b[k] + rb - b0) / db)));
        for (long i = i0; i <= i1; ++i) {
            const double u = a0 + i * da - a[k];
            double* row = out + static_cast<size_t>(i) * mb;
            for (long j = j0; j <= j1; ++j) {
                const double v = b0 + j * db - b[k];
                const double q = iaa * u * u + 2.0 * iab * u * v + ibb * v * v;
                if (q < CUT * CUT) row[j] += std::exp(-0.5 * q);
            }
        }
    }
    for (size_t i = 0; i < ma * mb; ++i) out[i] *= norm;
}

}  // namespace kde
