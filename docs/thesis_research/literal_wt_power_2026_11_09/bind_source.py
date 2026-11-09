"""Bind wt_power (mcmc_functions_v1.6.R:1003-1012) and record which run steps reach it for Kinneret (add.sites <- T, slot 3).

Reads only. Hashes must match; the pinned lines are compared after removing trailing comments. The R 3.5.0 tarball is read in memory: only line
numbers are stored, no GPL text enters this repository. Writes source_binding.json and determinism_audit.json. Usage: bind_source.py <R-3.5.0.tar.gz>
"""
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TARBALL_SHA256 = 'fd1725535e21797d3d9fea8963d99be0ba4c3aecadcf081b43e261458b416870'
F, P, M = 'src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'main_v1.2.R'
SOURCES = {M: '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
           F: 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           P: 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e'}
LITERAL = {
    M: [(24, 'library(signal)'), (47, 'library(pracma)'), (51, 'library(WaveletComp)'), (52, 'library(biwavelet)'), (73, 'add.sites <- T'), (672, 'case_studies <- F')],
    F: [(1003, 'wt_power <- function (d,l_d, n.obs=n.obs, zeros=zeros, invflen=invflen){'), (1004, 'x <- d - (sum(d)/l_d)'), (1005, 'x <- c(x, zeros)'), (1006, 'f <- fft(x)'),
        (1007, 'test <- fft(f * wb, inverse = TRUE) * invflen'), (1008, 'wave <- mvfft(f * wb, inverse = TRUE) * invflen'), (1009, 'wave <- wave[seq_len(n.obs), ]'),
        (1010, 'power <- abs(wave)^2'), (1011, 'return(power)'), (1012, '}'), (2021, 'if(spectral){'),
        (2023, 'predictant_spec <- c(wt_power(predictant_current[ages_spec][hi_ages_spec],length_ages_spec2, n.obs=n.obs, zeros=zeros, invflen=invflen)[coi_array == 1])'),
        (2024, 'posterior_temp$post_spec <- cor(predictant_spec,predictors$iso_spec,na.rm=TRUE)'),
        (2025, 'if( posterior_temp$post_spec < 0 | is.na(posterior_temp$post_spec) ) posterior_temp$post_spec <- 1e-10'),
        (2026, 'posterior_temp$post_spec <- posterior_temp$post_spec^2'), (2029, 'if(hi_spec){')],
    P: [(772, 'sources$speleothem <- 1'), (773, 'if(max(included.ages) <= 0.5) sources$speleothem <- 0'), (790, 'if(sitename == "Lake Kinneret") which.site <- 105'),
        (813, 'sources$speleothem <- 1'), (814, 'if(included.ages <= 0.5) sources$speleothem <- 0'), (821, 'if(sources$speleothem == 1){'),
        (825, 'site.age <- seq(0,220500,by=500)'), (826, 'site.age <- site.age[-c(6:29)]'), (879, 'hi_spec <- FALSE'),
        (881, 'if(length(human_impact) != 0 & !any(sitename == c("Lake Kinneret","En Gedi","Birkat Ram")) ){'), (885, 'hi_ages_spec <- rep(TRUE,length(which(common.age.spec == TRUE)))'),
        (922, 'n.obs <- length(df_speleo_temp$isotopes)'), (931, 'invflen <- 1/length(f)'), (932, 'wb <- matrix(0, nrow = n, ncol = J1 + 1)'), (934, 'wb[,a1] <- rcpp_wt_bases_morlet(k, scale[a1], -1)$daughter'),
        (938, 'rep_length <- 2^ceiling(log2(n.obs) + 1) - n.obs'), (939, 'n.obs <- length(df_speleo_nhp$isotopes)'), (940, 'invflen <- 1/length(f)'), (941, 'zeros <- rep(0, rep_length)'),
        (947, 'test <- c(wt_power(df_speleo_nhp$iso,length(df_speleo_nhp$iso), n.obs=n.obs, zeros=zeros, invflen=invflen)[coi_array == 1] )'),
        (950, 'test2 <- wt_cpp(d,wb, invflen,coi_array,len_coi)'), (1140, 'if(case_studies){'), (1145, 'if(sitename == "Lake Kinneret")  sources$speleothem <- sources$marine_ice_core <- 0'),
        (1192, 'multiple <- spectral <- FALSE'), (1203, 'if( any(which_sources == 3) )  spectral <- TRUE'), (1815, 'ages_spec <- common.age.spec'), (1816, 'length_ages_spec <- length(which(ages_spec))'),
        (1817, 'length_ages_spec2 <- length(which(ages_spec[hi_ages_spec]))')],
}
R_LOCATIONS = {'src/library/stats/src/fourier.c': [(47, r'^SEXP fft\('), (60, r'coerceVector\(z, CPLXSXP\)'), (136, r'^SEXP mvfft\(')],
               'src/main/complex.c': [(211, r'toC99.*toC99'), (284, r'hypot\(')], 'src/main/arithmetic.c': [(200, r'^double R_pow\('), (204, r'y == 2\.0')]}


def strip(line):
    return line.split('#', 1)[0].strip()


def calls(lines, pattern):
    return [i + 1 for i, text in enumerate(lines) if re.search(pattern, strip(text))]


def main():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    lines = {}
    for rel, expected in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8', 'replace').split('\n')
    for rel, pins in LITERAL.items():
        for number, text in pins:
            if strip(lines[rel][number - 1]) != text:
                raise ValueError(f'{rel}:{number} differs: {lines[rel][number - 1]!r}')
    tar_path = Path(sys.argv[1])
    if hashlib.sha256(tar_path.read_bytes()).hexdigest() != TARBALL_SHA256:
        raise ValueError('R-3.5.0.tar.gz differs from the frozen binding')
    with tarfile.open(tar_path, 'r:gz') as archive:
        for name, marks in R_LOCATIONS.items():
            text = archive.extractfile('R-3.5.0/' + name).read().decode('latin-1').split('\n')
            for number, pattern in marks:
                if not re.search(pattern, text[number - 1]):
                    raise ValueError(f'{name}:{number} differs from the expected location')
    wt_in_f, wt_in_p = calls(lines[F], r'\bwt_power\s*\('), calls(lines[P], r'\bwt_power\s*\(')
    cpp_in_p = calls(lines[P], r'\bwt_cpp\s*\(')
    if wt_in_f != [991, 2023, 2031] or wt_in_p != [947, 966] or cpp_in_p != [950, 971]:
        raise ValueError(f'unexpected callers {wt_in_f} {wt_in_p} {cpp_in_p}')
    binding = {
        'kind': 'source_binding_wt_power', 'sources': SOURCES, 'r_tarball_sha256': TARBALL_SHA256, 'r_locations_line_numbers_only': {k: [n for n, _ in v] for k, v in R_LOCATIONS.items()},
        'pinned_lines': {k: [n for n, _ in v] for k, v in LITERAL.items()},
        'callers': {'mcmc_functions_v1.6.R': {'991': 'inside the reconstruction of line 855, replaced by line 1781; the call also passes removed arguments', '2023': 'active reader when spectral', '2031': 'only when hi_spec'},
                    'mcmc_prepare.R': {'947': 'test call inside if(sources$speleothem == 1), result without reader', '966': 'same, 100 times in system.time'},
                    'wt_cpp_only_in_mcmc_prepare': cpp_in_p, 'not_in_handoff': 'the handoff named only line 2023; lines 947 and 966 run it as well when sources$speleothem is 1'},
        'kinneret': {'add_sites': True, 'case_studies': False, 'line_1145_reached': False, 'line_1145_reason': 'behind if(case_studies), and main_v1.2.R:672 sets case_studies to F',
                     'spectral': 'open: mcmc_prepare.R:1203 needs sources$speleothem 1, which needs d18o.age and the final coordinates (lines 759-814)',
                     'hi_spec': False, 'hi_ages_spec': 'all TRUE (lines 881, 885)', 'line_2031_reached': False, 'length_ages_spec2_equals_length_of_d': True},
        'free_variables': {'wb': 'global, mcmc_prepare.R:932-935; real, since rcpp_wt_bases_morlet returns a NumericVector in all four biwavelet versions read (fft_masking.json)', 'formals_without_usable_default': ['n.obs', 'zeros', 'invflen']},
        'line_1007': 'assigns test, which has no reader; left out of the module',
        'marked_not_corrected': {'mcmc_prepare.R:1817': 'hi_ages_spec is shorter than ages_spec and is recycled; only for hi_spec sites can length_ages_spec2 then differ from length(d). Kinneret is not one.',
                                 'site_age_versus_age': 'site.age is overwritten at lines 825-826, so ages_spec need not have the length of predictant_current; the module applies R logical indexing and rejects a series that no longer fits wb'},
        'historical_reproduction_ready': False,
    }
    (OUT / 'source_binding.json').write_text(json.dumps(binding, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    audit = {'kind': 'determinism_audit_wt_power', 'random_state': False, 'package_semantics_in_module': ['stats::fft and stats::mvfft replaced by numpy.fft on the user decision of 2026-11-09'],
             'bit_identical_with_R': False, 'outside_this_step': ['rcpp_wt_bases_morlet (biwavelet)', 'biwavelet::wt for predictors$iso_spec and coi_array', 'smoothing of the speleothem series'],
             'no_gpl_text_copied': True, 'historical_reproduction_ready': False}
    (OUT / 'determinism_audit.json').write_text(json.dumps(audit, indent=1) + '\n', encoding='utf-8', newline='\n')
    print('bound', {k: len(v) for k, v in LITERAL.items()}, wt_in_f, wt_in_p, cpp_in_p)


if __name__ == '__main__':
    main()
