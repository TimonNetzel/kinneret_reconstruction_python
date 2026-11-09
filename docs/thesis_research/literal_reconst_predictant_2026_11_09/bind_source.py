"""Bind reconst/predictant (mcmc_functions_v1.6.R:1817-1820) and record who reads them in the Kinneret run (add.sites <- T, slot 3).

Reads only. Hashes must match. Comments and strings are stripped before searching, so a commented call or embedded C++ is not counted.
The R 3.5.0 source is read from the CRAN tarball in memory; only line numbers and hashes are stored, no GPL text enters this repository.
Writes source_binding.json, source_excerpt.json, determinism_audit.json. Usage: bind_source.py [R-3.5.0.tar.gz]
"""
import hashlib
import importlib.util
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '38775e1fc959b936fb22b7dd6dc88bef69fb7eb1'
TARBALL_SHA256 = 'fd1725535e21797d3d9fea8963d99be0ba4c3aecadcf081b43e261458b416870'
SOURCES = {'main_v1.2.R': '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
           'src/mcmc_functions_v1.6.R': 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           'src/mcmc_prepare.R': 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
           'src/mcmc_conduction.R': '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
ALSO_LOADED = ['src/loess_functions.R', 'src/check_ages.R', 'src/prior_climate.R', 'src/mcmc_trimming_posterior_trafo.R']
F, P, M, C, A = 'src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'main_v1.2.R', 'src/mcmc_conduction.R', 'src/check_ages.R'
LITERAL = {
    M: [(644, 'source("src/mcmc_functions_v1.6.R")'), (670, 'past_climate_module <- F'), (672, 'case_studies <- F'), (675, 'source("src/mcmc_prepare.R")'),
        (676, 'source("src/mcmc_conduction.R")'), (677, 'source("src/mcmc_trimming_posterior_trafo.R")')],
    C: [(57, 'posterior$how_many_taxa <- num_taxa'), (58, 'parameter_stage <- parameter_stage_weights'),
        (69, 'taxa.appear.core.temp <- taxa.appear.core'), (109, 'reconst <- array(NA, dim = c(n_age,2))'), (110, 'predictant_tann <- predictant_pann <- c()'),
        (115, 'posterior_temp$tf_sample <- array(NA, dim = c(num_taxa,2))'),
        (142, 'posterior_temp <- parameter_stage(posterior_temp=posterior_temp,taxa.appear.core=taxa.appear.core.temp,taxa_information=taxa_information)'),
        (146, 'posterior_temp <- reconstruction(posterior_temp=posterior_temp,tf.x1.current[posterior_temp$z,],tf.x2.current[posterior_temp$z,],tf_x12_mean_current[posterior_temp$z,],predictors,tann,recent,multiple,spectral,stad_interstad)'),
        (420, 'posterior$youngest_values[i,] <-  posterior_temp$youngest_values')],
    F: [(855, 'reconstruction <- function(taxa.appear.core,which.taxa,age,tf.x1.current,tf.x2.current,tf_x12_mean_current,predictors,tann,recent,multiple,spectral,stad_interstad){'),
        (1242, 'parameter_stage_weights <- function(posterior_temp=posterior_temp,taxa.appear.core=taxa.appear.core,taxa_information=taxa_information){'),
        (1277, 'posterior_temp$taxa.appear.core.temp <- taxa.appear.core.temp / rowSums(taxa.appear.core.temp, na.rm=T)'),
        (1781, 'reconstruction <- function(posterior_temp,tf.x1.current,tf.x2.current,tf_sample,predictors,tann,recent,multiple,spectral,stad_interstad){'),
        (1812, 'posterior_temp$tf_sample[,1] <- rtruncnorm_cpp(num_taxa,posterior$tf_sample[[i-1]][,1],sds_tann,tf_quarts_tann[,1], tf_quarts_tann[,3])'),
        (1813, 'posterior_temp$tf_sample[,2] <- rtruncnorm_cpp(num_taxa,posterior$tf_sample[[i-1]][,2],sds_pann,tf_quarts_pann[,1], tf_quarts_pann[,3])'),
        (1817, 'taxa.appear.core <- posterior_temp$taxa.appear.core.temp'), (1818, 'reconst <- taxa.appear.core[reconst_ages,] %*% posterior_temp$tf_sample'),
        (1819, 'predictant_tann <- reconst[,1]'), (1820, 'predictant_pann <- reconst[,2]'), (1825, 'posterior_temp$youngest_values <- reconst[1,]'),
        (1829, 'if(taxa_change){'), (1844, 'if(recent){'), (1850, 'posterior_temp$post_tann <- predictant_tann[1]'), (1851, 'posterior_temp$post_pann <- predictant_pann[1]'),
        (1853, 'posterior_temp$post_tann <- prior.recent$mean.tmp'), (1854, 'posterior_temp$post_pann <- prior.recent$mean.pann'),
        (1860, 'if(past_climate_module){'), (1921, 'if(stad_interstad){'), (1939, 'if(tann){'), (1940, 'predictant_current <- predictant_tann'), (1942, 'predictant_current <- predictant_pann')],
    P: [(91, 'all.avail.age <- array(NA, dim = c(length(age),ncol(all.avail)))'), (440, 'ages_hol <- age %in% hol'), (468, 'stad_interstad <- FALSE'),
        (728, 'which.modern.avail <- which(age <= 11000)'), (1183, 'recent <- TRUE'), (1184, 'if( length(which.modern.avail) == 0) recent <- FALSE'),
        (1249, 'tann <- FALSE'), (1454, 'taxa.appear.core <- all.avail.age[,which_taxa]'),
        (1460, 'taxa.merge <- anyNA(match(taxa.sampled,taxa.distribution.ww$names))'), (1462, 'taxa_change <- FALSE'), (1463, 'if(!taxa.merge){'), (1464, 'reconst_ages <- rep(TRUE, length(age))'),
        (1466, 'reconst_ages <- ages_hol'), (1470, 'if( taxa.merge){'), (1500, 'if(any(sitename == c("Birkat Ram","Dead Sea")) & FALSE){'), (1501, 'taxa_change <- TRUE'), (1522, 'reconst_ages <- rep(TRUE, length(age))')],
    A: [(179, 'site.age <- age <- core.data[[station_id]]$age[[age.reso]]'), (185, 'site.age <- age <- output$age[[age.reso]]'), (189, 'site.age <- age <- output$age[[(age.reso+1)]]')],
}
EXPECTED = {
    'reconst_ages': {F: [461, 462, 470, 1818], P: [1464, 1466, 1522]},
    'reconst': {C: [109], F: [9, 10, 11, 858, 859, 860, 1818, 1819, 1820, 1825]},
    'taxa_change': {F: [463, 471, 1829], P: [1462, 1501, 1583, 1615, 1632, 1652, 1671, 1696]},
    'reconst_tann': {F: [924, 925, 1923, 1924]}, 'reconst_pann': {F: [929, 930, 1928, 1929]},
}
ACTIVE_BODY_FIRST, ACTIVE_BODY_READS_OF_PREDICTANTS = 1782, {
    'predictant_tann': [1819, 1831, 1834, 1837, 1850, 1867, 1879, 1901, 1940, 1966, 1993],
    'predictant_pann': [1820, 1832, 1835, 1837, 1851, 1868, 1880, 1892, 1902, 1942, 1982, 2009]}
PREDICTANT_CURRENT_READS = [1964, 1999, 2023, 2031]
# Reachability for Kinneret; the flags are bound in LITERAL above and in the earlier local-cor packet.
READERS = [
    {'line': '1819, 1820', 'reads': 'reconst', 'guard': 'none', 'kinneret': 'reached in every sampler iteration'},
    {'line': 1825, 'reads': 'reconst', 'guard': 'none; stops in R when reconst has zero rows', 'kinneret': 'reached; written to posterior$youngest_values[i,] at mcmc_conduction.R:420'},
    {'line': '1831-1837', 'reads': 'predictant_tann, predictant_pann', 'guard': 'taxa_change', 'kinneret': 'not reached: taxa_change stays FALSE; its only TRUE assignment is 1501 behind `& FALSE`'},
    {'line': '1850, 1851', 'reads': 'predictant_tann, predictant_pann', 'guard': 'recent', 'kinneret': 'reached when recent is TRUE; recent needs an age <= 11000 (mcmc_prepare.R:728, 1184), open without the final ages'},
    {'line': '1867-1902', 'reads': 'predictant_tann, predictant_pann', 'guard': 'past_climate_module', 'kinneret': 'not reached: main_v1.2.R:670 sets past_climate_module to F'},
    {'line': '1923-1929', 'reads': 'reconst_tann, reconst_pann (never assigned in a loaded file)', 'guard': 'stad_interstad', 'kinneret': 'not reached: mcmc_prepare.R:468 sets stad_interstad to FALSE unconditionally'},
    {'line': 1940, 'reads': 'predictant_tann', 'guard': 'tann', 'kinneret': 'not reached: mcmc_prepare.R:1249 sets tann to FALSE for Kinneret'},
    {'line': 1942, 'reads': 'predictant_pann', 'guard': '!tann', 'kinneret': 'reached'},
    {'line': '1964, 1966', 'reads': 'predictant_current, predictant_tann', 'guard': 'multiple', 'kinneret': 'reached when sources$core is 1, see the local-cor packet'},
    {'line': 1993, 'reads': 'predictant_tann', 'guard': '!multiple & sources[,4]==1', 'kinneret': 'reached only if sources$core is 0'},
    {'line': 1999, 'reads': 'predictant_current', 'guard': '!multiple & sources[,4]!=1', 'kinneret': 'not reached: sources[,4] stays 1'},
    {'line': '1982, 2009', 'reads': 'predictant_pann', 'guard': 'hi_time', 'kinneret': 'not reached: add.sites <- T sets hi_time FALSE'},
    {'line': 2023, 'reads': 'predictant_current', 'guard': 'spectral', 'kinneret': 'depends on sources$speleothem; needs wt_power, not part of this module'},
    {'line': 2031, 'reads': 'predictant_current', 'guard': 'spectral & hi_spec', 'kinneret': 'not reached: mcmc_prepare.R:881 excludes Kinneret from hi_spec'},
]


def line_numbers(lines, pattern):
    return [n for n, line in enumerate(lines, 1) if re.search(pattern, line)]


def r_source_facts(path):
    """Line numbers of the matrix product in the R 3.5.0 sources. Only numbers and booleans leave this function."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TARBALL_SHA256:
        raise ValueError('R-3.5.0.tar.gz differs from the bound download')
    with tarfile.open(path, 'r:gz') as tar:
        members = {name: tar.extractfile('R-3.5.0/' + name).read() for name in ('src/main/array.c', 'src/extra/blas/blas.f', 'src/library/base/man/options.Rd')}
    array = members['src/main/array.c'].decode('utf-8', errors='replace').splitlines()
    index = {name: next(i for i, line in enumerate(array) if line.startswith('static void ' + name + '(')) for name in ('internal_matprod', 'simple_matprod', 'matprod')}
    first_type = {name: array[index[name] + 3].strip() for name in ('internal_matprod', 'simple_matprod')}
    if first_type != {'internal_matprod': 'LDOUBLE sum;', 'simple_matprod': 'double sum;'}:
        raise ValueError(f'accumulator types differ: {first_type}')
    body = array[index['matprod']:index['matprod'] + 55]
    calls = {key: next(i for i, line in enumerate(body) if text in line) + index['matprod'] + 1 for key, text in (
        ('zero_extent', 'zero-extent operations'), ('nan_test', 'mayHaveNaNOrInf(x, NRX*ncx)'), ('simple_call', 'simple_matprod(x, nrx, ncx, y, nry, ncy, z);'),
        ('dgemv_n', 'F77_CALL(dgemv)(transN'), ('dgemv_t', 'F77_CALL(dgemv)(transT'), ('dgemm', 'F77_CALL(dgemm)(transN'))}
    blas = members['src/extra/blas/blas.f'].decode('utf-8', errors='replace').splitlines()
    dgemm = next(i for i, line in enumerate(blas) if re.match(r'\s+SUBROUTINE DGEMM', line))
    update = next(i for i in range(dgemm, len(blas)) if 'C( I, J ) = C( I, J ) + TEMP*A( I, L )' in blas[i])
    zero_test_commented = any(blas[i].startswith('c') and 'B( L, J ).NE.ZERO' in blas[i] for i in range(update - 4, update))
    options = members['src/library/base/man/options.Rd'].decode('utf-8', errors='replace')
    flat = re.sub(r'\\code\{([^}]*)\}', r'\1', re.sub(r'\s+', ' ', options))
    phrase = 'uses (only) a double accumulator'
    return {'tarball': 'https://cran.r-project.org/src/base/R-3/R-3.5.0.tar.gz', 'tarball_sha256': TARBALL_SHA256,
            'file_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in members.items()},
            'array_c_lines': {'internal_matprod': index['internal_matprod'] + 1, 'internal_matprod_accumulator': 'LDOUBLE', 'simple_matprod': index['simple_matprod'] + 1, 'simple_matprod_accumulator': 'double', 'matprod': index['matprod'] + 1, **calls},
            'blas_f_lines': {'dgemm': dgemm + 1, 'dgemm_update': update + 1}, 'dgemm_zero_test_commented_out': zero_test_commented,
            'options_rd_states_double_accumulator_for_three_loop_path': phrase in flat,
            'route_for_this_call': 'x is taxa rows, y is the num_taxa x 2 tf_sample, so ncy is 2: dgemm for more than one row, dgemv with TRANS T for one row, simple_matprod whenever the NaN test fires',
            'correction_of_the_handover': 'The handover said the three-loop path uses long double. R 3.5.0 has two: internal_matprod (LDOUBLE, only with options(matprod = "internal")) and simple_matprod (double, the default path for NaN/Inf).'}


def main():
    spec = importlib.util.spec_from_file_location('ws_bind', OUT.parent / 'literal_weights_stage_2026_11_03/bind_source.py')
    binder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binder)
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    lines, code = {}, {}
    for rel in [*SOURCES, *ALSO_LOADED]:
        raw = (base / rel).read_bytes()
        if rel in SOURCES and hashlib.sha256(raw).hexdigest() != SOURCES[rel]:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8-sig', errors='replace').splitlines()
        code[rel] = binder._code(lines[rel])
    for rel, expected in LITERAL.items():
        for n, text in expected:
            if lines[rel][n - 1].strip() != text:
                raise ValueError(f'{rel}:{n} no longer reads {text!r}: {lines[rel][n - 1].strip()!r}')
    occurrences = {}
    for name in ('reconst_ages', 'reconst', 'reconst_tann', 'reconst_pann', 'taxa_change'):
        found = {rel: line_numbers(stripped, binder._name(name)) for rel, stripped in code.items()}
        occurrences[name] = {rel: v for rel, v in found.items() if v}
        if occurrences[name] != EXPECTED[name]:
            raise ValueError(f'{name} occurrences differ: {occurrences[name]}')
    depth = binder._depths(code[F])
    if depth[854] != 0 or depth[1780] != 0:
        raise ValueError('a reconstruction definition is not at top level')
    definitions = [n for n, line in enumerate(code[F], 1) if re.match(r'\s*reconstruction\s*<-\s*function', line)]
    if definitions != [855, 1781]:
        raise ValueError(f'reconstruction definitions differ: {definitions}')
    end = next(n for n in range(1782, len(code[F]) + 1) if depth[n - 1] > 0 and (n == len(code[F]) or depth[n] == 0))
    body = {n: code[F][n - 1] for n in range(ACTIVE_BODY_FIRST, end + 1)}
    tf_formal = [n for n, line in body.items() if re.search(binder._name('tf_sample'), line)]
    if tf_formal:
        raise ValueError(f'the formal tf_sample is read at {tf_formal}')
    reads = {name: [n for n, line in body.items() if re.search(binder._name(name), line)] for name in ('predictant_tann', 'predictant_pann')}
    if reads != ACTIVE_BODY_READS_OF_PREDICTANTS:
        raise ValueError(f'predictant occurrences differ: {reads}')
    current = [n for n, line in body.items() if re.search(binder._name('predictant_current'), line)]
    if current != [1940, 1942, 1964, 1999, 2023, 2031]:
        raise ValueError(f'predictant_current occurrences differ: {current}')
    outside = {rel: line_numbers(stripped, r'(?<![\w.$])predictant_(tann|pann|current)(?![\w.])') for rel, stripped in code.items() if rel != F}
    outside = {rel: v for rel, v in outside.items() if v}
    if outside != {C: [110]}:
        raise ValueError(f'predictants occur outside the functions file: {outside}')
    youngest = {rel: line_numbers(stripped, r'youngest_values') for rel, stripped in code.items()}
    youngest = {rel: v for rel, v in youngest.items() if v}
    age_writes = {rel: [n for n, line in enumerate(stripped, 1) if re.search(r'(?<![\w.$])age\s*<-', line)] for rel, stripped in code.items()}
    age_writes = {rel: v for rel, v in age_writes.items() if v}
    if set(age_writes) != {A}:
        raise ValueError(f'age is written outside check_ages.R: {age_writes}')
    source_ages = next(n for n, line in enumerate(lines[P], 1) if 'source("src/check_ages.R")' in line)
    if source_ages >= 91:
        raise ValueError('check_ages.R is not sourced before all.avail.age is built')
    past ={rel: [n for n, line in enumerate(stripped, 1) if re.search(binder._name('past_climate_module') + r'\s*<-', line)] for rel, stripped in code.items()}
    past = {rel: v for rel, v in past.items() if v}
    stad = {rel: [n for n, line in enumerate(stripped, 1) if re.search(binder._name('stad_interstad') + r'\s*<-', line)] for rel, stripped in code.items()}
    stad = {rel: v for rel, v in stad.items() if v}
    if past != {M: [670]} or stad != {P: [463, 465, 468]}:
        raise ValueError(f'flag writes differ: {past}, {stad}')
    final_callers = {rel: line_numbers(stripped, binder._name('final_reconstruction') + r'\s*\(') for rel, stripped in code.items()}
    final_callers = {rel: v for rel, v in final_callers.items() if v}
    r_facts = r_source_facts(sys.argv[1]) if len(sys.argv) > 1 else None
    span = lines[F][1810:1825]
    (OUT / 'source_excerpt.json').write_text(json.dumps({'status': 'read_only_spans', 'spans': [
        {'source': 'original:src/mcmc_functions_v1.6.R', 'first_line': 1811, 'last_line': 1825, 'role': 'implemented_1817_1820_and_1825_given_tf_sample', 'text': '\n'.join(span)},
        {'source': 'original:src/mcmc_functions_v1.6.R', 'first_line': 1829, 'last_line': 1857, 'role': 'implemented_branch_recent_taxa_change_false', 'text': '\n'.join(lines[F][1828:1857])},
        {'source': 'original:src/mcmc_functions_v1.6.R', 'first_line': 1939, 'last_line': 1943, 'role': 'implemented_predictant_current', 'text': '\n'.join(lines[F][1938:1943])},
        {'source': 'original:src/mcmc_prepare.R', 'first_line': 1462, 'last_line': 1470, 'role': 'mask', 'text': '\n'.join(lines[P][1461:1470])},
        {'source': 'original:src/mcmc_prepare.R', 'first_line': 1500, 'last_line': 1500, 'role': 'mask_branch', 'text': lines[P][1499]},
        {'source': 'original:src/mcmc_prepare.R', 'first_line': 1521, 'last_line': 1522, 'role': 'mask_reset', 'text': '\n'.join(lines[P][1520:1522])}]},
        indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'source_binding.json').write_text(json.dumps({
        'status': 'bounded_historical_source_fragment_not_case_reproduction', 'basis_commit': BASIS, 'sources_sha256': SOURCES,
        'implemented_spans': [[F, 1817, 1820], [F, 1825, 1825], [F, 1829, 1857], [F, 1939, 1943]], 'left_out': {'1812-1813': 'rtruncnorm_cpp draws; tf_sample is an explicit input'},
        'active_definition': {'line': 1781, 'replaces': 855, 'called_at': 'mcmc_conduction.R:146', 'body_last_line': end},
        'formal_tf_sample_read_in_body': False, 'formal_tf_sample_note': 'line 146 passes tf_x12_mean_current[posterior_temp$z,] as the formal tf_sample; the body reads only posterior_temp$tf_sample, the formal stays an unevaluated promise',
        'mask': {'writes': occurrences['reconst_ages'][P], 'reads': occurrences['reconst_ages'][F], 'reads_in_active_reconstruction': [1818],
                 'effect': 'always all TRUE: 1464 for !taxa.merge; for taxa.merge 1466 is overwritten at 1522 because line 1500 ends in `& FALSE`',
                 'length_binding': 'all.avail.age has length(age) rows (mcmc_prepare.R:91), check_ages.R is sourced before that (line %d) and is the only writer of age, so mask length equals nrow' % source_ages,
                 'consequence': 'a single-row or row-dropping selection needs length(age) == 1 in the live path; ages_hol at 1466 never reaches line 1818'},
        'matrix_source': {'parameter_stage': 'mcmc_conduction.R:58 parameter_stage_weights (gibbs F)', 'written_at': 'mcmc_functions_v1.6.R:1277', 'dimnames': 'none: built by matrix() at 1276',
                          'may_hold_missing_values': 'yes in the source domain: rowSums(na.rm=T) and row division (weights-stage packet); depends on final inputs'},
        'tf_sample_source': {'filled_at': [1812, 1813], 'random': True, 'initial': 'logical NA array at mcmc_conduction.R:115, replaced by doubles before line 1818'},
        'occurrences': occurrences, 'active_body_reads': reads, 'predictant_current_occurrences': current,
        'global_shadowed': {'mcmc_conduction.R:109-110': 'reconst and predictant_* are assigned as globals and never read; the function assigns locals'},
        'youngest_values': youngest, 'age_writes': age_writes, 'flag_writes': {'past_climate_module': past, 'stad_interstad': stad},
        'final_reconstruction_callers_in_loaded_files': final_callers,
        'readers': READERS,
        'kinneret_flags': {'add.sites': True, 'case_studies': False, 'past_climate_module': False, 'stad_interstad': False, 'taxa_change': False, 'tann': False, 'hi_time': False, 'hi_spec': False},
        'open_conditions': ['recent needs at least one age <= 11000 (mcmc_prepare.R:728, 1184); not evaluated without the final v1.2 ages',
                            'sources$core decides 1964/1966 versus 1993, as in the local-cor packet',
                            'whether taxa.appear.core.temp holds NA, NaN or Inf depends on the final pollen matrix'],
        'r_matprod_source': r_facts, 'packages_needed': [], 'base_functions_used': ['[', '%*%', '$'], 'globals_made_explicit': ['reconst_ages', 'taxa_change', 'recent', 'prior.recent', 'tann'],
        'portable_excerpt': 'source_excerpt.json', 'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'actual_original_inputs_used_for_numerics': False, 'run_binding': None, 'historical_reproduction_ready': False,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps({
        'kind': 'determinism_audit_from_source_text_and_R_cross_check',
        'question': 'Which part of the reconstruction step is deterministic given tf_sample, and who reads its result for Kinneret?',
        'candidates': [{'id': 'selection_and_product', 'lines': 'mcmc_functions_v1.6.R:1817-1820', 'random_state': False, 'package_semantics_needed': False, 'reached_by_bound_run': True, 'decision': 'implemented with explicit matrices'},
                       {'id': 'youngest_values', 'lines': '1825', 'random_state': False, 'reached_by_bound_run': True, 'decision': 'implemented'},
                       {'id': 'taxa_change_block', 'lines': '1829-1839', 'random_state': True, 'reached_by_bound_run': False, 'decision': 'not implemented; rejected when requested'},
                       {'id': 'recent_values', 'lines': '1844-1857', 'random_state': False, 'reached_by_bound_run': 'when recent is TRUE', 'decision': 'implemented, both branches'},
                       {'id': 'predictant_current', 'lines': '1939-1943', 'random_state': False, 'reached_by_bound_run': True, 'decision': 'implemented'},
                       {'id': 'proposal_draws', 'lines': '1812-1813', 'random_state': True, 'reached_by_bound_run': True, 'decision': 'given explicitly; no draws'}],
        'selected_for_implementation': 'selection, matrix product, column split and direct readers', 'final_variant_selected': False, 'R_execution_for_binding': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('body end', end, 'youngest', youngest, 'age writes', age_writes, 'final callers', final_callers)
    print('source_ages line', source_ages, 'r facts', r_facts and r_facts['array_c_lines'], r_facts and r_facts['blas_f_lines'], r_facts and r_facts['options_rd_states_double_accumulator_for_three_loop_path'], r_facts and r_facts['dgemm_zero_test_commented_out'])


if __name__ == '__main__':
    main()
