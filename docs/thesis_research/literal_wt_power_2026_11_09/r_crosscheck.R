# wt_power: mcmc_functions_v1.6.R 1003-1012 and 2023-2026 with the local cor (2355-2366), read verbatim, on artificial series in R 3.5.0.
# mcmc_prepare.R 1816-1817 give length_ages_spec2. Arguments: <input .R> <output file>. No package is attached.
args <- commandArgs(trailingOnly = TRUE)
source(args[1], local = TRUE)
out <- file(args[2], open = "w", encoding = "UTF-8")
token <- function(v) {
  if (is.nan(v)) return("NaN")
  if (is.na(v)) return("NA")
  if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
  if (v == 0) return(if (1 / v < 0) "-0" else "0")
  sprintf("%a", v)
}
emit <- function(case, key, x) writeLines(paste(case, key, paste(vapply(as.numeric(x), token, ""), collapse = " "), sep = "\t"), out)
line <- function(case, key, text) writeLines(paste(case, key, text, sep = "\t"), out)
code <- readLines(source_file, warn = FALSE, encoding = "UTF-8")
prep <- readLines(prepare_file, warn = FALSE, encoding = "UTF-8")
wt_env <- new.env(parent = globalenv())
block <- function(lines, a, b) parse(text = paste(lines[a:b], collapse = "\n"))
eval(block(code, 1003, 1012), envir = wt_env)
eval(block(code, 2355, 2366), envir = wt_env)
body_block <- block(code, 1004, 1010)
reader_block <- block(code, 2023, 2026)
length_block <- block(prep, 1816, 1817)

run_case <- function(name, c1, steps = TRUE) {
  wt_env$wb <- c1$wb
  env <- new.env(parent = wt_env)
  env$d <- c1$d; env$l_d <- c1$l_d; env$n.obs <- c1$n_obs; env$zeros <- rep(0, c1$zeros); env$invflen <- c1$invflen
  for (e in body_block) eval(e, envir = env)
  emit(name, "x", env$x)
  emit(name, "f_re", Re(env$f)); emit(name, "f_im", Im(env$f))
  line(name, "dim", if (is.null(dim(env$wave))) "none" else paste(dim(env$wave), collapse = "x"))
  emit(name, "wave_re", Re(env$wave)); emit(name, "wave_im", Im(env$wave))
  emit(name, "power", env$power)
  called <- wt_env$wt_power(c1$d, c1$l_d, n.obs = c1$n_obs, zeros = rep(0, c1$zeros), invflen = c1$invflen)
  line(name, "function_equals_steps", identical(called, env$power))
  line(name, "wb_type", typeof(c1$wb))
}

run_chain <- function(name, ch) {
  wt_env$wb <- ch$wb
  env <- new.env(parent = wt_env)
  env$ages_spec <- ch$ages_spec; env$hi_ages_spec <- ch$hi_ages_spec; env$predictant_current <- ch$current; env$n.obs <- ch$n_obs
  env$zeros <- rep(0, ch$zeros); env$invflen <- ch$invflen; env$coi_array <- ch$coi; env$predictors <- list(iso_spec = ch$iso)
  env$posterior_temp <- list()
  for (e in length_block) eval(e, envir = env)
  line(name, "length_ages_spec2", env$length_ages_spec2)
  for (e in reader_block) eval(e, envir = env)
  emit(name, "predictant_spec", env$predictant_spec)
  emit(name, "post_spec", env$posterior_temp$post_spec)
}

for (name in names(cases)) run_case(name, cases[[name]])
for (name in names(stress)) run_case(name, stress[[name]])
for (name in names(chains)) run_chain(name, chains[[name]])
writeLines(paste("info", R.version.string, environmentName(environment(fft)), capabilities("long.double"), sep = "\t"), out)
close(out)
