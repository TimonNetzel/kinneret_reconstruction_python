# reconst and the predictants: mcmc_functions_v1.6.R 1817-1820, 1825, 1829-1857 and 1939-1943, read verbatim, on artificial matrices in R 3.5.0.
# Six chains also run 1939-2018 with the local cor (2355-2366). Arguments: <input .R> <output file>. No package is attached.
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
fn_env <- new.env(parent = globalenv())
eval(parse(text = paste(code[2355:2366], collapse = "\n")), envir = fn_env)
block <- function(a, b) parse(text = paste(code[a:b], collapse = "\n"))
reconst_block <- block(1817, 1820)
youngest_block <- block(1825, 1825)
recent_block <- block(1829, 1857)
current_block <- block(1939, 1943)
time_block <- block(1939, 2018)

run_steps <- function(name, c1, chain = FALSE) {
  env <- new.env(parent = fn_env)
  env$posterior_temp <- list(taxa.appear.core.temp = c1$core, tf_sample = c1$tf)
  env$reconst_ages <- c1$mask
  env$taxa_change <- FALSE
  for (key in c("recent", "tann", "prior.recent", "multiple", "hi_time", "spectral", "hi_ages_time", "ages_pollen", "predictors1", "ages_iso", "predictors2",
                "sources", "ages_iso1", "ages_iso2", "predictors", "ages_pollen1", "ages_pollen2")) {
    if (!is.null(c1[[key]])) assign(key, c1[[key]], envir = env)
  }
  ok <- tryCatch({ for (e in reconst_block) eval(e, envir = env); TRUE }, error = function(err) { line(name, "stop", paste0("1818: ", conditionMessage(err))); FALSE })
  if (!ok) return(invisible())
  selected <- eval(parse(text = "taxa.appear.core[reconst_ages,]"), envir = env)
  line(name, "selected", if (is.null(dim(selected))) paste0("vector:", length(selected)) else paste(dim(selected), collapse = "x"))
  line(name, "reconst_dim", paste(dim(env$reconst), collapse = "x"))
  line(name, "reconst_dimnames", if (is.null(dimnames(env$reconst))) "none" else "some")
  emit(name, "predictant_tann", env$predictant_tann)
  emit(name, "predictant_pann", env$predictant_pann)
  ok <- tryCatch({ for (e in youngest_block) eval(e, envir = env); TRUE }, error = function(err) { line(name, "stop", paste0("1825: ", conditionMessage(err))); FALSE })
  if (!ok) return(invisible())
  emit(name, "youngest", env$posterior_temp$youngest_values)
  if (!is.null(c1$recent)) {
    for (e in recent_block) eval(e, envir = env)
    emit(name, "post_tann", env$posterior_temp$post_tann)
    emit(name, "post_pann", env$posterior_temp$post_pann)
  }
  if (chain) {
    for (e in time_block) eval(e, envir = env)
    for (key in c("post_pollen", "post_iso", "post_iso_hi")) emit(name, key, env$posterior_temp[[key]])
  } else if (!is.null(c1$tann)) {
    for (e in current_block) eval(e, envir = env)
    emit(name, "predictant_current", env$predictant_current)
    line(name, "current_is_tann", identical(env$predictant_current, env$predictant_tann))
  }
  invisible()
}

for (name in names(cases)) run_steps(name, cases[[name]])
for (name in names(stress)) run_steps(name, stress[[name]])
for (name in names(chains)) run_steps(name, chains[[name]], chain = TRUE)
writeLines(paste("info", R.version.string, getOption("matprod"), capabilities("long.double"), RNGkind()[1], unname(extSoftVersion()["BLAS"]), sep = "\t"), out)
close(out)
