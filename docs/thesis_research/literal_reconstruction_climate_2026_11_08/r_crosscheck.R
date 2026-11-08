# reconstruction_climate, read verbatim from mcmc_functions_v1.6.R 1188-1199, on artificial inputs in R 3.5.0.
# Arguments: <library folder with matrixStats> <input .R> <output file>. Arguments that R must not force are stop() promises.
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(matrixStats))
source(args[2], local = TRUE)
out <- file(args[3], open = "w", encoding = "UTF-8")
hex <- function(x) vapply(as.numeric(x), function(v) {
  if (is.nan(v)) return("NaN")
  if (is.na(v)) return("NA")
  if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
  if (v == 0) return(if (1 / v < 0) "-0" else "0")
  sprintf("%a", v)
}, "")
emit <- function(case, key, x) writeLines(paste(case, key, paste(hex(x), collapse = " "), sep = "\t"), out)
code <- readLines(source_file, warn = FALSE, encoding = "UTF-8")
fn_env <- new.env(parent = globalenv())
eval(parse(text = paste(code[1188:1199], collapse = "\n")), envir = fn_env)

run <- function(name, c1) {
  fn_env$x1range.bck <- c1$x1range
  fn_env$x2range.bck <- c1$x2range
  fn_env$prior.recent <- list(mean.tmp = c1$prior_tmp, mean.pann = c1$prior_pann)
  if (c1$recent) {
    core <- matrix(c1$core, nrow = c1$n_row, byrow = TRUE)
    tf1 <- matrix(c1$tf1, nrow = c1$n_taxa, byrow = TRUE)
    tf2 <- matrix(c1$tf2, nrow = c1$n_taxa, byrow = TRUE)
    got <- fn_env$reconstruction_climate(core, stop("which.taxa was read"), tf1, tf2, TRUE)
    emit(name, "tann_weights", (core[1, ] %*% tf1)[1, ])
    emit(name, "pann_weights", (core[1, ] %*% tf2)[1, ])
  } else {
    got <- fn_env$reconstruction_climate(stop("core was read"), stop("which.taxa was read"), stop("tf1 was read"), stop("tf2 was read"), FALSE)
  }
  emit(name, "tann", got$tann)
  emit(name, "pann", got$pann)
}
for (name in names(cases)) run(name, cases[[name]])
writeLines(paste("info", R.version.string, as.character(packageVersion("matrixStats")),
                 environmentName(environment(get("weightedMedian", envir = fn_env))), RNGkind()[1], sep = "\t"), out)
close(out)
