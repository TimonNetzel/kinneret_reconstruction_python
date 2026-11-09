# ratios_tfs_no_ages on artificial inputs in R 3.5.0. The C++ of mcmc_functions_v1.6.R:1485-1525 needs a compiler and Rcpp, neither is available here.
# Run instead, read verbatim: the R twin of the same function in the comment lines 1529-1552, and the call and reader mcmc_conduction.R:228 and 230.
# dtnorm below is a short R statement of the truncated normal density; pnorm and dnorm are the functions of R 3.5.0. Arguments: <input .R> <output file>.
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
code <- readLines(source_file, warn = FALSE, encoding = "UTF-8")
conduction <- readLines(conduction_file, warn = FALSE, encoding = "UTF-8")
env0 <- new.env(parent = globalenv())
env0$dtnorm <- function(x, mean, sd, lower, upper) {
  if (x < lower || x > upper) return(0)
  dnorm(x, mean, sd) / (pnorm(upper, mean, sd) - pnorm(lower, mean, sd))
}
eval(parse(text = paste(sub("^# ?", "", code[1529:1552]), collapse = "\n")), envir = env0)
call_block <- parse(text = conduction[228])
reader_block <- parse(text = conduction[230])

run_case <- function(name, c1) {
  env <- new.env(parent = env0)
  n <- c1$num_taxa
  env$num_taxa <- n; env$tann_range <- c1$tann_range; env$pann_range <- c1$pann_range
  env$taxa_tann_dens <- c1$tann_dists; env$taxa_pann_dens <- c1$pann_dists
  env$tf_quarts_tann <- cbind(c1$lower_tann, 0, c1$upper_tann); env$tf_quarts_pann <- cbind(c1$lower_pann, 0, c1$upper_pann)
  env$sds_tann <- c1$sds_tann_trunc; env$sds_pann <- c1$sds_pann_trunc
  env$i <- 2L
  env$posterior_temp <- list(tf_sample = cbind(c1$new_tann, c1$new_pann))
  env$posterior <- list(tf_sample = list(cbind(c1$old_tann, c1$old_pann)), transfer_ratio = numeric(2))
  eval(call_block[[1]], envir = env)
  eval(reader_block[[1]], envir = env)
  r <- env$ratios_tfs
  emit(name, "tann_tf_ratio", r[[1]]); emit(name, "pann_tf_ratio", r[[2]]); emit(name, "tann_dists_tf_ratio", r[[3]]); emit(name, "pann_dists_tf_ratio", r[[4]])
  emit(name, "transfer_ratio", env$posterior$transfer_ratio[2])
  index <- dens <- p_a <- p_b <- value <- c()
  for (i in seq_len(n)) {
    index <- c(index, which.min(abs(c1$tann_range - c1$new_tann[i])), which.min(abs(c1$tann_range - c1$old_tann[i])), which.min(abs(c1$pann_range - c1$new_pann[i])), which.min(abs(c1$pann_range - c1$old_pann[i])))
    calls <- list(c(c1$old_tann[i], c1$new_tann[i], c1$sds_tann_trunc[i], c1$lower_tann[i], c1$upper_tann[i]), c(c1$new_tann[i], c1$old_tann[i], c1$sds_tann_trunc[i], c1$lower_tann[i], c1$upper_tann[i]),
                  c(c1$old_pann[i], c1$new_pann[i], c1$sds_pann_trunc[i], c1$lower_pann[i], c1$upper_pann[i]), c(c1$new_pann[i], c1$old_pann[i], c1$sds_pann_trunc[i], c1$lower_pann[i], c1$upper_pann[i]))
    for (a in calls) {
      dens <- c(dens, dnorm(a[1], a[2], a[3])); p_a <- c(p_a, pnorm(a[4], a[2], a[3])); p_b <- c(p_b, pnorm(a[5], a[2], a[3]))
      value <- c(value, env0$dtnorm(a[1], a[2], a[3], a[4], a[5]))
    }
  }
  writeLines(paste(name, "indices", paste(index - 1L, collapse = " "), sep = "\t"), out)
  emit(name, "dnorm", dens); emit(name, "p_a", p_a); emit(name, "p_b", p_b); emit(name, "d_truncnorm", value)
}

for (name in names(cases)) run_case(name, cases[[name]])
for (name in names(stress)) run_case(name, stress[[name]])
writeLines(paste("info", R.version.string, environmentName(environment(pnorm)), environmentName(environment(dnorm)), capabilities("long.double"), sep = "\t"), out)
close(out)
