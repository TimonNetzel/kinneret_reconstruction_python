# Local cor (mcmc_functions_v1.6.R 2355-2366) and the time-space block (1939-2018), both read verbatim, on artificial inputs in R 3.5.0.
# Arguments: <input .R> <output file>. No package is attached.
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
fn_env <- new.env(parent = globalenv())
eval(parse(text = paste(code[2355:2366], collapse = "\n")), envir = fn_env)
block <- parse(text = paste(code[1939:2018], collapse = "\n"))

for (name in names(cases)) {
  c1 <- cases[[name]]
  got <- tryCatch(fn_env$cor(c1$x, c1$y, na.rm = c1$na_rm), error = function(e) NULL)
  if (is.null(got)) writeLines(paste(name, "cor", "ERROR", sep = "\t"), out) else {
    emit(name, "cor", got)
    writeLines(paste(name, "dim", paste(dim(got), collapse = "x"), sep = "\t"), out)
  }
}
for (name in names(readers)) {
  case_env <- new.env(parent = fn_env)
  for (key in names(readers[[name]])) assign(key, readers[[name]][[key]], envir = case_env)
  case_env$posterior_temp <- list()
  for (e in block) eval(e, envir = case_env)
  for (key in c("post_pollen", "post_iso", "post_iso_hi")) emit(name, key, case_env$posterior_temp[[key]])
  writeLines(paste(name, "post_pollen_dim", paste(dim(case_env$posterior_temp$post_pollen), collapse = "x"), sep = "\t"), out)
}
emit("constants", "one_e_minus_10", 1e-10)
emit("constants", "squared", (1e-10)^2)
emit("constants", "ninety_five", 0.95)
emit("constants", "half", 0.5)
writeLines(paste("info", R.version.string, environmentName(environment(fn_env$cor)), "cor_is_closure_in_test_env", exists("cor", envir = fn_env, inherits = FALSE),
                 RNGkind()[1], sep = "\t"), out)
close(out)
