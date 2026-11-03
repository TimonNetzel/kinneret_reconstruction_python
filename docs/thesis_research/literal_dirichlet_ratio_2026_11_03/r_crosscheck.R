# Evaluate mcmc_functions_v1.6.R 1269-1274 verbatim with LaplacesDemon attached (R 3.5.0, synthetic weights only).
# Arguments: <library folder with LaplacesDemon> <input .R from run_r_crosscheck.py> <output file>.
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(LaplacesDemon))
source(args[2], local = TRUE)
out <- file(args[3], open = "w", encoding = "UTF-8")
hex <- function(x) vapply(x, function(v) {
  if (is.na(v) && !is.nan(v)) return("NA")
  if (is.nan(v)) return("NaN")
  if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
  if (v == 0) return(if (1 / v < 0) "-0" else "0")
  sprintf("%a", v)
}, "")
emit <- function(case, key, x) writeLines(paste(case, key, paste(hex(as.numeric(x)), collapse = " "), sep = "\t"), out)
code <- paste(readLines(source_file, warn = FALSE, encoding = "UTF-8")[1269:1274], collapse = "\n")

for (name in names(ratio_cases)) {
  input <- ratio_cases[[name]]
  env <- new.env(parent = globalenv())
  env$old_weights <- input$old
  env$new_weights <- input$new
  env$num_taxa <- input$num_taxa
  env$posterior_temp <- list()
  eval(parse(text = code), envir = env)
  emit(name, "weights_ratio", env$weights_ratio)
  emit(name, "proposal_ratio", env$proposal_ratio)
  emit(name, "parameter_ratio", env$posterior_temp$parameter_ratio)
  emit(name, "densities", c(ddirichlet(input$new, rep(0.5, input$num_taxa)), ddirichlet(input$old, rep(0.5, input$num_taxa)),
                            ddirichlet(input$old, input$new), ddirichlet(input$new, input$old)))
}
for (name in names(density_cases)) {
  input <- density_cases[[name]]
  emit(name, "density", ddirichlet(input$x, input$alpha))
  emit(name, "log_density", ddirichlet(input$x, input$alpha, log = TRUE))
}
writeLines(paste("R", R.version.string, as.character(packageVersion("LaplacesDemon")),
                 environmentName(environment(ddirichlet)), sep = "\t"), out)
close(out)
