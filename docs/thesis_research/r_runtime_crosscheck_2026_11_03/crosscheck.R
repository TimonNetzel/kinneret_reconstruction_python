# Evaluate the literal original lines on artificial inputs; base R only, no package is attached.
# Arguments: <input .R written by run_crosscheck.py> <output file>. The input script defines `sources`
# (paths of the hash-checked originals) and `cases`. Doubles travel as C99 hex literals both ways, so the
# comparison in Python is bit exact; NA, NaN, Inf, -Inf and the sign of zero are kept apart.
args <- commandArgs(trailingOnly = TRUE)
source(args[1], local = TRUE)
out <- file(args[2], open = "w", encoding = "UTF-8")

hex <- function(x) {
  vapply(x, function(v) {
    if (is.na(v) && !is.nan(v)) return("NA")
    if (is.nan(v)) return("NaN")
    if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
    if (v == 0) return(if (1 / v < 0) "-0" else "0")
    sprintf("%a", v)
  }, "")
}
emit <- function(case, key, x) writeLines(paste(case, key, paste(hex(as.numeric(x)), collapse = " "), sep = "\t"), out)
lines_of <- function(file, first, last) paste(readLines(file, warn = FALSE, encoding = "UTF-8")[first:last], collapse = "\n")

conduction_53_79 <- lines_of(sources$conduction, 53, 79)
weights_1276_1280 <- lines_of(sources$functions, 1276, 1280)

for (name in names(cases$sampler)) {
  input <- cases$sampler[[name]]
  env <- new.env(parent = baseenv())
  env$gibbs <- input$gibbs
  env$num_taxa <- input$num_taxa
  env$taxa.appear.core <- matrix(input$values, nrow = input$nrow, ncol = input$ncol, byrow = TRUE)
  env$posterior <- list()
  env$parameter_stage_weights <- "weights"
  env$parameter_stage_gibbs <- "gibbs"
  eval(parse(text = conduction_53_79), envir = env)
  emit(name, "how_many_taxa", env$posterior$how_many_taxa)
  emit(name, "values", env$taxa_information$values)
  emit(name, "taxa_means", env$taxa_information$taxa_means)
  emit(name, "jeffreys_prior", env$posterior$jeffreys_prior)
  writeLines(paste(name, "parameter_stage", env$parameter_stage, sep = "\t"), out)
}

for (name in names(cases$weights)) {
  input <- cases$weights[[name]]
  env <- new.env(parent = baseenv())
  env$taxa_information <- list(values = input$values, ncol_taxa_array = input$ncol, nrow_taxa_array = input$nrow)
  env$new_weights <- input$new_weights
  env$posterior_temp <- list()
  eval(parse(text = weights_1276_1280), envir = env)
  emit(name, "unnormalised", t(env$taxa.appear.core.temp))
  emit(name, "row_sums", rowSums(env$taxa.appear.core.temp, na.rm = TRUE))
  emit(name, "result", t(env$posterior_temp$taxa.appear.core.temp))
  emit(name, "z", env$posterior_temp$z)
}
prepare_1270_1281 <- lines_of(sources$prepare, 1270, 1281)
for (name in names(cases$gamma)) {
  input <- cases$gamma[[name]]
  env <- new.env(parent = baseenv())
  env$prior.recent <- list(mean.pann = input$mean_pann, sd.pann = input$sd_pann)
  eval(parse(text = prepare_1270_1281), envir = env)
  emit(name, "intersect_id", env$intersect_id)
  emit(name, "shape", env$shape)
  emit(name, "rate", env$rate)
}

prepare_1681_1685 <- lines_of(sources$prepare, 1681, 1685)
for (n in cases$priors) {
  env <- new.env(parent = baseenv())
  env$num_taxa <- n
  env$prior <- list()
  env$posterior <- list(nr.taxa.mean = NA)
  eval(parse(text = prepare_1681_1685), envir = env)
  name <- paste0("priors_", n)
  emit(name, "which_taxa", env$which_taxa)
  emit(name, "taxa_weights", env$prior$taxa.weights)
  emit(name, "phi", env$prior$phi)
  emit(name, "nr_taxa_mean", env$prior$nr.taxa.mean)
  emit(name, "nr_taxa_sd", env$prior$nr.taxa.sd)
}
writeLines(paste("R", R.version.string, .Machine$sizeof.longdouble, sep = "\t"), out)
close(out)
