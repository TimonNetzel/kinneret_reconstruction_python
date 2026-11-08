# Evaluate the literal original lines of six checked modules on artificial inputs (R 3.5.0).
# Arguments: <library folder with matrixStats> <input .R written by run_six_crosscheck.py> <output file>.
# Only rowCumsums (lines 485-486) needs a package; every other span uses base R. No random state is touched.
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
code <- lapply(sources, function(f) readLines(f, warn = FALSE, encoding = "UTF-8"))
span <- function(file, ...) parse(text = paste(code[[file]][c(...)], collapse = "\n"))
fresh <- function() new.env(parent = globalenv())
mat <- function(values, nrow) matrix(values, nrow = nrow, byrow = TRUE)
EX <- list(pick = span("ages", 177:179), shift = span("ages", 230:232), kernel = span("prepare", 11:20),
           zeros = span("prepare", 83), trafo = span("prepare", 91:94),
           grids = span("prepare", 435:437), masks = span("prepare", 440:442), first = span("prepare", 1462:1467),
           effective = span("prepare", 1522), product = span("functions", 461:462), mean = span("functions", 470),
           scale = span("functions", 411:419), scale_age = span("functions", 422:424),
           unique = span("functions", 375), rw_depth = span("functions", 450:451), rw_age = span("functions", 454:455),
           rw_depth_tf = span("functions", 458:459), rw_age_tf = span("functions", 461:462),
           gibbs = span("functions", 434:436), gibbs_tf = span("functions", 439:443),
           add = span("functions", 543:547), take = span("functions", 712:715), norm = span("functions", 756:760),
           place = span("functions", 369:371), cum = span("functions", 485:486), emd = span("functions", 487:490),
           tf_norm = span("prepare", 1581:1586), tf_mask = span("prepare", 1613:1618),
           pruning = parse(text = paste0("skipped <- TRUE\nfor (.once in 1L) {\n",
                                         paste(code$prepare[51:80], collapse = "\n"), "\nskipped <- FALSE\n}")))

for (name in names(cases$age_depth)) {
  c1 <- cases$age_depth[[name]]
  e <- fresh()
  rho <- mat(c1$rho, c1$n_age)
  e$core.data <- list(list(rho_depth_given_age = list(rho * 2, rho), depth = c1$depth, age = list(c1$age + 1, c1$age)))
  e$station_id <- 1L
  e$age.reso <- 2L
  eval(EX$pick, e)
  emit(name, "picked_rho", t(e$rho_depth_given_age))
  emit(name, "picked_age", e$age)
  before <- e$site.age
  eval(EX$shift, e)
  emit(name, "shifted_age", e$age)
  emit(name, "site_age_untouched", e$site.age)
  eval(EX$kernel, e)
  emit(name, "weights", t(e$rho.d.g.a))
  e$all.avail <- mat(c1$pollen, length(c1$depth))
  eval(EX$zeros, e)
  emit(name, "pollen_zero_replaced", t(e$all.avail))
  eval(EX$trafo, e)
  emit(name, "pollen_at_ages", t(e$all.avail.age))
}

for (name in names(cases$selection)) {
  c1 <- cases$selection[[name]]
  e <- fresh()
  e$age <- c1$age
  eval(EX$grids, e)
  eval(EX$masks, e)
  for (k in c("hol", "lg", "eem")) {
    emit(name, paste0("grid_", k), e[[k]])
    emit(name, paste0("mask_", k), e[[paste0("ages_", k)]])
  }
  e$taxa.merge <- c1$taxa_merge
  eval(EX$first, e)
  emit(name, "initial_mask", e$reconst_ages)
  initial <- e$reconst_ages
  eval(EX$effective, e)
  emit(name, "effective_mask", e$reconst_ages)
  effective <- e$reconst_ages
  e$taxa.spectrum.age <- mat(c1$coefficients, c1$n_row)
  e$tf.x1.current <- e$tf.x2.current <- e$tf_x12_mean_current <- mat(c1$tf, c1$n_tf)
  if (sum(initial) >= 2) {
    e$reconst_ages <- initial
    eval(EX$product, e)
    emit(name, "product_initial", t(e$tmp.recon.age))
  }
  e$reconst_ages <- effective
  eval(EX$product, e)
  emit(name, "product_effective", t(e$tmp.recon.age))
  emit(name, "product_effective_pann", t(e$pann.recon.age))
  eval(EX$mean, e)
  emit(name, "product_effective_mean", t(e$reconst_mean))
}

for (name in names(cases$final_pdf)) {
  c1 <- cases$final_pdf[[name]]
  e <- fresh()
  e$all.avail <- e$all.avail.age <- mat(c1$available, c1$n_row)
  e$num_taxa <- c1$n_taxa
  e$taxa_weights <- mat(c1$weights, c1$n_draw)
  e$tf.x1.current <- e$tf.x2.current <- mat(c1$tf, c1$n_taxa)
  e$reconst_ages <- rep(TRUE, c1$n_row)
  e$energy <- c1$energy
  eval(EX$scale, e)
  eval(EX$scale_age, e)
  emit(name, "scaled", t(e$all.avail.temp))
  eval(EX$unique, e)
  emit(name, "unique_ids", e$combs_unique)
  zeros <- array(0, dim = c(c1$n_row, ncol(e$tf.x1.current)))
  e$tmp_mixt_avg_pdfs_depth <- e$pann_mixt_avg_pdfs_depth <- e$tmp_mixt_avg_pdfs_age <- e$pann_mixt_avg_pdfs_age <- zeros
  for (i in seq_len(c1$n_draw)) {
    e$i <- i
    eval(EX$rw_depth, e)
    eval(EX$rw_age, e)
    eval(EX$rw_depth_tf, e)
    eval(EX$rw_age_tf, e)
    emit(name, paste0("coefficients_", i), t(e$taxa.spectrum.depth))
    emit(name, paste0("density_", i), t(e$tmp.recon.depth))
    if (i %in% e$combs_unique) eval(EX$add, e)
  }
  eval(EX$take, e)
  eval(EX$norm, e)
  emit(name, "pdf_sum_depth", t(e$tmp.recon.depth.mixt))
  emit(name, "pdf_sum_age", t(e$tmp.recon.age.mixt))
  if (length(c1$gibbs_ids) >= 2) {
    e$post.which.combs <- matrix(c1$gibbs_ids, nrow = 1)
    e$which.taxa.unique <- array(NA, dim = c(1, length(c1$gibbs_ids)))
    e$count <- 1L
    e$i <- 1L
    e$tf_x12_mean_current <- e$tf.x1.current
    eval(EX$gibbs, e)
    eval(EX$gibbs_tf, e)
    emit(name, "gibbs_coefficients", t(e$pollen_info_depth))
    emit(name, "gibbs_density", t(e$tmp.recon.depth))
  }
}

for (name in names(cases$cdf)) {
  c1 <- cases$cdf[[name]]
  e <- fresh()
  e$rho.d.g.a <- array(1, dim = c(c1$n_row, 1))
  e$dims <- c1$inherited_dims
  if (c1$placeholder) {
    e$ecdf_ref1 <- 0
    e$ecdf_ref2 <- "not inspected"
  } else {
    e$ecdf_ref1 <- e$ecdf_ref2 <- mat(c1$reference, c1$n_row)
  }
  eval(EX$place, e)
  emit(name, "reference_first", t(e$ecdf_ref1))
  emit(name, "reference_second", t(e$ecdf_ref2))
  e$tmp.recon.age <- e$pann.recon.age <- mat(c1$density, c1$n_row)
  eval(EX$cum, e)
  emit(name, "cumulative", t(e$ecdf_tmp_my_reconst))
  e$dims <- c1$current_dims
  e$age <- seq_len(c1$n_row)
  e$count <- 1L
  e$emd_tmp <- e$emd_pann <- array(NA, dim = c(1L, c1$n_row))
  eval(EX$emd, e)
  emit(name, "score", e$emd_tmp[1, ])
}

for (name in names(cases$tf)) {
  c1 <- cases$tf[[name]]
  e <- fresh()
  e$taxa_change <- FALSE
  e$tf.x1.current <- e$tf.x2.current <- mat(c1$raw, c1$n_row)
  eval(EX$tf_norm, e)
  emit(name, "normalized", t(e$tf.x1.current))
  e$x1range.bck <- e$x2range.bck <- NULL
  e$tann_tf_range <- e$pann_tf_range <- c1$keep
  e$inside.range <- function(x, r) r  # the explicit artificial mask stands in for the spatstat.utils test
  eval(EX$tf_mask, e)
  emit(name, "masked", t(e$tf.x1.current))
}

for (name in names(cases$pruning)) {
  c1 <- cases$pruning[[name]]
  e <- fresh()
  e$all.avail <- mat(c1$pollen, c1$n_row)
  e$null.ratio.check <- c1$check
  e$metro_hastings <- c1$metro
  e$sitename <- if (c1$relict) "Lake Ohrid" else "Kunstort"
  eval(EX$pruning, e)
  emit(name, "skipped", e$skipped)
  emit(name, "metro_hastings", e$metro_hastings)
  emit(name, "pollen", t(e$all.avail))
  emit(name, "pollen_dim", dim(e$all.avail))
  if (c1$check) {
    emit(name, "null_ratio", e$null.ratio)
    emit(name, "removed", e$rm.id)
    if (!e$skipped) emit(name, "spectrum", e$sorting.available.spectrum)
  }
}
writeLines(paste("info", R.version.string, .Machine$sizeof.longdouble, as.character(packageVersion("matrixStats")),
                 environmentName(environment(rowCumsums)), sep = "\t"), out)
close(out)
