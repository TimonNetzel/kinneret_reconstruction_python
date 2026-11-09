# weightedMedian(x, w) from matrixStats in R 3.5.0 on artificial inputs, and mcmc_prepare.R 1628-1640/1648-1660/1664
# verbatim on one artificial TF matrix. Arguments: <library folder> <input .R> <output file>.
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(matrixStats))
source(args[2], local = TRUE)
out <- file(args[3], open = "w", encoding = "UTF-8")
hex <- function(x) vapply(x, function(v) {
  if (is.nan(v)) return("NaN")
  if (is.na(v)) return("NA")
  if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
  if (v == 0) return(if (1 / v < 0) "-0" else "0")
  sprintf("%a", v)
}, "")
emit <- function(section, key, x) writeLines(paste(section, key, paste(hex(as.numeric(x)), collapse = " "), sep = "\t"), out)

for (name in names(cases)) emit("case", name, weightedMedian(cases[[name]]$x, cases[[name]]$w))
for (i in seq_along(stress)) emit("stress", i, weightedMedian(stress[[i]]$x, stress[[i]]$w))

code <- readLines(source_file, warn = FALSE, encoding = "UTF-8")
literal <- paste(code[c(1628:1640, 1648:1660, 1664)], collapse = "\n")
env <- new.env(parent = globalenv())
env$taxa_change <- FALSE
env$x1range.bck <- tf_case$x1range
env$x2range.bck <- tf_case$x2range
env$tf.x1.current <- tf_case$tf_x1
env$tf.x2.current <- tf_case$tf_x2
eval(parse(text = literal), envir = env)
emit("tf", "x1", env$tf.x1.current.mean)
emit("tf", "x2", env$tf.x2.current.mean)
writeLines(paste("tf", "which", paste(env$which.tf.missing, collapse = " "), sep = "\t"), out)

fn <- matrixStats::weightedMedian
writeLines(paste("info", R.version.string, as.character(packageVersion("matrixStats")),
                 environmentName(environment(fn)), packageDescription("matrixStats")$Built,
                 environmentName(environment(get("weightedMedian", envir = env))),
                 paste(names(formals(fn)), collapse = ","), sep = "\t"), out)
close(out)
