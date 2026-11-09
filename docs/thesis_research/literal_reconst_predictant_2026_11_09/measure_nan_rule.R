# Which of NA and NaN survives in x %*% y? 2 x 2 matrices over six values; 1296 combinations. Also the accumulator of the three-loop path.
# Argument: <output tsv>. No package is attached; no source file of the dissertation is read here.
token <- function(v) {
  if (is.nan(v)) return("NaN")
  if (is.na(v)) return("NA")
  if (is.infinite(v)) return(if (v > 0) "Inf" else "-Inf")
  sprintf("%a", v)
}
vals <- list(one = 1, NA_ = NA_real_, NaN_ = NaN, Inf_ = Inf, MInf = -Inf, Zero = 0)
nm <- names(vals)
out <- character()
tie <- rbind(c(1, 2^-53, 2^-53), c(NaN, 0, 0))
ones <- cbind(c(1, 1, 1), c(1, 1, 1))
out <- c(out, paste("tie_with_nan_in_other_row", token((tie %*% ones)[1, 1]), sep = "\t"))
tie[2, ] <- c(1, 0, 0)
out <- c(out, paste("tie_finite", token((tie %*% ones)[1, 1]), sep = "\t"))
for (a in nm) for (b in nm) for (c1 in nm) for (d in nm) {
  x <- matrix(c(vals[[a]], vals[[b]], 1, 1), nrow = 2, byrow = TRUE)
  y <- matrix(c(vals[[c1]], 1, vals[[d]], 1), nrow = 2, byrow = TRUE)
  out <- c(out, paste(a, b, c1, d, token((x %*% y)[1, 1]), sep = "\t"))
}
writeLines(out, commandArgs(trailingOnly = TRUE)[1])
