# Independent R benchmark: fit glm() on the same split as Python, using base R only.
# Run from the repository root: Rscript r/glm_benchmark.R
df <- read.csv("data/credit_default.csv")
split <- read.csv("data/split.csv")
df <- merge(df, split, by = "ID")
df$EDUCATION[!(df$EDUCATION %in% c(1, 2, 3))] <- 4
df$MARRIAGE[!(df$MARRIAGE %in% c(1, 2))] <- 3

amounts <- c("LIMIT_BAL", paste0("BILL_AMT", 1:6), paste0("PAY_AMT", 1:6))
df[amounts] <- df[amounts] / 1e4

num_vars <- c("AGE", amounts, "PAY_0", paste0("PAY_", 2:6))
form <- as.formula(paste("default ~ factor(SEX) + factor(EDUCATION) + factor(MARRIAGE) +",
                         paste(num_vars, collapse = " + ")))
train <- df[df$set == "train", ]
holdout <- df[df$set == "holdout", ]
holdout <- holdout[order(holdout$ID), ]

auc <- function(y, p) {
  r <- rank(p)
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
ks <- function(y, p) {
  grid <- sort(unique(p))
  max(abs(ecdf(p[y == 1])(grid) - ecdf(p[y == 0])(grid)))
}

coefs <- data.frame()
metrics <- data.frame()
preds <- data.frame(ID = holdout$ID, default = holdout$default)
for (link in c("logit", "probit")) {
  fit <- glm(form, data = train, family = binomial(link = link),
             control = glm.control(epsilon = 1e-10, maxit = 100))
  stopifnot(fit$converged)
  coefs <- rbind(coefs, data.frame(link = link, term = names(coef(fit)), estimate = unname(coef(fit))))
  p <- predict(fit, holdout, type = "response")
  preds[[link]] <- unname(p)
  metrics <- rbind(metrics, data.frame(
    link = link, auc = auc(holdout$default, p), ks = ks(holdout$default, p),
    brier = mean((p - holdout$default)^2), aic = AIC(fit), deviance = deviance(fit)
  ))
}
write.csv(coefs, "results/r_glm_coefs.csv", row.names = FALSE)
write.csv(metrics, "results/r_metrics.csv", row.names = FALSE)
write.csv(preds, "results/r_glm_preds.csv", row.names = FALSE)
print(metrics)
