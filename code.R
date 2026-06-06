# ================================
# Set working directory
# ================================
setwd("C:/Users/Desktop/data")


# ================================
# Random Forest analysis
# Identify key predictors of hgcA abundance
# ================================
install.packages("randomForest")
library(randomForest)
install.packages("rfPermute")
library(rfPermute)

Total_Random <- read.csv("RF.csv", row.names = 1)
Total_Random.rf <- rfPermute(
  hgcA ~ .,
  data = Total_Random,
  importance = TRUE,
  ntree = 1000,
  nrep = 1000,
  num.cores = 1
)
print(Total_Random.rf)
plot(rp.importance(Total_Random.rf, scale = TRUE))
imp.score <- rp.importance(Total_Random.rf, scale = TRUE)
write.csv(imp.score, "RFresult.csv")


# ================================
# Assessment of climate-associated differences
# Mixed-effects models with study site included as a random effect
# Boxplots of TN and nitrogen-cycling ratios
# ================================

install.packages("tidyverse")
install.packages("lme4")
install.packages("lmerTest")
install.packages("scales")

library(tidyverse)
library(lme4)
library(lmerTest)
library(scales)

read_one <- function(file, variable_name) {
  x <- read.csv(file, check.names = FALSE)
  colnames(x) <- trimws(colnames(x))
  
  x %>%
    dplyr::rename(
      Site = Site,
      value = data,
      climate = climate
    ) %>%
    dplyr::mutate(
      Site = factor(Site),
      climate = factor(climate, levels = c("tro", "sub", "tem", "arc")),
      climate_order = as.numeric(climate),
      value = as.numeric(value),
      Variable = variable_name
    ) %>%
    dplyr::filter(
      !is.na(Site),
      !is.na(climate),
      !is.na(value)
    )
}

TN_df   <- read_one("N.csv", "TN")
nifH_df <- read_one("Nifh.csv", "nifH/nosZ")
nirS_df <- read_one("nirS.csv", "nifH/(nirS+nirK)")

df <- dplyr::bind_rows(TN_df, nifH_df, nirS_df)

df$Variable <- factor(
  df$Variable,
  levels = c("TN", "nifH/nosZ", "nifH/(nirS+nirK)")
)

write.csv(
  df,
  "Climate_N_metrics_combined_cleaned.csv",
  row.names = FALSE
)

# ================================
# Mixed-effects model analysis
# Model: value ~ climate_order + (1 | Site)
# climate_order: tro = 1, sub = 2, tem = 3, arc = 4
# ================================

lmm_results <- list()
p_values <- c()

for (v in levels(df$Variable)) {
  sub_df <- df %>%
    dplyr::filter(Variable == v)
  
  model <- lmer(
    value ~ climate_order + (1 | Site),
    data = sub_df
  )
  
  anova_tab <- as.data.frame(anova(model))
  anova_tab$Effect <- rownames(anova_tab)
  anova_tab$Variable <- v
  
  lmm_results[[v]] <- anova_tab
  p_values[v] <- anova_tab$`Pr(>F)`[
    anova_tab$Effect == "climate_order"
  ]
  
  sink(
    paste0(
      "Mixed_effects_model_summary_",
      gsub("[/()+]", "_", v),
      ".txt"
    )
  )
  cat("Mixed-effects model\n")
  cat("Formula: value ~ climate_order + (1 | Site)\n")
  cat("Study site was included as a random effect.\n")
  cat("Variable:", v, "\n")
  cat("climate_order: tro = 1, sub = 2, tem = 3, arc = 4\n\n")
  print(summary(model))
  cat("\nANOVA:\n")
  print(anova(model))
  cat("\nRandom effects:\n")
  print(VarCorr(model))
  cat("\nSingular fit:\n")
  print(isSingular(model))
  sink()
}

lmm_table <- dplyr::bind_rows(lmm_results)

write.csv(
  lmm_table,
  "Mixed_effects_model_ANOVA_climate_associated_differences.csv",
  row.names = FALSE
)

format_p <- function(p) {
  ifelse(
    is.na(p),
    "NA",
    ifelse(
      p < 0.001,
      "<0.001",
      sprintf("%.3f", p)
    )
  )
}

p_label <- paste0(
  "Mixed-effects model\n",
  "(Site as random effect)\n",
  "TN = ", format_p(p_values["TN"]), "\n",
  "nifH/nosZ = ", format_p(p_values["nifH/nosZ"]), "\n",
  "nifH/(nirS+nirK) = ", format_p(p_values["nifH/(nirS+nirK)"])
)

# ================================
# Prepare data for dual-axis plotting
# Left axis: TN
# Right axis: nitrogen-cycling ratios
# ================================

scale_factor <- 8

df_plot <- df %>%
  dplyr::mutate(
    value_plot = dplyr::case_when(
      Variable == "TN" ~ value,
      TRUE ~ value * scale_factor
    )
  )

summary_plot <- df_plot %>%
  dplyr::group_by(climate, Variable) %>%
  dplyr::summarise(
    n_sample = dplyr::n(),
    n_site = dplyr::n_distinct(Site),
    mean_raw = mean(value, na.rm = TRUE),
    sd_raw = sd(value, na.rm = TRUE),
    mean_plot = mean(value_plot, na.rm = TRUE),
    sd_plot = sd(value_plot, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  summary_plot,
  "Climate_N_metrics_summary_mean_SD.csv",
  row.names = FALSE
)

# ================================
# Boxplot visualization
# Error bars indicate mean ± SD
# ================================

var_cols <- c(
  "TN" = "#168061",
  "nifH/nosZ" = "#A12E24",
  "nifH/(nirS+nirK)" = "#2496BF"
)

var_cols_80 <- scales::alpha(var_cols, 0.80)

climate_labels <- c(
  tro = "Tropic",
  sub = "Subtropic",
  tem = "Temperate-boreal",
  arc = "Arctic"
)

lw_axis <- 0.177
lw_box <- 0.177
lw_error <- 0.32

dodge_width <- 0.38
box_width <- 0.34
jitter_width <- 0.014

dodge <- position_dodge(width = dodge_width)

ymax <- max(
  summary_plot$mean_plot + summary_plot$sd_plot,
  df_plot$value_plot,
  na.rm = TRUE
)

p <- ggplot(
  df_plot,
  aes(
    x = climate,
    y = value_plot,
    fill = Variable,
    color = Variable
  )
) +
  geom_boxplot(
    position = dodge,
    width = box_width,
    alpha = 0.80,
    linewidth = lw_box,
    outlier.shape = NA
  ) +
  geom_point(
    position = position_jitterdodge(
      jitter.width = jitter_width,
      jitter.height = 0,
      dodge.width = dodge_width
    ),
    aes(color = Variable),
    size = 1.2,
    alpha = 0.45,
    stroke = 0,
    shape = 16
  ) +
  geom_errorbar(
    data = summary_plot,
    aes(
      x = climate,
      ymin = mean_plot - sd_plot,
      ymax = mean_plot + sd_plot,
      group = Variable,
      color = Variable
    ),
    position = dodge,
    width = 0.13,
    linewidth = lw_error,
    inherit.aes = FALSE
  ) +
  annotate(
    "text",
    x = 2.55,
    y = ymax * 1.08,
    label = p_label,
    size = 3.1,
    hjust = 0.5
  ) +
  scale_fill_manual(values = var_cols_80) +
  scale_color_manual(values = var_cols) +
  scale_x_discrete(labels = climate_labels) +
  scale_y_continuous(
    name = "Total nitrogen (mg/g)",
    sec.axis = sec_axis(
      ~ . / scale_factor,
      name = "Nitrogen-cycling ratios"
    ),
    expand = expansion(mult = c(0.02, 0.16))
  ) +
  labs(
    x = NULL,
    fill = NULL,
    color = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.title.y.left = element_text(color = "black"),
    axis.title.y.right = element_text(color = "black"),
    axis.line = element_line(linewidth = lw_axis, color = "black"),
    axis.ticks = element_line(linewidth = lw_axis, color = "black"),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.key.width = unit(0.58, "cm"),
    legend.text = element_text(size = 11),
    plot.margin = margin(8, 10, 8, 8)
  )

ggsave(
  "Climate_N_metrics_boxplot_mixed_effects_model.pdf",
  p,
  width = 6.8,
  height = 4.4
)

ggsave(
  "Climate_N_metrics_boxplot_mixed_effects_model.png",
  p,
  width = 6.8,
  height = 4.4,
  dpi = 600
)



# ================================
# PLS-PM analysis
# Structural equation modeling of climate, nitrogen cycling, and mercury cycling
# ================================
install.packages("devtools")
install.packages("plspm")
library(devtools)
library(plspm)

dat <- read.delim("PLS.txt", sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)

dat_blocks <- list(
  MAT_Lat   = c("MAT", "latitude"),
  TN        = c("TN"),
  nifH.nosZ = c("nifH.nosZ"),
  hgcA      = c("hgcA"),
  merB      = c("merB")
)

required_vars <- unique(unlist(dat_blocks))
missing_vars <- setdiff(required_vars, colnames(dat))
if (length(missing_vars) > 0) {
  stop("PLS.txt is missing required column(s): ",
       paste(missing_vars, collapse = ", "))
}

MAT_Lat   <- c(0, 0, 0, 0, 0)
TN        <- c(1, 0, 0, 0, 0)
nifH.nosZ <- c(0, 1, 0, 0, 0)
hgcA      <- c(0, 0, 1, 0, 0)
merB      <- c(0, 0, 1, 0, 0)

dat_path <- rbind(MAT_Lat, TN, nifH.nosZ, hgcA, merB)
colnames(dat_path) <- rownames(dat_path)

dat_modes <- rep("A", 5)

dat_pls <- plspm(dat, dat_path, dat_blocks, modes = dat_modes)
summary(dat_pls)
dat_pls$path_coefs
dat_pls$inner_model

innerplot(
  dat_pls,
  colpos = "red",
  colneg = "blue",
  show.values = TRUE,
  lcol = "gray",
  box.lwd = 0
)


# ================================
# 10-fold cross-validation
# Linear regression between MAT and TN
# ================================
install.packages("ggplot2")
library(ggplot2)

dat <- read.csv("10fold.csv", header = TRUE)
colnames(dat) <- c("MAT", "TN")

set.seed(123)
K <- 10
n <- nrow(dat)
folds <- sample(rep(1:K, length.out = n))

pred_all <- numeric(n)
obs_all <- dat$TN
train_r2 <- numeric(K)
test_r2 <- numeric(K)

for (k in 1:K) {
  train_idx <- which(folds != k)
  test_idx <- which(folds == k)

  train <- dat[train_idx, ]
  test <- dat[test_idx, ]

  mod <- lm(TN ~ MAT, data = train)

  pred_all[test_idx] <- predict(mod, newdata = test)

  train_r2[k] <- summary(mod)$r.squared

  ss_res <- sum((test$TN - predict(mod, newdata = test))^2)
  ss_tot <- sum((test$TN - mean(test$TN))^2)
  test_r2[k] <- 1 - ss_res / ss_tot
}

mean_train_r2 <- mean(train_r2)
mean_test_r2 <- mean(test_r2)
mean_bias <- mean(pred_all - obs_all)

df_plot <- data.frame(Observed = obs_all, Predicted = pred_all)
df_plot$upper <- df_plot$Observed * 1.5
df_plot$lower <- df_plot$Observed * 0.5

min_val <- min(df_plot$Observed, df_plot$Predicted)
max_val <- max(df_plot$Observed, df_plot$Predicted)

ggplot(df_plot, aes(x = Observed, y = Predicted)) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = "grey80", alpha = 0.3) +
  geom_point(color = "red", size = 2) +
  geom_abline(slope = 1, intercept = 0,
              linetype = 2, color = "black") +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  coord_equal(xlim = c(min_val, max_val),
              ylim = c(min_val, max_val)) +
  labs(
    title = "10-fold CV: Observed vs Predicted TN",
    subtitle = paste0(
      "Mean Train R²=", round(mean_train_r2, 3),
      ", Mean Test R²=", round(mean_test_r2, 3),
      ", Mean Bias=", round(mean_bias, 3)
    ),
    x = "Observed TN",
    y = "Predicted TN"
  ) +
  theme_minimal()


# ================================
# Leave-One-Out Cross Validation (LOOCV)
# Linear regression between TN and hgcA
# ================================
install.packages("ggplot2")
library(ggplot2)

dat <- read.csv("LOOCV.csv", header = TRUE)
colnames(dat) <- c("hgcA", "TN")

n <- nrow(dat)

pred_all <- numeric(n)
obs_all <- dat$hgcA
train_r2 <- numeric(n)

for (i in 1:n) {
  train <- dat[-i, , drop = FALSE]
  test  <- dat[i,  , drop = FALSE]

  mod <- lm(hgcA ~ TN, data = train)

  pred_all[i] <- predict(mod, newdata = test)

  train_r2[i] <- summary(mod)$r.squared
}

ss_res <- sum((obs_all - pred_all)^2)
ss_tot <- sum((obs_all - mean(obs_all))^2)
loocv_r2 <- 1 - ss_res / ss_tot

mean_train_r2 <- mean(train_r2)
mean_test_r2 <- loocv_r2
mean_bias <- mean(pred_all - obs_all)

df_plot <- data.frame(Observed = obs_all, Predicted = pred_all)
df_plot$upper <- df_plot$Observed * 1.5
df_plot$lower <- df_plot$Observed * 0.5

ggplot(df_plot, aes(x = Observed, y = Predicted)) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = "grey80", alpha = 0.3) +
  geom_point(color = "red", size = 2) +
  geom_abline(slope = 1, intercept = 0,
              linetype = 2, color = "black") +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  coord_equal() +
  labs(
    title = "LOOCV: Observed vs Predicted hgcA",
    subtitle = paste0(
      "Mean Train R²=", round(mean_train_r2, 3),
      ", LOOCV R²=", round(mean_test_r2, 3),
      ", Mean Bias=", round(mean_bias, 3)
    ),
    x = "Observed hgcA",
    y = "Predicted hgcA"
  ) +
  theme_minimal()
