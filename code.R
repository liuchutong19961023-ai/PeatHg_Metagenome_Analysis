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
# Mixed-effects model analysis
# hgcA and merB relative abundance
# Study site included as a random effect
# ================================

library(tidyverse)
library(lme4)
library(lmerTest)

# ================================
# Read data
# ================================

read_gene_file <- function(file, gene_name) {
  
  dat <- read.csv(
    file,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  colnames(dat) <- trimws(colnames(dat))
  
  dat %>%
    dplyr::rename(
      site = site,
      abundance = abundance,
      climate = climate
    ) %>%
    dplyr::mutate(
      gene = gene_name,
      site = factor(site),
      
      climate = factor(
        climate,
        levels = c(
          "Tropic",
          "Subtropic",
          "Temperate",
          "arc"
        )
      ),
      
      climate_order = as.numeric(climate),
      abundance = as.numeric(abundance)
    ) %>%
    dplyr::filter(
      !is.na(site),
      !is.na(climate),
      !is.na(abundance)
    )
}

hgcA_df <- read_gene_file(
  "hgcA.csv",
  "hgcA"
)

merB_df <- read_gene_file(
  "merB.csv",
  "merB"
)

df <- dplyr::bind_rows(
  hgcA_df,
  merB_df
)

# ================================
# Mixed-effects model
# abundance ~ climate_order + (1|site)
# ================================

model_results <- list()
anova_results <- list()

for (g in unique(df$gene)) {
  
  sub_df <- df %>%
    dplyr::filter(gene == g)
  
  model <- lmer(
    abundance ~ climate_order + (1 | site),
    data = sub_df
  )
  
  anova_tab <- as.data.frame(
    anova(model)
  )
  
  anova_tab$Effect <- rownames(anova_tab)
  anova_tab$Gene <- g
  
  anova_results[[g]] <- anova_tab
  
  p_value <- anova_tab$`Pr(>F)`[
    anova_tab$Effect == "climate_order"
  ]
  
  model_results[[g]] <- data.frame(
    Gene = g,
    Model = "abundance ~ climate_order + (1 | site)",
    P_value = p_value,
    stringsAsFactors = FALSE
  )
  
  sink(
    paste0(
      "Mixed_effects_model_summary_",
      g,
      ".txt"
    )
  )
  
  cat("Mixed-effects model\n")
  cat("Formula: abundance ~ climate_order + (1 | site)\n")
  cat("Study site was included as a random effect.\n\n")
  
  print(summary(model))
  
  cat("\nANOVA:\n")
  print(anova(model))
  
  cat("\nRandom effects:\n")
  print(VarCorr(model))
  
  cat("\nSingular fit:\n")
  print(isSingular(model))
  
  sink()
}

# ================================
# Export results
# ================================

p_table <- dplyr::bind_rows(
  model_results
)

anova_table <- dplyr::bind_rows(
  anova_results
)

write.csv(
  p_table,
  "Fig2a_mixed_effects_model_p_values.csv",
  row.names = FALSE
)

write.csv(
  anova_table,
  "Fig2a_mixed_effects_model_ANOVA.csv",
  row.names = FALSE
)

print(p_table)



# ================================
# hgcA and merB gene-sequence taxonomic composition
# Climate-associated variation in dominant taxa
# Tweedie GLMM with study site included as a random effect
#
# Input:
#   hgcA.csv: first column = taxon annotation, remaining columns = sample IDs
#   merB.csv: first column = taxon annotation, remaining columns = sample IDs
#   group.csv: site, sample, climate
#
# Data processing:
#   Empty cells are treated as 0
#   Values are converted to within-sample percentages
#
# Model:
#   relative_abundance_tweedie ~ climate4 + (1 | site)
#
# Display:
#   Four climate zones: tro, sub, tem, arc
#   Bubble size = mean within-sample relative abundance (%)
#   Red taxon label = at least one significant pairwise contrast
#   Letters above bubbles = compact letter display from emmeans
# ================================

library(tidyverse)
library(glmmTMB)
library(emmeans)
library(ggtext)
library(multcomp)
library(patchwork)

# ================================
# Global parameters
# ================================

epsilon <- 1e-6
pairwise_adjust_method <- "none"

climate_levels <- c("tro", "sub", "tem", "arc")

climate_labels <- c(
  tro = "Tropic",
  sub = "Subtropic",
  tem = "Temperate-boreal",
  arc = "Arctic"
)

size_breaks <- c(0, 25, 50, 75, 100)
size_labels <- c("0", "25", "50", "75", "100")

p_to_sig <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE ~ ""
  )
}

# ================================
# Function for one gene
# ================================

run_taxon_bubble_analysis <- function(
    taxon_file,
    group_file,
    gene_name,
    fill_color,
    edge_color
) {
  
  # ----------------
  # Read data
  # ----------------
  
  taxon_raw <- read.csv(
    taxon_file,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  group <- read.csv(
    group_file,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  colnames(taxon_raw) <- trimws(colnames(taxon_raw))
  colnames(group) <- trimws(colnames(group))
  
  colnames(taxon_raw)[1] <- "Taxon"
  colnames(group) <- c("site", "sample", "climate")
  
  group <- group %>%
    dplyr::mutate(
      site = factor(site),
      sample = as.character(sample),
      climate4 = factor(climate, levels = climate_levels)
    )
  
  sample_cols <- setdiff(colnames(taxon_raw), "Taxon")
  
  # ----------------
  # Empty cells are treated as 0
  # ----------------
  
  taxon_raw <- taxon_raw %>%
    dplyr::mutate(
      Taxon = as.character(Taxon),
      dplyr::across(
        dplyr::all_of(sample_cols),
        ~ as.numeric(dplyr::na_if(as.character(.x), ""))
      )
    ) %>%
    dplyr::mutate(
      dplyr::across(
        dplyr::all_of(sample_cols),
        ~ tidyr::replace_na(.x, 0)
      )
    )
  
  write.csv(
    taxon_raw,
    paste0("01_", gene_name, "_taxon_abundance_empty_as_zero.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Sum duplicated taxon annotations if present
  # ----------------
  
  taxon_sum <- taxon_raw %>%
    dplyr::group_by(Taxon) %>%
    dplyr::summarise(
      dplyr::across(
        dplyr::all_of(sample_cols),
        ~ sum(.x, na.rm = TRUE)
      ),
      .groups = "drop"
    )
  
  write.csv(
    taxon_sum,
    paste0("02_", gene_name, "_taxon_abundance_sample_matrix.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Within-sample percentage normalization
  # Each sample column sums to 100 if total abundance > 0
  # ----------------
  
  sample_totals <- colSums(
    taxon_sum[, sample_cols, drop = FALSE],
    na.rm = TRUE
  )
  
  taxon_percent <- taxon_sum
  
  for (s in sample_cols) {
    if (!is.na(sample_totals[s]) && sample_totals[s] > 0) {
      taxon_percent[[s]] <- taxon_sum[[s]] / sample_totals[s] * 100
    } else {
      taxon_percent[[s]] <- 0
    }
  }
  
  write.csv(
    taxon_percent,
    paste0("03_", gene_name, "_taxon_within_sample_relative_abundance_percent.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Long table
  # ----------------
  
  sample_long <- taxon_percent %>%
    tidyr::pivot_longer(
      cols = -Taxon,
      names_to = "sample",
      values_to = "relative_abundance"
    ) %>%
    dplyr::mutate(
      sample = as.character(sample),
      relative_abundance = as.numeric(relative_abundance),
      relative_abundance = pmin(pmax(relative_abundance, 0), 100),
      relative_abundance_tweedie = relative_abundance + epsilon
    ) %>%
    dplyr::left_join(group, by = "sample") %>%
    dplyr::filter(
      !is.na(site),
      !is.na(climate4),
      !is.na(relative_abundance),
      !is.na(relative_abundance_tweedie)
    )
  
  write.csv(
    sample_long,
    paste0("04_", gene_name, "_taxon_sample_level_long_table.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Select Top10 taxa based on overall mean relative abundance
  # ----------------
  
  top10 <- sample_long %>%
    dplyr::group_by(Taxon) %>%
    dplyr::summarise(
      overall_mean = mean(relative_abundance, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(overall_mean)) %>%
    dplyr::slice_head(n = 10) %>%
    dplyr::pull(Taxon)
  
  plot_df <- sample_long %>%
    dplyr::filter(Taxon %in% top10)
  
  write.csv(
    plot_df,
    paste0("05_", gene_name, "_taxon_top10_model_input.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Run Tweedie GLMM + pairwise letters
  # ----------------
  
  summary_list <- list()
  pairwise_list <- list()
  letter_list <- list()
  
  for (taxon_i in top10) {
    
    tmp <- plot_df %>%
      dplyr::filter(Taxon == taxon_i)
    
    model4 <- tryCatch(
      glmmTMB(
        relative_abundance_tweedie ~ climate4 + (1 | site),
        data = tmp,
        family = tweedie(link = "log")
      ),
      error = function(e) NULL
    )
    
    climate4_status <- "failed"
    min_pairwise4_p <- NA_real_
    n_sig_pairwise4 <- 0L
    sig_pairwise4_contrasts <- ""
    pairwise4_sig <- ""
    pairwise4_is_sig <- FALSE
    
    if (!is.null(model4)) {
      
      climate4_status <- "success"
      
      emm4 <- tryCatch(
        emmeans(model4, ~ climate4, type = "response"),
        error = function(e) NULL
      )
      
      pair4 <- NULL
      letters4 <- NULL
      
      if (!is.null(emm4)) {
        
        pair4 <- tryCatch(
          as.data.frame(
            pairs(emm4, adjust = pairwise_adjust_method)
          ),
          error = function(e) NULL
        )
        
        if (!is.null(pair4)) {
          
          pair4 <- pair4 %>%
            dplyr::mutate(
              Taxon = taxon_i,
              Gene = gene_name,
              model = "climate4_Tweedie_GLMM",
              significance = p_to_sig(p.value)
            )
          
          min_pairwise4_p <- min(pair4$p.value, na.rm = TRUE)
          
          sig4 <- pair4 %>%
            dplyr::filter(!is.na(p.value), p.value < 0.05)
          
          n_sig_pairwise4 <- nrow(sig4)
          
          if (n_sig_pairwise4 > 0) {
            sig_pairwise4_contrasts <- paste(sig4$contrast, collapse = "; ")
          }
          
          pairwise4_sig <- p_to_sig(min_pairwise4_p)
          pairwise4_is_sig <- !is.na(min_pairwise4_p) & min_pairwise4_p < 0.05
          
          pairwise_list[[taxon_i]] <- pair4
        }
        
        letters4 <- tryCatch(
          as.data.frame(
            multcomp::cld(
              emm4,
              adjust = pairwise_adjust_method,
              Letters = letters,
              alpha = 0.05
            )
          ),
          error = function(e) NULL
        )
        
        if (!is.null(letters4)) {
          letters4 <- letters4 %>%
            dplyr::mutate(
              Taxon = taxon_i,
              Gene = gene_name,
              climate4 = factor(climate4, levels = climate_levels),
              letter = gsub(" ", "", .group)
            ) %>%
            dplyr::select(
              Gene,
              Taxon,
              climate4,
              letter
            )
          
          letter_list[[taxon_i]] <- letters4
        }
      }
      
      sink(
        paste0(
          "06_",
          gene_name,
          "_model_climate4_pairwise_letters_",
          gsub("[/()+ ]", "_", taxon_i),
          ".txt"
        )
      )
      
      cat("Tweedie GLMM climate4 model for", gene_name, "-", taxon_i, "\n")
      cat("Formula: relative_abundance_tweedie ~ climate4 + (1 | site)\n")
      cat("Family: tweedie(link = 'log')\n")
      cat("Study site was included as a random effect.\n\n")
      print(summary(model4))
      cat("\nPairwise climate4 comparisons:\n")
      print(pair4)
      cat("\nCompact letter display:\n")
      print(letters4)
      cat("\nRandom effects:\n")
      print(VarCorr(model4))
      sink()
    }
    
    summary_list[[taxon_i]] <- data.frame(
      Gene = gene_name,
      Taxon = taxon_i,
      min_pairwise4_p = min_pairwise4_p,
      pairwise4_sig = pairwise4_sig,
      pairwise4_is_sig = pairwise4_is_sig,
      n_sig_pairwise4 = n_sig_pairwise4,
      sig_pairwise4_contrasts = sig_pairwise4_contrasts,
      climate4_status = climate4_status,
      stringsAsFactors = FALSE
    )
  }
  
  summary_table <- dplyr::bind_rows(summary_list)
  pairwise_table <- dplyr::bind_rows(pairwise_list)
  letter_table <- dplyr::bind_rows(letter_list)
  
  write.csv(
    summary_table,
    paste0("07_", gene_name, "_Tweedie_climate4_pairwise_summary.csv"),
    row.names = FALSE
  )
  
  write.csv(
    pairwise_table,
    paste0("08_", gene_name, "_Tweedie_climate4_pairwise_results.csv"),
    row.names = FALSE
  )
  
  write.csv(
    letter_table,
    paste0("09_", gene_name, "_Tweedie_climate4_letters.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Bubble plot source
  # ----------------
  
  bubble_df <- plot_df %>%
    dplyr::group_by(Taxon, climate4) %>%
    dplyr::summarise(
      mean_relative_abundance = mean(relative_abundance, na.rm = TRUE),
      sd_relative_abundance = sd(relative_abundance, na.rm = TRUE),
      n_sample = dplyr::n(),
      n_site = dplyr::n_distinct(site),
      .groups = "drop"
    ) %>%
    dplyr::left_join(summary_table, by = "Taxon") %>%
    dplyr::left_join(letter_table, by = c("Gene", "Taxon", "climate4"))
  
  write.csv(
    bubble_df,
    paste0("10_", gene_name, "_bubbleplot_pairwise_letters_source.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Order taxa
  # Top-to-bottom = high-to-low Top10 relative abundance
  # ggplot puts first factor level at bottom, so use rev(top10)
  # ----------------
  
  taxon_order <- rev(top10)
  
  bubble_df$Taxon <- factor(
    bubble_df$Taxon,
    levels = taxon_order
  )
  
  # ----------------
  # Y-axis labels
  # ----------------
  
  label_table <- summary_table %>%
    dplyr::mutate(
      Taxon = factor(Taxon, levels = taxon_order),
      label = dplyr::case_when(
        pairwise4_is_sig ~ paste0(
          "<span style='color:#B22222'><i>",
          as.character(Taxon),
          "</i> ",
          pairwise4_sig,
          "</span>"
        ),
        TRUE ~ paste0("<i>", as.character(Taxon), "</i>")
      )
    )
  
  label_vec <- label_table$label
  names(label_vec) <- as.character(label_table$Taxon)
  
  # ----------------
  # Plot
  # ----------------
  
  p <- ggplot(
    bubble_df,
    aes(
      x = climate4,
      y = Taxon
    )
  ) +
    geom_point(
      aes(size = mean_relative_abundance),
      shape = 21,
      fill = fill_color,
      color = edge_color,
      stroke = 0.35,
      alpha = 0.88
    ) +
    geom_text(
      aes(
        label = ifelse(pairwise4_is_sig, letter, "")
      ),
      vjust = -1.05,
      size = 3.2,
      color = "black",
      na.rm = TRUE
    ) +
    scale_x_discrete(labels = climate_labels) +
    scale_y_discrete(labels = label_vec) +
    scale_size_continuous(
      name = "Mean relative abundance (%)",
      limits = c(0, 100),
      breaks = size_breaks,
      labels = size_labels,
      range = c(1.8, 12)
    ) +
    coord_cartesian(clip = "off") +
    labs(
      x = NULL,
      y = NULL,
      title = gene_name
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(
        hjust = 0,
        face = "italic",
        size = 14
      ),
      axis.text.x = element_text(
        angle = 25,
        hjust = 1,
        color = "black",
        size = 12
      ),
      axis.text.y = ggtext::element_markdown(
        size = 11,
        color = "black"
      ),
      axis.line = element_line(linewidth = 0.5),
      axis.ticks = element_line(linewidth = 0.5),
      legend.position = "right",
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      plot.margin = margin(12, 12, 12, 12)
    )
  
  ggsave(
    paste0("11_", gene_name, "_taxon_bubbleplot_Tweedie_climate4_pairwise_letters.pdf"),
    p,
    width = 6.7,
    height = 5.0
  )
  
  ggsave(
    paste0("11_", gene_name, "_taxon_bubbleplot_Tweedie_climate4_pairwise_letters.png"),
    p,
    width = 6.7,
    height = 5.0,
    dpi = 600
  )
  
  return(
    list(
      plot = p,
      summary = summary_table,
      pairwise = pairwise_table,
      letters = letter_table,
      bubble_source = bubble_df,
      percent_matrix = taxon_percent,
      long_table = sample_long
    )
  )
}

# ================================
# Run hgcA and merB
# ================================

res_hgcA <- run_taxon_bubble_analysis(
  taxon_file = "hgcA.csv",
  group_file = "group.csv",
  gene_name = "hgcA",
  fill_color = "#8FB7C9",
  edge_color = "#3A5661"
)

res_merB <- run_taxon_bubble_analysis(
  taxon_file = "merB.csv",
  group_file = "group.csv",
  gene_name = "merB",
  fill_color = "#D9A36A",
  edge_color = "#6E4A2E"
)

# ================================
# Export combined tables
# ================================

all_summary <- dplyr::bind_rows(
  res_hgcA$summary,
  res_merB$summary
)

all_pairwise <- dplyr::bind_rows(
  res_hgcA$pairwise,
  res_merB$pairwise
)

all_letters <- dplyr::bind_rows(
  res_hgcA$letters,
  res_merB$letters
)

all_bubble_source <- dplyr::bind_rows(
  res_hgcA$bubble_source,
  res_merB$bubble_source
)

all_long_table <- dplyr::bind_rows(
  res_hgcA$long_table,
  res_merB$long_table
)

write.csv(
  all_summary,
  "12_hgcA_merB_gene_sequence_Tweedie_pairwise_summary_combined.csv",
  row.names = FALSE
)

write.csv(
  all_pairwise,
  "13_hgcA_merB_gene_sequence_Tweedie_pairwise_results_combined.csv",
  row.names = FALSE
)

write.csv(
  all_letters,
  "14_hgcA_merB_gene_sequence_Tweedie_letters_combined.csv",
  row.names = FALSE
)

write.csv(
  all_bubble_source,
  "15_hgcA_merB_gene_sequence_bubbleplot_source_combined.csv",
  row.names = FALSE
)

write.csv(
  all_long_table,
  "16_hgcA_merB_gene_sequence_relative_abundance_long_table_combined.csv",
  row.names = FALSE
)

# ================================
# Combined figure
# ================================

combined_plot <- res_hgcA$plot / res_merB$plot +
  patchwork::plot_layout(guides = "collect") &
  theme(
    legend.position = "right"
  )

ggsave(
  "17_hgcA_merB_gene_sequence_taxon_bubbleplot_Tweedie_pairwise_letters_combined.pdf",
  combined_plot,
  width = 7.2,
  height = 9.6
)

ggsave(
  "17_hgcA_merB_gene_sequence_taxon_bubbleplot_Tweedie_pairwise_letters_combined.png",
  combined_plot,
  width = 7.2,
  height = 9.6,
  dpi = 600
)

cat("Finished. hgcA and merB gene-sequence taxonomic bubble plots were generated.\n")





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
# hgcA and merB MAG phylum composition
# Mixed-effects Tweedie GLMM with study site included as a random effect
# Bubble plots of dominant phyla across climate zones
# ================================

install.packages("tidyverse")
install.packages("glmmTMB")
install.packages("emmeans")
install.packages("ggtext")
install.packages("multcomp")
install.packages("patchwork")

library(tidyverse)
library(glmmTMB)
library(emmeans)
library(ggtext)
library(multcomp)
library(patchwork)

# ----------------
# 1. Global parameters
# ----------------

epsilon <- 1e-6
pairwise_adjust_method <- "none"

climate_levels <- c("tro", "sub", "tem", "arc")

climate_labels <- c(
  tro = "Tropic",
  sub = "Subtropic",
  tem = "Temperate-boreal",
  arc = "Arctic"
)

size_breaks <- c(0, 0.25, 0.50, 0.75, 1.00)
size_labels <- c("0.00", "0.25", "0.50", "0.75", "1.00")

p_to_sig <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE ~ ""
  )
}

# ----------------
# 2. Function for one gene
# ----------------

run_bubble_analysis <- function(
    mag_file,
    group_file,
    gene_name,
    fill_color,
    edge_color
) {
  
  # ----------------
  # Read data
  # ----------------
  
  abund <- read.csv(mag_file, check.names = FALSE)
  group <- read.csv(group_file, check.names = FALSE)
  
  colnames(abund)[1] <- "Phylum"
  colnames(group) <- c("sample", "site", "group")
  
  group <- group %>%
    dplyr::mutate(
      sample = as.character(sample),
      site = factor(site),
      climate4 = factor(group, levels = climate_levels)
    )
  
  sample_cols <- setdiff(colnames(abund), "Phylum")
  
  abund <- abund %>%
    dplyr::mutate(
      dplyr::across(
        dplyr::all_of(sample_cols),
        ~ as.numeric(.x)
      )
    )
  
  # ----------------
  # Sum MAGs to phylum level
  # ----------------
  
  phylum_abund <- abund %>%
    dplyr::group_by(Phylum) %>%
    dplyr::summarise(
      dplyr::across(
        dplyr::all_of(sample_cols),
        ~ sum(.x, na.rm = TRUE)
      ),
      .groups = "drop"
    )
  
  write.csv(
    phylum_abund,
    paste0("01_", gene_name, "_phylum_abundance_sample_matrix.csv"),
    row.names = FALSE
  )
  
  sample_long <- phylum_abund %>%
    tidyr::pivot_longer(
      cols = -Phylum,
      names_to = "sample",
      values_to = "abundance"
    ) %>%
    dplyr::mutate(
      sample = as.character(sample),
      abundance = pmin(pmax(abundance, 0), 1),
      abundance_tweedie = abundance + epsilon
    ) %>%
    dplyr::left_join(group, by = "sample") %>%
    dplyr::filter(
      !is.na(site),
      !is.na(climate4),
      !is.na(abundance),
      !is.na(abundance_tweedie)
    )
  
  write.csv(
    sample_long,
    paste0("02_", gene_name, "_phylum_sample_level_long_table.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Select Top10 phyla
  # ----------------
  
  top10 <- sample_long %>%
    dplyr::group_by(Phylum) %>%
    dplyr::summarise(
      overall_mean = mean(abundance, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(overall_mean)) %>%
    dplyr::slice_head(n = 10) %>%
    dplyr::pull(Phylum)
  
  plot_df <- sample_long %>%
    dplyr::filter(Phylum %in% top10)
  
  write.csv(
    plot_df,
    paste0("03_", gene_name, "_phylum_top10_model_input.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Run Tweedie GLMM + pairwise letters
  # ----------------
  
  summary_list <- list()
  pairwise_list <- list()
  letter_list <- list()
  
  for (p in top10) {
    
    tmp <- plot_df %>%
      dplyr::filter(Phylum == p)
    
    model4 <- tryCatch(
      glmmTMB(
        abundance_tweedie ~ climate4 + (1 | site),
        data = tmp,
        family = tweedie(link = "log")
      ),
      error = function(e) NULL
    )
    
    climate4_status <- "failed"
    min_pairwise4_p <- NA_real_
    n_sig_pairwise4 <- 0L
    sig_pairwise4_contrasts <- ""
    pairwise4_sig <- ""
    pairwise4_is_sig <- FALSE
    
    if (!is.null(model4)) {
      
      climate4_status <- "success"
      
      emm4 <- tryCatch(
        emmeans(model4, ~ climate4, type = "response"),
        error = function(e) NULL
      )
      
      pair4 <- NULL
      letters4 <- NULL
      
      if (!is.null(emm4)) {
        
        pair4 <- tryCatch(
          as.data.frame(
            pairs(emm4, adjust = pairwise_adjust_method)
          ),
          error = function(e) NULL
        )
        
        if (!is.null(pair4)) {
          
          pair4 <- pair4 %>%
            dplyr::mutate(
              Phylum = p,
              Gene = gene_name,
              model = "climate4_Tweedie_GLMM",
              significance = p_to_sig(p.value)
            )
          
          min_pairwise4_p <- min(pair4$p.value, na.rm = TRUE)
          
          sig4 <- pair4 %>%
            dplyr::filter(!is.na(p.value), p.value < 0.05)
          
          n_sig_pairwise4 <- nrow(sig4)
          
          if (n_sig_pairwise4 > 0) {
            sig_pairwise4_contrasts <- paste(sig4$contrast, collapse = "; ")
          }
          
          pairwise4_sig <- p_to_sig(min_pairwise4_p)
          pairwise4_is_sig <- !is.na(min_pairwise4_p) & min_pairwise4_p < 0.05
          
          pairwise_list[[p]] <- pair4
        }
        
        letters4 <- tryCatch(
          as.data.frame(
            multcomp::cld(
              emm4,
              adjust = pairwise_adjust_method,
              Letters = letters,
              alpha = 0.05
            )
          ),
          error = function(e) NULL
        )
        
        if (!is.null(letters4)) {
          letters4 <- letters4 %>%
            dplyr::mutate(
              Phylum = p,
              Gene = gene_name,
              climate4 = factor(climate4, levels = climate_levels),
              letter = gsub(" ", "", .group)
            ) %>%
            dplyr::select(
              Gene,
              Phylum,
              climate4,
              letter
            )
          
          letter_list[[p]] <- letters4
        }
      }
      
      sink(
        paste0(
          "04_",
          gene_name,
          "_model_climate4_pairwise_letters_",
          gsub("[/()+ ]", "_", p),
          ".txt"
        )
      )
      cat("Tweedie GLMM climate4 model for", gene_name, "-", p, "\n")
      cat("Formula: abundance_tweedie ~ climate4 + (1 | site)\n")
      cat("Family: tweedie(link = 'log')\n")
      cat("Study site was included as a random effect.\n\n")
      print(summary(model4))
      cat("\nPairwise climate4 comparisons:\n")
      print(pair4)
      cat("\nCompact letter display:\n")
      print(letters4)
      cat("\nRandom effects:\n")
      print(VarCorr(model4))
      sink()
    }
    
    summary_list[[p]] <- data.frame(
      Gene = gene_name,
      Phylum = p,
      min_pairwise4_p = min_pairwise4_p,
      pairwise4_sig = pairwise4_sig,
      pairwise4_is_sig = pairwise4_is_sig,
      n_sig_pairwise4 = n_sig_pairwise4,
      sig_pairwise4_contrasts = sig_pairwise4_contrasts,
      climate4_status = climate4_status,
      stringsAsFactors = FALSE
    )
  }
  
  summary_table <- dplyr::bind_rows(summary_list)
  pairwise_table <- dplyr::bind_rows(pairwise_list)
  letter_table <- dplyr::bind_rows(letter_list)
  
  write.csv(
    summary_table,
    paste0("05_", gene_name, "_Tweedie_climate4_pairwise_summary.csv"),
    row.names = FALSE
  )
  
  write.csv(
    pairwise_table,
    paste0("06_", gene_name, "_Tweedie_climate4_pairwise_results.csv"),
    row.names = FALSE
  )
  
  write.csv(
    letter_table,
    paste0("07_", gene_name, "_Tweedie_climate4_letters.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Bubble plot source
  # ----------------
  
  bubble_df <- plot_df %>%
    dplyr::group_by(Phylum, climate4) %>%
    dplyr::summarise(
      mean_abundance = mean(abundance, na.rm = TRUE),
      sd_abundance = sd(abundance, na.rm = TRUE),
      n_sample = dplyr::n(),
      n_site = dplyr::n_distinct(site),
      .groups = "drop"
    ) %>%
    dplyr::left_join(summary_table, by = "Phylum") %>%
    dplyr::left_join(letter_table, by = c("Gene", "Phylum", "climate4"))
  
  write.csv(
    bubble_df,
    paste0("08_", gene_name, "_bubbleplot_pairwise_letters_source.csv"),
    row.names = FALSE
  )
  
  # ----------------
  # Order phyla
  # Top-to-bottom = high-to-low Top10 abundance
  # ggplot puts first factor level at bottom, so use rev(top10)
  # ----------------
  
  phylum_order <- rev(top10)
  
  bubble_df$Phylum <- factor(
    bubble_df$Phylum,
    levels = phylum_order
  )
  
  # ----------------
  # Y-axis labels
  # ----------------
  
  label_table <- summary_table %>%
    dplyr::mutate(
      Phylum = factor(Phylum, levels = phylum_order),
      label = dplyr::case_when(
        pairwise4_is_sig ~ paste0(
          "<span style='color:#B22222'><i>",
          as.character(Phylum),
          "</i> ",
          pairwise4_sig,
          "</span>"
        ),
        TRUE ~ paste0("<i>", as.character(Phylum), "</i>")
      )
    )
  
  label_vec <- label_table$label
  names(label_vec) <- as.character(label_table$Phylum)
  
  # ----------------
  # Plot
  # ----------------
  
  p <- ggplot(
    bubble_df,
    aes(
      x = climate4,
      y = Phylum
    )
  ) +
    geom_point(
      aes(size = mean_abundance),
      shape = 21,
      fill = fill_color,
      color = edge_color,
      stroke = 0.35,
      alpha = 0.88
    ) +
    geom_text(
      aes(
        label = ifelse(pairwise4_is_sig, letter, "")
      ),
      vjust = -1.05,
      size = 3.2,
      color = "black",
      na.rm = TRUE
    ) +
    scale_x_discrete(labels = climate_labels) +
    scale_y_discrete(labels = label_vec) +
    scale_size_continuous(
      name = "Mean relative abundance",
      limits = c(0, 1),
      breaks = size_breaks,
      labels = size_labels,
      range = c(1.8, 12)
    ) +
    coord_cartesian(clip = "off") +
    labs(
      x = NULL,
      y = NULL,
      title = gene_name
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(
        hjust = 0,
        face = "italic",
        size = 14
      ),
      axis.text.x = element_text(
        angle = 25,
        hjust = 1,
        color = "black",
        size = 12
      ),
      axis.text.y = ggtext::element_markdown(
        size = 11,
        color = "black"
      ),
      axis.line = element_line(linewidth = 0.5),
      axis.ticks = element_line(linewidth = 0.5),
      legend.position = "right",
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      plot.margin = margin(12, 12, 12, 12)
    )
  
  ggsave(
    paste0("09_", gene_name, "_bubbleplot_Tweedie_climate4_pairwise_letters.pdf"),
    p,
    width = 6.7,
    height = 5.0
  )
  
  ggsave(
    paste0("09_", gene_name, "_bubbleplot_Tweedie_climate4_pairwise_letters.png"),
    p,
    width = 6.7,
    height = 5.0,
    dpi = 600
  )
  
  return(
    list(
      plot = p,
      summary = summary_table,
      pairwise = pairwise_table,
      letters = letter_table,
      bubble_source = bubble_df
    )
  )
}

# ----------------
# 3. Run hgcA and merB
# ----------------

res_hgcA <- run_bubble_analysis(
  mag_file = "hgcAMAGs.csv",
  group_file = "hgcAgroup.csv",
  gene_name = "hgcA",
  fill_color = "#8FB7C9",
  edge_color = "#3A5661"
)

res_merB <- run_bubble_analysis(
  mag_file = "merBMAGs.csv",
  group_file = "merBgroup.csv",
  gene_name = "merB",
  fill_color = "#D9A36A",
  edge_color = "#6E4A2E"
)

# ----------------
# 4. Export combined tables
# ----------------

all_summary <- dplyr::bind_rows(
  res_hgcA$summary,
  res_merB$summary
)

all_pairwise <- dplyr::bind_rows(
  res_hgcA$pairwise,
  res_merB$pairwise
)

all_letters <- dplyr::bind_rows(
  res_hgcA$letters,
  res_merB$letters
)

all_bubble_source <- dplyr::bind_rows(
  res_hgcA$bubble_source,
  res_merB$bubble_source
)

write.csv(
  all_summary,
  "10_hgcA_merB_Tweedie_pairwise_summary_combined.csv",
  row.names = FALSE
)

write.csv(
  all_pairwise,
  "11_hgcA_merB_Tweedie_pairwise_results_combined.csv",
  row.names = FALSE
)

write.csv(
  all_letters,
  "12_hgcA_merB_Tweedie_letters_combined.csv",
  row.names = FALSE
)

write.csv(
  all_bubble_source,
  "13_hgcA_merB_bubbleplot_source_combined.csv",
  row.names = FALSE
)

# ----------------
# 5. Combined figure
# ----------------

combined_plot <- res_hgcA$plot / res_merB$plot +
  patchwork::plot_layout(guides = "collect") &
  theme(
    legend.position = "right"
  )

ggsave(
  "14_hgcA_merB_bubbleplot_Tweedie_pairwise_letters_combined.pdf",
  combined_plot,
  width = 7.2,
  height = 9.6
)

ggsave(
  "14_hgcA_merB_bubbleplot_Tweedie_pairwise_letters_combined.png",
  combined_plot,
  width = 7.2,
  height = 9.6,
  dpi = 600
)

cat("Finished. hgcA and merB bubble plots were generated with unified bubble-size legends.\n")





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
