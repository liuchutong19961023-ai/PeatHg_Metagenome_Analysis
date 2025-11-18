# set work directory
setwd("C:/Users/Desktop/data")

# Random forest analysis
install.packages("randomForest")
library(randomForest)
install.packages("rfPermute")
library(rfPermute)

Total_Random<-read.csv("RF.csv",row.names=1)
Total_Random.rf<-rfPermute(hgcA~.,data=Total_Random,importance=TRUE,ntree=1000,nrep=1000 ,num.cores = 1)
print(Total_Random.rf)
plot(rp.importance(Total_Random.rf,scale = TRUE))
imp.score <- rp.importance(Total_Random.rf, scale = TRUE)
imp.score
write.csv(imp.score,'RFresult.csv')

# PLS-PM model
install.packages('devtools')
install.packages("plspm")
library(devtools)
library(plspm)

dat <- read.delim('PLS.txt', sep = '\t')
dat_blocks <- list(
    MAT_Lat = c('MAT', 'latitude'), 
        TN = 'TN',
        nifH.nosZ = 'nifH.nosZ',
        hgcA = 'hgcA',
        merB = 'merB'
)

dat_blocks

MAT_Lat <- c(0, 0, 0, 0, 0)
TN <- c(1, 0, 0, 0, 0)
nifH.nosZ <- c(0, 1, 0, 0, 0)
hgcA  <- c(0, 0, 1, 0, 0)
merB <- c(0, 0, 1, 0, 0)

dat_path <- rbind(MAT_Lat, TN, nifH.nosZ, hgcA, merB)
colnames(dat_path) <- rownames(dat_path)
dat_path

dat_modes <- rep('A', 5)
dat_modes

dat_pls <- plspm(dat, dat_path, dat_blocks, modes = dat_modes)
dat_pls
summary(dat_pls)

dat_pls$path_coefs
dat_pls$inner_model

innerplot(dat_pls, colpos = 'red', colneg = 'blue', show.values = TRUE, lcol = 'gray', box.lwd = 0)


#10-fold cross-validation
install.packages('ggplot2')
library(ggplot2)

dat <- read.csv("10fold.csv", header=TRUE)
colnames(dat) <- c("MAT","TN")

set.seed(123)
K <- 10
n <- nrow(dat)
folds <- sample(rep(1:K, length.out=n))

pred_all <- numeric(n)
obs_all <- dat$TN
train_r2 <- numeric(K)
test_r2 <- numeric(K)

for (k in 1:K){
train_idx <- which(folds != k)
test_idx <- which(folds == k)
  
train <- dat[train_idx, ]
test <- dat[test_idx, ]
  
mod <- lm(TN ~ MAT, data=train)

pred_all[test_idx] <- predict(mod, newdata=test)

train_r2[k] <- summary(mod)$r.squared

ss_res <- sum((test$TN - predict(mod, newdata=test))^2)
  ss_tot <- sum((test$TN - mean(test$TN))^2)
  test_r2[k] <- 1 - ss_res/ss_tot
}

mean_train_r2 <- mean(train_r2)
mean_test_r2 <- mean(test_r2)
mean_bias <- mean(pred_all - obs_all)

cat("Ave treat R² =", round(mean_train_r2,3), "\n")
cat("Ave test R² =", round(mean_test_r2,3), "\n")
cat("Mean Bias =", round(mean_bias,3), "\n")

df_plot <- data.frame(Observed=obs_all, Predicted=pred_all)

df_plot$upper <- df_plot$Observed * 1.5
df_plot$lower <- df_plot$Observed * 0.5

min_val <- min(df_plot$Observed, df_plot$Predicted)
max_val <- max(df_plot$Observed, df_plot$Predicted)

ggplot(df_plot, aes(x=Observed, y=Predicted)) +
  geom_ribbon(aes(ymin=lower, ymax=upper), fill="grey80", alpha=0.3) +  
  geom_point(color="red", size=2) +
  geom_abline(slope=1, intercept=0, linetype=2, color="black") +        
  geom_smooth(method="lm", color="blue", se=FALSE) +                    
  coord_equal(xlim=c(min_val, max_val), ylim=c(min_val, max_val)) +     
  labs(title="10-fold CV: Observed vs Predicted TN",
       subtitle=paste0("Mean Train R²=", round(mean_train_r2,3),
                       ", Mean Test R²=", round(mean_test_r2,3),
                       ", Mean Bias=", round(mean_bias,3)),
       x="Observed TN", y="Predicted TN") +
  theme_minimal()

#Leave-One-Out Cross Validation
install.packages('ggplot2')
library(ggplot2)

dat <- read.csv("LOOCV.csv", header=TRUE)
colnames(dat) <- c("hgcA","TN")

n <- nrow(dat)

pred_all <- numeric(n)
obs_all <- dat$hgcA
train_r2 <- numeric(n)
test_r2 <- numeric(n)

for (i in 1:n){
train <- dat[-i, ]
test <- dat[i, , drop=FALSE]
  
mod <- lm(hgcA ~ TN, data=train)

pred_all[i] <- predict(mod, newdata=test)

train_r2[i] <- summary(mod)$r.squared

ss_res <- (test$TN - pred_all[i])^2
  ss_tot <- (test$TN - mean(test$TN))^2
  test_r2[i] <- 1 - ss_res/ss_tot
}

mean_train_r2 <- mean(train_r2)
mean_test_r2 <- mean(test_r2, na.rm=TRUE)   
mean_bias <- mean(pred_all - obs_all)

cat("Ave treat R² =", round(mean_train_r2,3), "\n")
cat("Ave test R² =", round(mean_test_r2,3), "\n")
cat("Mean Bias =", round(mean_bias,3), "\n")

df_plot <- data.frame(Observed=obs_all, Predicted=pred_all)
df_plot$upper <- df_plot$Observed * 1.5
df_plot$lower <- df_plot$Observed * 0.5

ggplot(df_plot, aes(x=Observed, y=Predicted)) +
  geom_ribbon(aes(ymin=lower, ymax=upper), fill="grey80", alpha=0.3) +  
  geom_point(color="red", size=2) +
  geom_abline(slope=1, intercept=0, linetype=2, color="black") +        
  geom_smooth(method="lm", color="blue", se=FALSE) +                    
  coord_equal() +
  labs(title="LOOCV: Observed vs Predicted hgcA",
       subtitle=paste0("Mean Train R²=", round(mean_train_r2,3),
                       ", Mean Test R²=", round(mean_test_r2,3),
                       ", Mean Bias=", round(mean_bias,3)),
       x="Observed TN", y="Predicted TN") +
  theme_minimal()
