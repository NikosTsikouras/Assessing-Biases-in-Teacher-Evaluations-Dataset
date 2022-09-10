library(randomForest)
library(MASS)
library(tidyr)
library(dplyr)
library(tibble)
library(data.table)
library(ggplot2)
library(matrixStats)
#Read dataset
Dataset <- read.csv("evaluations.csv")


# model selecion using cross-validation
set.seed(12345)
TEST_SIZE <- 0.3
# split the examples into training and test
test_indices <- sample(1:nrow(Dataset), size=as.integer(TEST_SIZE*nrow(Dataset)), replace=FALSE)
x_train <- Dataset[-test_indices, c(2,3,4,5,10,12,19)]
y_train <- Dataset[-test_indices, 1]
x_test <- Dataset[test_indices, c(2,3,4,5,10,12,19)]
y_test <- Dataset[test_indices, 1]

# grid over which we will perform the hyperparameter search:
hparam_grid <- as.data.frame(expand.grid(mtry=seq(3, 5, by=2), maxnodes=seq(10, 50, by=5),nodesize = seq(1,21,by=2), ntree=seq(100,1000,by=50)))

# to store the Out Of Bag (OOB) estimates of the MSE
oob_mses <- rep(0.0, nrow(hparam_grid))

# perform the gridsearch
for(hparam_idx in 1:nrow(hparam_grid)) {
  # train candidate model
  this_mtry <- hparam_grid[hparam_idx, 1]
  this_maxnodes <- hparam_grid[hparam_idx, 2]
  this_ntrees <- hparam_grid[hparam_idx, 4]
  this_nodesize <- hparam_grid[hparam_idx, 3]
  
  rf <- randomForest(x_train, y_train, mtry=this_mtry, maxnodes=this_maxnodes, nodesize = this_nodesize, ntree = this_ntrees)
  
  # calculate OOB MSE
  oob_mses[hparam_idx] <- mse(y_train, predict(rf))
}

# select the best model (that which has the minimum OOB MSE)
best_hparam_set <- hparam_grid[which.min(oob_mses),]

# train a model on the whole training set with the selected hyperparameters
rf_final <- randomForest(x_train, y_train,
                         mtry=best_hparam_set$mtry,
                         maxnodes=best_hparam_set$maxnodes,
                         importance=TRUE)

# the test performance of the final model
yhat_test <- predict(rf_final, newdata=x_test)

# default hyperparmaeter model
rf_default <- randomForest(x_train, y_train)
yhat_test_default <- predict(rf_default, newdata=x_test)

# MSEs
test_mse <- mse(y_test, yhat_test)
test_mse_default <- mse(y_test, yhat_test_default)

cat(sprintf("Test MSE with default hyperparameters: %.3f, Test MSE with OOB-tuned hyperparameters: %.3f\n", test_mse_default, test_mse))



# calculate each type of RF importance
rf_importance <- cbind(importance(rf_final, type=1), importance(rf_final, type=2)) %>%
  as.data.frame() %>%
  rownames_to_column("variable") 

# plot importances
rf_importance %>%
  mutate_if(is.numeric, function(x) x/sum(x)) %>%
  pivot_longer(-variable, names_to="varimp_type", values_to="normalised_importance") %>%
  ggplot(aes(x=variable, y=normalised_importance, fill=varimp_type)) + 
  geom_col(position="dodge") + 
  ggtitle(sprintf("Spearman(Gini,perm)=%.2f",
                  cor(rf_importance$IncNodePurity, rf_importance$`%IncMSE`, method="spearman")))
