# Case Study 4

# This script performs regression and classification analysis on two datasets:
# 1. Boston Housing Data: Predicts median home value using a regression tree (CART)
#    and linear regression.
# 2. German Credit Score Data: Predicts credit default using a classification tree
#    and logistic regression, incorporating custom misclassification costs.

# Load necessary libraries
library(MASS)
library(rpart)
library(rpart.plot)
library(dplyr)
library(ROCR) # Loaded here as it's used in the German Credit Score section

# -------------------------------------------------------------------------
# 1. Boston Housing Data
# -------------------------------------------------------------------------

# Load the Boston dataset
data(Boston)

# Set a random seed for reproducibility (optional, but good practice for sampling)
# set.seed(123)

# Create training and testing splits (90% train, 10% test)
sample_index <- sample(nrow(Boston), nrow(Boston) * 0.90)
train <- Boston[sample_index, ]
test <- Boston[-sample_index, ]

# (i) Fit a regression tree (CART) on training data and report in-sample MSE
boston_rpart <- rpart(formula = medv ~ ., data = train)
print(boston_rpart)
prp(boston_rpart, digits = 4, extra = 1)

# In-Sample Prediction for Regression Tree
boston_train_pred_tree = predict(boston_rpart)
mse.tree <- mean((boston_train_pred_tree - train$medv)^2)
cat("In-sample MSE for Regression Tree:", mse.tree, "\n")

# (ii) Report the model's out-of-sample MSE performance and compare with linear regression

# Out-of-Sample Prediction for Regression Tree
boston_test_pred_tree = predict(boston_rpart, test)
mpse.tree <- mean((boston_test_pred_tree - test$medv)^2)
cat("Out-of-sample MSE for Regression Tree:", mpse.tree, "\n")

# Comments on Regression Tree Performance
# From the dataset, the in-sample MSE for the regression tree is generally lower
# than the out-of-sample MSE, suggesting some overfitting within the training data.

# Compare with a Linear Regression Model
# Fit a linear regression model, excluding 'indus' and 'age' as per original script
boston.reg = lm(medv ~ . - indus - age, data = train)

# In-Sample Prediction for Linear Regression
boston_train_pred_reg = predict(boston.reg, train)
mse.reg <- mean((boston_train_pred_reg - train$medv)^2)
cat("In-sample MSE for Linear Regression:", mse.reg, "\n")

# Out-of-Sample Prediction for Linear Regression
boston_test_pred_reg = predict(boston.reg, test)
mpse.reg <- mean((boston_test_pred_reg - test$medv)^2)
cat("Out-of-sample MSE for Linear Regression:", mpse.reg, "\n")

# Comparison of Models
# In comparison to the out-of-sample MSE, the regression tree model often
# performs better on this test set compared to the linear regression model,
# which can have a higher MSE depending on the specific data split.

# -------------------------------------------------------------------------
# 2. German Credit Score Data
# -------------------------------------------------------------------------

# Load the credit default data
credit_data <- read.csv(file = "https://xiaoruizhu.github.io/Data-Mining-R/lecture/data/credit_default.csv", header = T)

# Rename the target variable for clarity
credit_data <- rename(credit_data, default = default.payment.next.month)

# Convert categorical variables to factors
credit_data$SEX <- as.factor(credit_data$SEX)
credit_data$EDUCATION <- as.factor(credit_data$EDUCATION)
credit_data$MARRIAGE <- as.factor(credit_data$MARRIAGE)

# Create training and testing splits (80% train, 20% test)
# set.seed(123) # Optional: for reproducibility of this split as well
index <- sample(nrow(credit_data), nrow(credit_data) * 0.80)
credit_train = credit_data[index, ]
credit_test = credit_data[-index, ]

# Fit a default classification tree (without custom loss matrix)
credit_rpart0 <- rpart(formula = default ~ ., data = credit_train, method = "class")
cat("\nDefault Classification Tree (credit_rpart0) - Training Confusion Matrix:\n")
pred0 <- predict(credit_rpart0, type = "class")
print(table(credit_train$default, pred0, dnn = c("True", "Pred")))

# Fit a classification tree with a custom loss matrix
# Loss matrix:
#        Predicted
# True    0   1
#    0    0   1  (Cost of misclassifying actual 0 as 1 is 1)
#    1    5   0  (Cost of misclassifying actual 1 as 0 is 5)
credit_rpart <- rpart(formula = default ~ ., data = credit_train, method = "class", parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)))
print(credit_rpart)
prp(credit_rpart, extra = 1)

# Predictions with the custom-cost classification tree
cat("\nCustom-Cost Classification Tree (credit_rpart) - Training Confusion Matrix:\n")
credit_train.pred.tree1 <- predict(credit_rpart, credit_train, type = "class")
print(table(credit_train$default, credit_train.pred.tree1, dnn = c("Truth", "Predicted")))

cat("\nCustom-Cost Classification Tree (credit_rpart) - Test Confusion Matrix:\n")
credit_test.pred.tree1 <- predict(credit_rpart, credit_test, type = "class")
print(table(credit_test$default, credit_test.pred.tree1, dnn = c("Truth", "Predicted")))

# Define a custom cost function
cost <- function(r, phat) {
  weight1 <- 5 # Cost for false negative (actual 1, predicted 0)
  weight0 <- 1 # Cost for false positive (actual 0, predicted 1)
  pcut <- weight0 / (weight1 + weight0) # Optimal cutoff for given costs
  c1 <- (r == 1) & (phat < pcut) # True if actual 1 but predict 0
  c0 <- (r == 0) & (phat > pcut) # True if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

# Calculate cost for the custom-cost classification tree
train_cost_tree <- cost(credit_train$default, predict(credit_rpart, credit_train, type = "prob"))
cat("\nTraining Cost for Custom-Cost Classification Tree:", train_cost_tree, "\n")

test_cost_tree <- cost(credit_test$default, predict(credit_rpart, credit_test, type = "prob"))
cat("Test Cost for Custom-Cost Classification Tree:", test_cost_tree, "\n")

# Fit Logistic Regression Model
credit_glm <- glm(default ~ ., data = credit_train, family = binomial)

# Get binary prediction from logistic regression on test set
credit_test_pred_glm <- predict(credit_glm, credit_test, type = "response")

# Calculate cost for Logistic Regression using the test set
test_cost_glm <- cost(credit_test$default, credit_test_pred_glm)
cat("Test Cost for Logistic Regression:", test_cost_glm, "\n")

# Confusion matrix for Logistic Regression with optimal cutoff (1/6 for weights 5 and 1)
cat("\nLogistic Regression - Test Confusion Matrix (using 1/6 cutoff):\n")
print(table(credit_test$default, as.numeric(credit_test_pred_glm > 1 / 6), dnn = c("Truth", "Predicted")))


# Evaluate Classification Tree using ROC Curve and AUC
# Get probabilities for the positive class (default = 1) from the custom-cost tree
credit_test_prob_rpart = predict(credit_rpart, credit_test, type = "prob")

# Create ROCR prediction object
pred = prediction(credit_test_prob_rpart[, 2], credit_test$default)

# Plot ROC curve
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE, main = "ROC Curve for Classification Tree (Test Set)")

# Calculate Area Under the Curve (AUC)
auc_value <- slot(performance(pred, "auc"), "y.values")[[1]]
cat("\nAUC for Classification Tree (Test Set):", auc_value, "\n")

# Generate confusion matrix for classification tree using the optimal cost-based cutoff
credit_test_pred_rpart = as.numeric(credit_test_prob_rpart[, 2] > 1 / (5 + 1))
cat("\nCustom-Cost Classification Tree - Test Confusion Matrix (using 1/6 cutoff):\n")
print(table(credit_test$default, credit_test_pred_rpart, dnn = c("Truth", "Predicted")))
