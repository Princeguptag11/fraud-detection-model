library(dplyr)
library(caret)
library(e1071)
data <- read.csv("Fraud.csv")
summary(data)
data <- na.omit(data)
sum(is.na(data))
data_clean <- na.omit(data)
for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
  }
  # Assuming 'data' is your dataframe
  for (col in names(data)) {
    # Check if the column is numeric
    if (is.numeric(data[[col]])) {
      # Impute missing values with the column's mean, excluding NAs
      data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
      cat("Imputed NA in column:", col, "\n") # Optional: Print which column was imputed
    }
  }
  # Identify outliers using the IQR method as an example
  for (col in names(data)) {
    if (is.numeric(data[[col]])) {
      Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      
      # Option 1: Remove outliers
      # data <- data[data[[col]] >= lower_bound & data[[col]] <= upper_bound, ]
      
      # Option 2: Cap outliers
      data[[col]] <- ifelse(data[[col]] < lower_bound, lower_bound, data[[col]])
      data[[col]] <- ifelse(data[[col]] > upper_bound, upper_bound, data[[col]])
    }
  }
  # Detecting multi-collinearity using Variance Inflation Factor (VIF)
  library(car)
  # Assuming your model formula would be something like this:
  model <- lm(isFraud ~ ., data=data)
  vif(model)  # Check VIF scores for each predictor
  
  # Consider removing or combining features with high VIF scores (> 5 or 10 as common thresholds)
data_adjusted <- data[, !(names(data) %in% c("oldbalanceOrg"))]
data_adjusted$feature_new <- data$amount + data$oldbalanceOrg # Example combination  
library(dplyr) 
data_adjusted <- select(data, -oldbalanceOrg)
# Using glmnet for Lasso (L1) regularization
library(glmnet)
x <- model.matrix(isFraud ~ . - 1, data=data_adjusted) # Prepare model matrix, excluding intercept
y <- data_adjusted$isFraud
# Fit the model with lambda chosen via cross-validation
cv_fit <- cv.glmnet(x, y, family="binomial", alpha=1) # alpha=1 for Lasso
plot(cv_fit)
set.seed(123) # Ensure reproducibility
data_sample <- data_adjusted[sample(nrow(data_adjusted), size = 10000), ]
memory.limit(size=16000)
data_adjusted_slim <- select(data_adjusted, -c(less_important_var1, less_important_var2))
data_adjusted_slim <- select(data_adjusted, -column1, -column2)
less_important_var1 <- "column1"
less_important_var2 <- "column2"
data_adjusted_slim <- select(data_adjusted, -all_of(c(less_important_var1, less_important_var2)))
data_adjusted_slim <- dplyr::select(data_adjusted, -dplyr::all_of(c(less_important_var1, less_important_var2)))
data_adjusted_slim <- dplyr::select(data_adjusted, -dplyr::all_of(c(less_important_var1, less_important_var2)))
data_adjusted_slim <- dplyr::select(data_adjusted, -column1, -column2)
print(names(data_adjusted))
# Example: Creating an interaction term
data_adjusted$interaction_feature <- data_adjusted$variable1 * data_adjusted$variable2
indexes <- createDataPartition(data_adjusted$isFraud, p=0.8, list=FALSE)
train_set <- data_adjusted[indexes, ]
test_set <- data_adjusted[-indexes, ]
table(data_adjusted$isFraud)
set.seed(123)  # For reproducibility
sample_size <- floor(0.8 * nrow(data_adjusted))
indexes <- sample(seq_len(nrow(data_adjusted)), size = sample_size)
train_set <- data_adjusted[indexes, ]
test_set <- data_adjusted[-indexes, ]
# Build a logistic regression model
model <- glm(isFraud ~ ., data = train_set, family = binomial)
predictions <- predict(model, newdata = test_set, type = "response")
predicted_class <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix <- table(Predicted = predicted_class, Actual = test_set$isFraud)
print(confusionMatrix)
# Example: Imputing 'oldbalanceOrg' with the mean (or another appropriate value) if it was removed due to NA handling
if("oldbalanceOrg" %in% names(test_set)) {
  +   mean_value <- mean(train_set$oldbalanceOrg, na.rm = TRUE) # Calculate mean from the training set
  +   test_set_imputed$oldbalanceOrg <- ifelse(is.na(test_set$oldbalanceOrg), mean_value, test_set$oldbalanceOrg)
  + } else {
    +   test_set_imputed$oldbalanceOrg <- rep(mean_value, nrow(test_set_imputed)) # Add 'oldbalanceOrg' back if it was removed
    + }
> # Example: Imputing 'oldbalanceOrg' with the mean (or another appropriate value) if it was removed due to NA handling
  > if("oldbalanceOrg" %in% names(test_set)) {
    +   mean_value <- mean(train_set$oldbalanceOrg, na.rm = TRUE) # Calculate mean from the training set
    +   test_set_imputed$oldbalanceOrg <- ifelse(is.na(test_set$oldbalanceOrg), mean_value, test_set$oldbalanceOrg)
    + } else {
      +   test_set_imputed$oldbalanceOrg <- rep(mean_value, nrow(test_set_imputed)) # Add 'oldbalanceOrg' back if it was removed
      + }
> predictions_imputed <- predict(model, newdata = test_set_imputed, type = "response")
> binary_predictions <- ifelse(predictions_imputed > 0.5, 1, 0)
> library(caret)
> actual_responses <- test_set_imputed$isFraud
> conf_matrix <- confusionMatrix(factor(binary_predictions, levels=c(0, 1)), factor(actual_responses, levels=c(0, 1)))
> print(conf_matrix)
# Assuming 'train_set' and 'test_set' are already defined
# Check if 'oldbalanceOrg' exists in 'test_set' and impute missing values if necessary
if("oldbalanceOrg" %in% names(test_set)) {
  mean_value <- mean(train_set$oldbalanceOrg, na.rm = TRUE)  # Calculate mean from the training set
  test_set$oldbalanceOrg <- ifelse(is.na(test_set$oldbalanceOrg), mean_value, test_set$oldbalanceOrg)
} else {
  test_set$oldbalanceOrg <- rep(mean(train_set$oldbalanceOrg, na.rm = TRUE), nrow(test_set))  # Add 'oldbalanceOrg' back if it was removed
}
predictions_imputed <- predict(model, newdata = test_set, type = "response")
binary_predictions <- ifelse(predictions_imputed > 0.5, 1, 0)
# Ensure 'actual_responses' is a factor with appropriate levels
actual_responses <- factor(test_set$isFraud, levels = c(0, 1))

# Calculate the confusion matrix
conf_matrix <- confusionMatrix(factor(binary_predictions, levels = c(0, 1)), actual_responses)

# Print the confusion matrix and related statistics
print(conf_matrix)