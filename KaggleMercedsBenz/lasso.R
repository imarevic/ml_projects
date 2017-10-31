library(caTools)
library(readr)
library(glmnet)

# Read in test and training data and match the dimensions
data = read.csv("train.csv")
test_f = read.csv("test.csv")
test_f <- as.data.frame(append(test_f, list(y = NA), after = 1))

# Loop over levels from test set and match them with training set
for (f in 3:10) {
  if (levels(test_f[,f]) > levels(data[,f])) {    
    levels(test_f[,f]) = levels(data[,f])       
  } else {
    levels(data[,f]) = levels(test_f[,f])      
  }
}

# split data in train and test set
data$split= sample.split(data$y,SplitRatio=0.8)
train = subset(data, split==T)
test = subset(data, split==F)

# extract features and encode all factors with > 2 levels
features_train = train[,grep("X",names(train))]
out = matrix(model.matrix( ~ ., features_train[,1:8]),nrow=nrow(features_train))
X_train = cbind(out,features_train[,-c(1:8)])

# same for test set
features_test = test[,grep("X",names(test))]
out = matrix(model.matrix( ~ ., features_test[,1:8]),nrow=nrow(features_test))
X_test = cbind(out,features_test[,-c(1:8)])

# and for the test set given by Mercedes
features_test_f = test_f[,grep("X",names(test_f))]
out = matrix(model.matrix( ~ ., features_test_f[,1:8]),nrow=nrow(features_test_f))
X_test_f = cbind(out,features_test_f[,-c(1:8)])


# convert to matrices
X_train = data.matrix(X_train)
X_test = data.matrix(X_test)
X_test_f = data.matrix(X_test_f)
y_train = train$y


# Fitting the model
set.seed(999)
cv.lasso <- cv.glmnet(X_train, y_train, alpha=1, intercept=T, standardize=T, type.measure='deviance')

# Results
plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)

test$y_pred = predict(cv.lasso, newx = X_test, s=cv.lasso$lambda.min)

# Mean Square Error (who can beat me? ;)
mse = mean((test$y_pred - test$y)^2)
mse

# R-Squared for y_pred in relation to actual y
R2 = 1 - (sum((test$y-test$y_pred)^2)/sum((test$y-mean(test$y))^2))
R2

# Prediction on test data given by Mercedes
test_f$y = predict(cv.lasso, newx = X_test_f, s=cv.lasso$lambda.min)

# Format prediction dataset and write to disk
subdata <- subset(test_f, select=c(ID, y))
write.csv(subdata, file ="SubmissionFile.csv")
