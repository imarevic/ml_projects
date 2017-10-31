library(h2o)
library(caTools)
library(splitstackshape)
library(h2oEnsemble)

# Import datasets
origData = read.csv("train.csv", sep=",")
origtestSetMerc = read.csv("test.csv", sep=",")

testData = origData[, c(3:ncol(origData), 2)]


# Remove rows with missing values and id
testData = testData[complete.cases(testData), 1:ncol(testData)]
newtestSetMerc = origtestSetMerc[complete.cases(origtestSetMerc), 2:ncol(origtestSetMerc)]

# ===== Init connection ===== #
h2o.init()

# Split training data
splits = h2o.splitFrame(as.h2o(testData),          
                        c(0.75,0.2), 
                        seed=1234) 

trainSetEL = h2o.assign(splits[[1]], "train.hex")
testSetEL = h2o.assign(splits[[2]], "test.hex")
outcomeEL = "y"
featuresEL = setdiff(names(trainSetEL), outcomeEL)

# ===== ENSEMBLE LEARNING APPROACH===== #
# specifiy base learner models
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 100, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 100, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 100, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)

# Add base learners to learner object
learner <- c("h2o.gbm.wrapper", "h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3", "h2o.gbm.4", "h2o.gbm.5", "h2o.gbm.6", "h2o.gbm.7", "h2o.gbm.8")

# Specify metalearner
metalearner <- "h2o.gbm.wrapper"
family = "gaussian"

# Create Ensemble Classifier
ensClassifier <- h2o.ensemble(x = featuresEL, y = outcomeEL, 
                              training_frame = trainSetEL, 
                              model_id = "ensClassModel",
                              family = family, 
                              learner = learner, 
                              metalearner = metalearner,
                              cvControl = list(V = 6))

# obtain predictions
ensPred <- h2o.ensemble_performance(ensClassifier, newdata = testSetEL)
print(ensPred, metric = "MSE")

# prediction on Mercedes Test Data
ensTestSetMerc = as.h2o(newtestSetMerc)
predMercEL = predict(ensClassifier, newdata = ensTestSetMerc)
yMercEL = as.vector(predMercEL$pred)

finalMercDataEL <- origtestSetMerc
finalMercDataEL["y"] <- yMercEL
subMCDataEL <- subset(finalMercDataEL, select = c("ID", "y"))

#write submissions file to disk
write.csv(subMCDataEL, file = "ELSubmission.csv", row.names = FALSE)

# ===== Disconnect ===== #
h2o.shutdown(prompt = FALSE)
