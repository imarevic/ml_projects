library(h2o)
library(caTools)
library(splitstackshape)
library(h2oEnsemble)

# ===== Import datasets ===== #
origData = read.csv("train.csv", sep=",")
origtestSetMerc = read.csv("test.csv", sep=",")

testData = origData[, c(3:ncol(origData), 2)]


# ===== Remove rows with missing values and id ===== #
testData = testData[complete.cases(testData), 1:ncol(testData)]
newtestSetMerc = origtestSetMerc[complete.cases(origtestSetMerc), 2:ncol(origtestSetMerc)]


# ===== Init connection ===== #
h2o.init(nthreads = -1)


# ===== Split data into training, validation and test===== #
splits = h2o.splitFrame(as.h2o(testData),          
                        c(0.6,0.2), 
                        seed=1234) 
trainingSet = h2o.assign(splits[[1]], "train.hex")   
validationSet = h2o.assign(splits[[2]], "valid.hex")  
testSet = h2o.assign(splits[[3]], "test.hex") 

# ===== Get outcome and features names ===== #

outcome = "y"
features = setdiff(names(trainingSet), outcome)

# ===== Create a classifier ===== #
classifier = h2o.deeplearning(
    model_id="dl_model", 
    training_frame=trainingSet, 
    validation_frame=validationSet,
    x=features,
    y=outcome,
    hidden=c(150, 150, 150, 150, 150),                  
    epochs=1000,
    input_dropout_ratio = 0.05,
    activation = "RectifierWithDropout",
    train_samples_per_iteration=-2,
    l1=1e-5,  # Regularization
    l2=1e-5,  # Regularization 
    max_w2=10
)
summary(classifier)

# ===== Obtain predictions ===== #
yPred = h2o.predict(classifier, 
                    type="response", 
                    newdata = testSet[-ncol(testSet)])

yPred = as.vector(yPred)
yEmp = as.vector(testSet$y)
mse = mean((yPred - yEmp)^2); mse
R2 = 1 - (sum((yEmp-yPred)^2)/sum((yEmp-mean(yEmp))^2)); R2

# ===== Obtain predictions on Mercedes Test Set===== #
testSetMerc = as.h2o(newtestSetMerc)
yMerc = h2o.predict(classifier, 
                    type="response", 
                    newdata = testSetMerc[-ncol(testSetMerc)])

yMerc = as.vector(yMerc)


# ===== GRID SEARCH APPROACH ===== #
# Setting hyperparameters
hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(100,100),c(150,150)),
  input_dropout_ratio=c(0,0.05),
  hidden_dropout_ratio = c(0.2, 0.3, 0.4, 0.5, 0.6),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

# Set stopping rules for model selection
searchCriteria = list(strategy = "RandomDiscrete", max_runtime_secs = 10000, max_models = 500, seed=1234567, 
                      stopping_rounds=100, stopping_tolerance=1e-2)
# Create classifier
dlRandomGrid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dlRandomGrid",
  training_frame=trainingSet,
  validation_frame=validationSet, 
  x=features, 
  y=outcome,
  epochs=100,
  stopping_metric="AUTO",
  max_w2=10,
  hyper_params = hyper_params,
  search_criteria = searchCriteria
)         

# Retrieve grid
grid <- h2o.getGrid("dlRandomGrid",sort_by="mse",decreasing=FALSE)
grid

# Look at best model (model with lowest MSE)
grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]])
best_model

# Prediction on Mercedes Test Data
yMercGS = h2o.predict(best_model, 
                    type="response", 
                    newdata = testSetMerc[-ncol(testSetMerc)])

yMercGS = as.vector(yMercGS)
finalMercData = origtestSetMerc
finalMercData["y"] <- yMercGS
subMCData = subset(finalMercData, select = c("ID", "y"))

write.csv(subMCData, file = "GSSubmission.csv", row.names = FALSE)

# ===== ENSEMBLE LEARNING APPROACH===== #
# specifiy models and metamodel
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper")
metalearner <- "h2o.glm.wrapper"
family = "gaussian"

# get training data
splits = h2o.splitFrame(as.h2o(testData),          
                        c(0.6,0.2), 
                        seed=1234) 

trainSetEL = h2o.assign(splits[[1]], "train.hex")
testSetEL = h2o.assign(splits[[2]], "test.hex")
outcomeEL = "y"
featuresEL = setdiff(names(trainSetEL), outcomeEL)

# Create Ensemble Classifier
ensClassifier <- h2o.ensemble(x = featuresEL, y = outcomeEL, 
                    training_frame = trainSetEL, 
                    model_id = "ensClassModel",
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))

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
h2o.shutdown(prompt=FALSE)
