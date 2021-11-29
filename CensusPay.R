## @knitr CensusPayR

# Note: This script will take a while to run. In particular the knn and random forest algorithms with tuning grids will take
# more time. please be patient if you happen to execute it. The execution report is available in the github location as well

# Execute the given source code for the project
source("DatasetProcessingCode.R")

if (!require(randomForest))
  install.packages("randomForest", repos = "http://cran.us.r-project.org")

if (!require(purrr))
  install.packages("purrr", repos = "http://cran.us.r-project.org")

if (!require(e1071))
  install.packages("e1071")

if (!require(pROC))
  install.packages("pRoc")

if (!require(ROCit))
  install.packages("ROCit")

library(caret)
library(gridExtra)
library(kableExtra)
library(randomForest)
library(purrr)
library(e1071)
library(caTools)
library(pROC)
library(ROCit)

#set the seed for reproducible results
set.seed(2008, sample.kind = "Rounding")

# the simplest possible machine algorithm: guessing the outcome
seat_of_the_pants <-
  sample(c("Above50K", "AtBelow50K"), length(test_index), replace = TRUE) %>% factor(levels = levels(adultpayclean_validation$income))
# calculate the accuracy of this sampling
accuracy_guess <-
  mean(seat_of_the_pants == adultpayclean_validation$income)

# build a confusion matrix for this simple model
table(predicted = seat_of_the_pants, actual = adultpayclean_validation$income)

# tabulate accuracy by income levels
adultpayclean_validation %>%
  mutate(y_hat = seat_of_the_pants) %>%
  group_by(income) %>%
  summarize(accuracy = mean(y_hat == income))

# confusion matrix using R function
cm <-
  confusionMatrix(data = seat_of_the_pants , reference = adultpayclean_validation$income)
# display the confusion matrix
cm

#record the sensitivity, specificity, and prevalence
sensitivity_guess <- cm$byClass[["Sensitivity"]]
specificity_guess <- cm$byClass[["Specificity"]]
prevalence_guess <- cm$byClass[["Prevalence"]]
f1_guess <- cm$byClass[["F1"]]

#find the area under the curve/ROC
auc(ifelse(adultpayclean_validation$income == "Above50K",1,2), ifelse(seat_of_the_pants == "Above50K",1,2))

set.seed(2008)
#logistic linear model
# create the model
lm_fit <- adultpayclean_train %>%
  mutate(y = as.numeric(income == "Above50K")) %>%
  lm(y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship + education,
     data = .)

# predict using test set
p_hat_logit <- predict(lm_fit, newdata = adultpayclean_validation)

#translate predicted data into factor
y_hat_logit <-
  ifelse(p_hat_logit > 0.5, "Above50K", "AtBelow50K") %>% factor

#compare the predicted vs observed values and use confusionMatrix to get the accuracy and other metrics
cm_lm <-
  confusionMatrix(y_hat_logit, adultpayclean_validation$income)
accuracy_lm <-
  confusionMatrix(y_hat_logit, adultpayclean_validation$income)$overall[["Accuracy"]]

cm_lm

#record the sensitivity, specificity, and prevalence
sensitivity_lm <- cm_lm$byClass[["Sensitivity"]]
specificity_lm <- cm_lm$byClass[["Specificity"]]
prevalence_lm <- cm_lm$byClass[["Prevalence"]]
f1_lm <- cm_lm$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_logit) == "Above50K",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)

set.seed(2008)
#general linear model
#create the glm model
glm_fit <- adultpayclean_train %>%
  mutate(y = as.numeric(income == "Above50K")) %>%
  glm(
    y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship + education,
    data = .,
    family = "binomial"
  )

# predict using validation set
p_hat_logit <- predict(glm_fit, newdata = adultpayclean_validation)

# translate the predicted data into factor
y_hat_logit <-
  ifelse(p_hat_logit > 0.5, "Above50K", "AtBelow50K") %>% factor

# compare the predicted vs observed values and use confusionMatrix to get the accuracy and other metrics for the glm model
cm_glm <-
  confusionMatrix(y_hat_logit, adultpayclean_validation$income)
accuracy_glm <-
  confusionMatrix(y_hat_logit, adultpayclean_validation$income)$overall[["Accuracy"]]

cm_glm

#record the sensitivity, specificity, and prevalence
sensitivity_glm <- cm_glm$byClass[["Sensitivity"]]
specificity_glm <- cm_glm$byClass[["Specificity"]]
prevalence_glm <- cm_glm$byClass[["Prevalence"]]
f1_glm <- cm_glm$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_logit) == "Above50K",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)

#Naive bayes
set.seed(2008)
#create the naive bayes model
train_nb <- adultpayclean_train %>%
   mutate(y = as.factor(income == "Above50K")) %>% 
   naiveBayes(y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship+education,data = .)

#predict using the validation dataset
y_hat_nb <- predict(train_nb, newdata = adultpayclean_validation)
#create the confusion matrix
cm_tab <- table(adultpayclean_validation$income == "Above50K", y_hat_nb)
cm_nb <- confusionMatrix(cm_tab)
cm_nb

#get the accuracy, sensitivity, specificity, prevalence and, F1 score
accuracy_nb <- cm_nb$overall[["Accuracy"]]
sensitivity_nb <- cm_nb$byClass[["Sensitivity"]]
specificity_nb <- cm_nb$byClass[["Specificity"]]
prevalence_nb <- cm_nb$byClass[["Prevalence"]]  
f1_nb <- cm_nb$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_nb) == "TRUE",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)

# translate income factor into binary outcome
temp <- adultpayclean_train %>%
  mutate(y = as.factor(income == "Above50K"))

#k-nearest neighbors with a train control and tuning
set.seed(2008)
# train control to use 10% of the observations each to speed up computations
control <- trainControl(method = "cv", number = 10, p = .9)
# train the model using knn. choose the best k value using tuning algorithm
train_knn <-
  train(
    y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship + education,
    method = "knn",
    data = temp,
    tuneGrid = data.frame(k = seq(3, 71, 2)),
    trControl = control
  )

#plot the resulting model
ggplot(train_knn, highlight = TRUE)
#verify which k value was used
train_knn$bestTune
train_knn$finalModel

#use this trained model to predict raw knn predictions
y_hat_knn <-
  predict(train_knn, adultpayclean_validation, type = "raw")

# compare the predicted and observed values using confusionMatrix to get the accuracy and other metrics
cm_knn <-
  confusionMatrix(y_hat_knn,
                  as.factor(adultpayclean_validation$income == "Above50K"))
accuracy_knn <-
  confusionMatrix(y_hat_knn,
                  as.factor(adultpayclean_validation$income == "Above50K"))$overall[["Accuracy"]]

cm_knn

#record the sensitivity, specificity, and prevalence
sensitivity_knn <- cm_knn$byClass[["Sensitivity"]]
specificity_knn <- cm_knn$byClass[["Specificity"]]
prevalence_knn <- cm_knn$byClass[["Prevalence"]]
f1_knn <- cm_knn$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_knn) == "TRUE",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)



#k-nearest classification using tuning function
set.seed(2008)

#train the model using knn3 classification
ks <- seq(3, 251, 2)
knntune <- map_df(ks, function(k) {
  temp <- adultpayclean_train %>%
    mutate(y = as.factor(income == "Above50K"))
  temp_test <- adultpayclean_validation %>%
    mutate(y = as.factor(income == "Above50K"))
  #create the kkn3 model
  knn_fit <-
    knn3(
      y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship+education,
      data = temp,
      k = k
    )
  #predict the model for the current k
  y_hat <- predict(knn_fit, temp, type = "class")
  #get the confusionmatrix for the current k
  cm_train <- confusionMatrix(y_hat, temp$y)
  train_error <- cm_train$overall["Accuracy"]
  #do the same for test model
  y_hat <- predict(knn_fit, temp_test, type = "class")
  cm_test <- confusionMatrix(y_hat, temp_test$y)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})
#get the accuracy for the k with maximum accuracy
accuracy_knntune <- max(knntune$test)
#get the confusion matrix for that k
knn_fit <-
  knn3(
    y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship+education,
    data = temp,
    k = ks[which.max(knntune$test)]
  )
#predict the knn tune using the model for the k neighbor
y_hat_knntune <- predict(knn_fit, adultpayclean_validation, type = "class")
cm_knntune <- confusionMatrix(y_hat_knntune, as.factor(adultpayclean_validation$income == "Above50K"))

cm_knntune

#record the sensitivity, specificity, and prevalence
sensitivity_knntune <- cm_knntune$byClass[["Sensitivity"]]
specificity_knntune <- cm_knntune$byClass[["Specificity"]]
prevalence_knntune <- cm_knntune$byClass[["Prevalence"]]
f1_knntune <- cm_knntune$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_knntune) == "TRUE",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)

#recursive partitioning using rpart
set.seed(2008)
#train the model with the recursive partitioning
train_rpart <-
  train(
    y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship+education,
    method = "rpart",
    tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
    data = temp
  )
#predict the outcomes with this model
y_hat_rpart <- predict(train_rpart, adultpayclean_validation)
#confusion matrix for the rpart model
cm_rpart <-
  confusionMatrix(y_hat_rpart,
                  as.factor(adultpayclean_validation$income  == "Above50K"))
#get the accuracy
accuracy_rpart <-
  confusionMatrix(y_hat_rpart,
                  as.factor(adultpayclean_validation$income  == "Above50K"))$overall["Accuracy"]

cm_rpart
#record the sensitivity, specificity, and prevalence
sensitivity_rpart <- cm_rpart$byClass[["Sensitivity"]]
specificity_rpart <- cm_rpart$byClass[["Specificity"]]
prevalence_rpart <- cm_rpart$byClass[["Prevalence"]]
f1_rpart <- cm_rpart$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_rpart) == "TRUE",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)

#random forest
set.seed(2008)
#train the vanilla random forest model 
train_rf <-
  randomForest(y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship+education,
               data = temp)

y_hat_rf <- predict(train_rf, adultpayclean_validation)

#create the confusionMatrix
cm_rf <-
  confusionMatrix(
    y_hat_rf,
    as.factor(adultpayclean_validation$income  == "Above50K")
  )
#get the accuracy
accuracy_rf <-
  confusionMatrix(
    y_hat_rf,
    as.factor(adultpayclean_validation$income  == "Above50K")
  )$overall["Accuracy"]

cm_rf

#record the sensitivity, specificity, and prevalence
sensitivity_rf <- cm_rf$byClass[["Sensitivity"]]
specificity_rf <- cm_rf$byClass[["Specificity"]]
prevalence_rf <- cm_rf$byClass[["Prevalence"]]
f1_rf <- cm_rf$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_rf) == "TRUE",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)

# Plot the error rate chart for the random forest
plot(train_rf)
legend("center", ifelse (colnames(train_rf$err.rate) == "FALSE","AtBelow50K",ifelse (colnames(train_rf$err.rate) == "TRUE","Above50K","OOB")),col=1:4,cex=0.8,fill=1:4)

set.seed(2008)
#random forest with tuning
nodesize <- seq(1, 90, 10)
acc <- sapply(nodesize, function(ns) {
  #train the model with tuning
  train(
    y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship+education,
    method = "rf",
    data = temp,
    tuneGrid = data.frame(mtry = 2),
    nodesize = ns
  )$results$Accuracy
})
qplot(nodesize, acc)
#get the trained model for the max node size
train_rf_2 <-
  randomForest(
    y ~ age + eduyears + sex + race + hoursperweek + maritalstatus + relationship+education,
    data = temp,
    nodesize = nodesize[which.max(acc)]
  )
#predict the outcomes
y_hat_rf2 <- predict(train_rf_2, adultpayclean_validation)
#get the confusion matrix for random forest model
cm_rf2 <-
  confusionMatrix(
    predict(train_rf_2, adultpayclean_validation),
    as.factor(adultpayclean_validation$income  == "Above50K")
  )
#get the accuracy
accuracy_rftune <-
  confusionMatrix(
    predict(train_rf_2, adultpayclean_validation),
    as.factor(adultpayclean_validation$income  == "Above50K")
  )$overall["Accuracy"]

cm_rf2

#record the sensitivity, specificity, and prevalence
sensitivity_rf2 <- cm_rf2$byClass[["Sensitivity"]]
specificity_rf2 <- cm_rf2$byClass[["Specificity"]]
prevalence_rf2 <- cm_rf2$byClass[["Prevalence"]]
f1_rf2 <- cm_rf2$byClass[["F1"]]

#Find the ROC and plot it. Show the AUC as well
pROC_bin <- ROCit::rocit(ifelse(adultpayclean_validation$income == "Above50K",1,0), ifelse(unname(y_hat_rf2) == "TRUE",1,0),method="bin")
ciROC_bin95 <- ROCit::ciROC(pROC_bin,level = 0.95)
plot(ciROC_bin95, col = 1, values=TRUE)
lines(ciROC_bin95$TPR~ciROC_bin95$FPR, col = 2, lwd = 2)
ROCit::ciAUC(pROC_bin)

# Plot the error rate chart for the random forest
plot(train_rf_2)
legend("center", ifelse (colnames(train_rf_2$err.rate) == "FALSE","AtBelow50K",ifelse (colnames(train_rf_2$err.rate) == "TRUE","Above50K","OOB")),col=1:4,cex=0.8,fill=1:4)

# tabulate all the accuracy results with sensitivity and specificity
accuracy_results <-
  matrix(
    c(
      "Plain old guess",
      round(accuracy_guess, 5),
      round(sensitivity_guess, 5),
      round(specificity_guess, 5),
      round(prevalence_guess, 5),
      round(f1_guess, 5),
      "linear model",
      round(accuracy_lm, 5),
      round(sensitivity_lm, 5),
      round(specificity_lm, 5),
      round(prevalence_lm, 5),
      round(f1_lm, 5),
      "General linear model",
      round(accuracy_glm, 5),
      round(sensitivity_glm, 5),
      round(specificity_glm, 5),
      round(prevalence_glm, 5),
      round(f1_glm, 5),
      "naive bayes",
      round(accuracy_nb, 5),
      round(sensitivity_nb, 5),
      round(specificity_nb, 5),
      round(prevalence_nb, 5),
      round(f1_nb, 5),
      "knn",
      round(accuracy_knn, 5),
      round(sensitivity_knn, 5),
      round(specificity_knn, 5),
      round(prevalence_knn, 5),
      round(f1_knn, 5),
      "knn tune",
      round(accuracy_knntune, 5),
      round(sensitivity_knntune, 5),
      round(specificity_knntune, 5),
      round(prevalence_knntune, 5),
      round(f1_knntune, 5),
      "rpart",
      round(accuracy_rpart, 5),
      round(sensitivity_rpart, 5),
      round(specificity_rpart, 5),
      round(prevalence_rpart, 5),
      round(f1_rpart, 5),
      "rf",
      round(accuracy_rf, 5),
      round(sensitivity_rf, 5),
      round(specificity_rf, 5),
      round(prevalence_rf, 5),
      round(f1_rf, 5),
      "rf tune",
      round(accuracy_rftune, 5),
      round(sensitivity_rf2, 5),
      round(specificity_rf2, 5),
      round(prevalence_rf2, 5),
      round(f1_rf2, 5)
    ),
    nrow = 9,
    ncol = 6,
    byrow = TRUE,
    dimnames = list(
      c("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."),
      c(
        "Method",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Prevalence",
        "F1"
      )
    )
  )
#style the table with knitr
accuracy_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))