#### MS&E 226: Project Part 2 ####
### Ankita Banerjea & Rehana Mohammed ###

# Loading required packages
library(knitr)
library(tidyverse)
library(plyr)
library(dplyr)
library(cvTools)
library(data.table)
library(GGally)
library(ggplot2)
library(glmnet)
library(corrplot)
library(RColorBrewer)
library(wesanderson)
library(glmnet)
library(broom)
library(pROC)
library(ROCR)
library(caret)
library(arm)
library(plotmo)
library(boot)
library(stargazer)
library(xtable)

# Loading dataset - edit path
college_dist <- read.csv("/Users/ankitabanerjea/Downloads/CollegeDistance.csv")

# Setting aside holdout set (20% of 4739 ~ 948)
set.seed(134)
M <- 948 
in.test <- sample(nrow(college_dist), size = M)
test <- college_dist[in.test,]
train <- college_dist[-in.test,]

#### 1/ Prediction on test set ####

### Regression ###

# Lasso with log(distance) - all covariates included.

# set y and x matrix
y_train = train$education
train$distance <- (train$distance + 0.5)
train$distance <- log(train$distance)
x_train = train %>%
  dplyr::select(-"education") %>%
  data.matrix()

# find best lambda using cv
lambda = cv.glmnet(x_train, y_train, alpha = 1, seed = 1)
best_lambda = lambda$lambda.min

# fit model with best lambda and estimate CV error
model_reg_lasso = glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, standardize = TRUE)
reg_lasso_predictions = predict(model_reg_lasso, s = best_lambda, newx = x_train)

# estimate RMSE on train set
reg_lasso_RMSE = sqrt(mean((train$education - reg_lasso_predictions) ^ 2)) # 1.532

# estimate RMSE on test set
x_test <- test %>%
  dplyr::select(-"education") %>%
  data.matrix()
pred_test = predict(model_reg_lasso, s = best_lambda, newx = x_test)

reg_test_RMSE <- sqrt(mean((test$education - pred_test) ^ 2)) # 1.522

### Classification ###

# Penalized logistic regression - LASSO with log(distance)

train <- train %>%
  mutate(educ_bin = ifelse(education > 12, 1, 0))

test <- test %>%
  mutate(educ_bin = ifelse(education > 12, 1, 0))

x <- train %>%
  dplyr::select(-c("educ_bin", "education")) %>%
  data.matrix()
y <- train$educ_bin

lambdas <- 10^seq(2, -3, by = -.1)

# get optimal lambda
lasso = cv.glmnet(x, y, alpha = 1, type.measure = "class", family = "binomial", lambda = lambdas)
min.lambda = lasso$lambda.min

model_class_lasso = glmnet(x, y, alpha = 1, lambda = min.lambda, type.measure = "auc")
class_lasso_predictions = predict(model_class_lasso, s = min.lambda, newx = x)

# estimate RMSE on train set
class_lasso_RMSE <- sqrt(mean((train$educ_bin - class_lasso_predictions)^2)) # 0.437

# estimate RMSE on test set

x.test <- test %>%
  dplyr::select(-c("educ_bin", "education")) %>%
  data.matrix()

pred.test = predict(model_class_lasso, s = min.lambda, newx = x.test)

class_test_RMSE <- sqrt(mean((test$educ_bin - pred.test) ^ 2)) # 0.434

#### 2/ Inference -- proceeding with the best regression model ####

coef(model_reg_lasso) # shows that nothing is dropped so we include all covariates in OLS

# re-running OLS with log(distance) - already coded - and all covariates 

model <- lm(education ~ gender + ethnicity + score + fcollege + mcollege +
              home + urban + unemp + wage + distance + tuition + income + 
              region, data = train)
summary(model)

# running model on test data: 

model.test <- lm(education ~ gender + ethnicity + score + fcollege + mcollege +
              home + urban + unemp + wage + distance + tuition + income + 
              region, data = test)
summary(model.test)

# regression table
stargazer(model, model.test, title = "Results", align = TRUE) 

# Computing confidence intervals for each coefficient (of 5 coefficients) - train only, assumes normality.
confint(model, "score", level = 0.95) # score

confint(model, "distance", level = 0.95) # distance

confint(model, "tuition", level = 0.95) # tuition

confint(model, "mcollege", level = 0.95) # mcollege

confint(model, "gender", level = 0.95) # gender

#### 3/ Bootstrap ####

coef.boot = function(data, indices) {
  fm = lm(data = data[indices,], education ~ gender + ethnicity + score + fcollege + mcollege +
            home + urban + unemp + wage + distance + tuition + income + region)
  return(coef(fm))
}
boot.out = boot(train, coef.boot, 50000) # takes ~10 mins to run

# Computing bootstrap conf. intervals & plots for some coeffs
boot.ci(boot.out, conf = 0.95, type = "perc", index = 4) # on score
plot(boot.out, index = 4)

boot.ci(boot.out, conf = 0.95, type = "perc", index = 11) # on distance
plot(boot.out, index = 11)

boot.ci(boot.out, conf = 0.95, type = "perc", index = 12) # on tuition
plot(boot.out, index = 12)

boot.ci(boot.out, conf = 0.95, type = "perc", index = 6) # on mcollege
plot(boot.out, index = 6)

boot.ci(boot.out, conf = 0.95, type = "perc", index = 2) # on gender
plot(boot.out, index = 2)

# Visualize output from the bootstrap

Names = names(boot.out$t0)
SEs = sapply(data.frame(boot.out$t), sd)
Coefs = as.numeric(boot.out$t0)
zVals = Coefs / SEs
Pvals = 2*pnorm(-abs(zVals))

Formatted_Results = cbind(Names, Coefs, SEs, Pvals)
print(xtable(Formatted_Results, type = "latex"), file = "filename.tex")

#### 4/ Model with fewer covariates ####

model1 <- lm(education ~ score + fcollege + mcollege + unemp + wage + 
               distance + tuition + income, data = train)
summary(model1) # R2 = 0.2625

model1.test <- lm(education ~ score + fcollege + mcollege + unemp + wage + 
                    distance + tuition + income, data = test)

model2 <- lm(education ~ score + fcollege + mcollege + 
               distance + tuition + income, data = train)
summary(model2) # R2 = 0.2573

model2.test <- lm(education ~ score + fcollege + mcollege + 
                    distance + tuition + income, data = test)

model3 <- lm(education ~ score + fcollege + mcollege + unemp + wage + unemp:wage +
               distance + I(distance^2) + tuition + income + tuition:income, data = train)
summary(model3)

# regression table
stargazer(model1, model1.test, model2, model2.test, title = "Results", align = TRUE)

