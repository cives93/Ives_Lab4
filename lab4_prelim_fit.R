library(tidyverse)
library(tidymodels)

full_train <- read_csv("data/train.csv")

splt <- initial_split(full_train)
train <- training(splt)
train_cv <- vfold_cv(train)

# basic recipe
rec <- recipe(classification ~ enrl_grd + lat + lon + econ_dsvntg, data = train)  %>%
  step_mutate(enrl_grd = as.factor(enrl_grd),
              classification = as.factor(classification)) %>% 
  step_unknown(enrl_grd, econ_dsvntg)  %>% 
  step_medianimpute(lat, lon) %>% 
  step_normalize(lat, lon) %>% 
  step_dummy(enrl_grd, econ_dsvntg)

# lknn model
knn1_mod <- nearest_neighbor() %>% 
set_engine("kknn") %>% 
  set_mode("classification")

fit1 <- fit_resamples(knn1_mod, 
                      preprocessor = rec, 
                      resamples = train_cv)

saveRDS(fit1, "fit_prelim.Rds")