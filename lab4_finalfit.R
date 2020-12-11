
library(tidyverse)
library(tidymodels)


full_train <- read_csv("data/train.csv") %>% 
  sample_frac(.01)

splt <- initial_split(full_train)
train <- training(splt)

train_cv <- vfold_cv(train)


knn_res <- readRDS("tune_fit_01.Rds")

rec <- recipe(classification ~ enrl_grd + lat + lon + econ_dsvntg, data = train)  %>%
  step_mutate(enrl_grd = as.factor(enrl_grd),
              classification = as.factor(classification)) %>% 
  step_unknown(enrl_grd, econ_dsvntg)  %>% 
  step_medianimpute(lat, lon) %>% 
  step_normalize(lat, lon) %>% 
  step_dummy(enrl_grd, econ_dsvntg)


knn_mod <- nearest_neighbor() %>%
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = tune(),
           dist_power = tune())

# Select best tuning parameters
knn_best <- knn_res %>%
  select_best(metric = "roc_auc")

# Finalize your model using the best tuning parameters
knn_mod_final <- knn_mod %>%
  finalize_model(knn_best) 

# Finalize your recipe using the best turning parameters
knn_rec_final <- rec %>% 
  finalize_recipe(rec)

saveRDS(knn_mod_final, "knn_mod_final.rds")
# Run your last fit on your initial data split
doParallel::registerDoParallel()

knn_final_res <- last_fit(
  knn_mod_final, 
  preprocessor = knn_rec_final, 
  split = splt)

#Collect metrics
knn_res <- knn_final_res %>% 
  collect_metrics()


write_csv(knn_res, "knn_final_res.csv")
