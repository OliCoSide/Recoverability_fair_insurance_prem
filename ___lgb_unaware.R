the_unaware_lightgbm_fun <- function(name, selectA=0:1){
  train_data <- sims[[name]] 
  valid_data <- valid[[name]] 
  test_data <- test[[name]]
  
  #### FOR SELECTION INTEGRATION
  selection_label <- ifelse(length(selectA) == 2,
                            "market",
                            ifelse(selectA == 0,
                                   'A0',
                                   'A1'))
  
  trn <- train_data %>% dplyr::select(X)
  vld <- valid_data %>% dplyr::select(X)
  tst <- test_data %>% dplyr::select(X)
  
  ## convert categorical vars (if applicable)
  conversion <- lightgbm::lgb.convert_with_rules(data = trn, rules = NULL)
  
  ## extract data, and apply transformation to other datasets
  trn_conv <- conversion$data
  vld_conv <- lightgbm::lgb.convert_with_rules(data = vld, rules = conversion$rules)$data
  tst_conv <- lightgbm::lgb.convert_with_rules(data = tst, rules = conversion$rules)$data
  
  ## find the id of categorical vars
#  categ_id <- names(conversion$rules) %>% 
#    sapply(function(n){which(n == names(trn_conv))})
  
  ## transform into lgb.dataset (only training and valid)
  ## Attention, si on utilise l'exposition, on devrait l'inclure dans 'init_score' en incluant correctement la fonction de lien
  trn_lgb_dt <- lightgbm::lgb.Dataset(trn_conv[which(train_data$A%in% selectA),] %>% as.matrix,
                                      label = train_data$Y[which(train_data$A%in% selectA)],
                                      params=list(feature_pre_filter=FALSE))
  vld_lgb_dt <- lightgbm::lgb.Dataset(vld_conv[which(valid_data$A%in% selectA),] %>% as.matrix,
                                      label = valid_data$Y[which(valid_data$A%in% selectA)],
                                      params=list(feature_pre_filter=FALSE))
  
  ## setup training, and train. Insert optimization of hyperparameters here.
  paramGrid <- expand.grid(max_depth = c(1:5,8),
                           min_data_in_leaf = floor(c(0.001, 0.0005)* nrow(trn)))
  
  valid_mse <- numeric(nrow(paramGrid))
  n.trees <- numeric(nrow(paramGrid))
  
  for (i in 1:nrow(paramGrid)){ 
  hyperparameters <- list(objective = 'mse',
                          learning_rate = 0.01,
                          bagging_fraction = 0.75,
                          min_data_in_leaf = paramGrid[i,"min_data_in_leaf"],
                          max_depth = paramGrid[i,"max_depth"],
                          num_threads = parallel::detectCores(logical = FALSE))
  
  ## Train
  lgb_data_list <- list(valid = vld_lgb_dt) # A VÉRIFIER AVEC OLIVIER
  model <- lightgbm::lgb.train(params = hyperparameters,
                               data = trn_lgb_dt,
                               nrounds = 1e4,
                               valids = lgb_data_list,
                               early_stopping_rounds = 10,
                               verbose = -1)
  valid_mse[i] <- model$best_score 
  n.trees[i] <- model$best_iter
  }
  
  ## train the best model again
  ind_best_mod <- which.min(valid_mse)
  hyperparameters <- list(objective = 'mse',
                          learning_rate = 0.01,
                          bagging_fraction = 0.75,
                          min_data_in_leaf = paramGrid[ind_best_mod,"min_data_in_leaf"],
                          max_depth = paramGrid[ind_best_mod,"max_depth"],
                          num_threads = parallel::detectCores(logical = FALSE))
  
  model <- lightgbm::lgb.train(params = hyperparameters,
                               data = trn_lgb_dt,
                               nrounds = n.trees[ind_best_mod],
                               verbose = -1)
  
  
  ## predict
  trn_pred <- model$predict(trn_conv[which(train_data$A%in% selectA),] %>% as.matrix, rawscore = FALSE)
  vld_pred <- model$predict(vld_conv[which(valid_data$A%in% selectA),] %>% as.matrix, rawscore = FALSE)
  tst_pred <- model$predict(tst_conv[which(test_data$A%in% selectA),] %>% as.matrix, rawscore = FALSE)
  
  ## function to call the lgb later... counterfactual?
  predict_for_lgb <- function(newdata){
    conv_data <- lightgbm::lgb.convert_with_rules(newdata,
                                                  rules = conversion$rules)$data
    model$predict(conv_data %>% as.matrix, rawscore = FALSE)
  }
  
  preds <- list('train' = trn_pred,
                'valid' = vld_pred,
                'test' = tst_pred)
  
  ## If the folder do not exist... 
  if (!dir.exists('preds')) dir.create('preds')
  
  ## save them preds
  preds %>% jsonlite::toJSON(., pretty = TRUE) %>%
    write(paste0('preds/', name, '_', selection_label, '_unaware.json'))
  
  to_return <- list('preds' = preds,
                    'rules' = conversion$rules,
                    'pred_fun' = predict_for_lgb,
                    'par_mod' = c(paramGrid[ind_best_mod,], n.trees[ind_best_mod]))
}
