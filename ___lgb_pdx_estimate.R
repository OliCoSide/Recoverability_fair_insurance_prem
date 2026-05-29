the_pdx_lightgbm_fun <- function(name, selectA=0:1){
  train_data <- sims[[name]]
  valid_data <- valid[[name]]
  test_data <- test[[name]]
  
  trn <- train_data %>% dplyr::select(X)
  vld <- valid_data %>% dplyr::select(X)
  tst <- test_data %>% dplyr::select(X)
  
  ## transform into lgb.dataset (only training and valid)
  ## Attention, si on utilise l'exposition, on devrait l'inclure dans 'init_score' en incluant correctement la fonction de lien
  trn_lgb_dt <- lightgbm::lgb.Dataset(trn[which(train_data$A%in% selectA),] %>% as.matrix,
                                      label = train_data$D[which(train_data$A%in% selectA)])
  vld_lgb_dt <- lightgbm::lgb.Dataset(vld[which(valid_data$A%in% selectA),] %>% as.matrix,
                                      label = valid_data$D[which(valid_data$A%in% selectA)])
  
  ## setup training, and train. Insert optimization of hyperparameters here.
  paramGrid <- expand.grid(max_depth = c(1:5,8))
  
  valid_score <- numeric(nrow(paramGrid))
  n.trees <- numeric(nrow(paramGrid))
  
  for (i in 1:nrow(paramGrid)){
  hyperparameters <- list(objective = 'binary',
                          learning_rate = 0.03,
                          bagging_fraction = 0.75,
                          min_data_in_leaf = 32,
                          max_depth = paramGrid[i,"max_depth"],
                          lambda_l1 = 3,  # L1 regularization
                          lambda_l2 = 3,  # L2 regularization
                          num_threads = parallel::detectCores(logical = FALSE))
  
  ## Train
  lgb_data_list <- list(valid = vld_lgb_dt) # A VERIFIER AVEC OLIVIER
  model <- lightgbm::lgb.train(params = hyperparameters,
                               data = trn_lgb_dt,
                               nrounds = 1e4,
                               valids = lgb_data_list,
                               early_stopping_rounds = 10,
                               verbose = -1)
  
  valid_score[i] <- model$best_score 
  n.trees[i] <- model$best_iter
  }
  
  ## train the best model again
  ind_best_mod <- which.min(valid_score)
  hyperparameters <- list(objective = 'binary',
                          learning_rate = 0.03,
                          bagging_fraction = 0.75,
                          min_data_in_leaf = 32,
                          max_depth = paramGrid[ind_best_mod,"max_depth"],
                          lambda_l1 = 3,  # L1 regularization
                          lambda_l2 = 3,  # L2 regularization
                          num_threads = parallel::detectCores(logical = FALSE))
  
  model <- lightgbm::lgb.train(params = hyperparameters,
                               data = trn_lgb_dt,
                               nrounds = n.trees[ind_best_mod],
                               verbose = -1)
  
  ## predict - CAREFUL, NO FILTER ON SELECTA HERE (WHY?)
  trn_pred <- model$predict(trn %>% as.matrix, rawscore = FALSE)
  vld_pred <- model$predict(vld %>% as.matrix, rawscore = FALSE)
  tst_pred <- model$predict(tst %>% as.matrix, rawscore = FALSE)
  
  ## Predict prob D = d
  predict_for_lgb <- function(newdata){
    pred <- model$predict(newdata %>% dplyr::select(X) %>% as.matrix, rawscore = FALSE)
    1:nrow(newdata) %>% sapply(function(i){
      d <- newdata$D[i]
      ifelse(d == 1, pred[i], 1 - pred[i])
    })
  }
  
  preds <- list('train' = trn_pred,
                'valid' = vld_pred,
                'test' = tst_pred)
  
  ## If the folder do not exist... 
  if (!dir.exists('preds')) dir.create('preds')
  
  ## clean, then save them preds
  preds %>% jsonlite::toJSON(., pretty = TRUE) %>% write(paste0('preds/', name, '_pdx.json'))
  
  to_return <- list('preds' = preds,
                    'pred_fun' = predict_for_lgb,
                    'par_mod' = c(paramGrid[ind_best_mod,], 
                                  n.trees[ind_best_mod]))
}
