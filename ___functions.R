################ Important variables

# Function to get the Python environment path from the .txt file
get_python_env_path <- function(file_path = "python_env_path.txt") {
  if (!file.exists(file_path)) {
    stop("The file 'python_env_path.txt' is missing. Please create it and specify your Python environment path.")
  }
  
  # Read the file and extract the first non-comment, non-empty line
  env_path <- readLines(file_path, warn = FALSE)
  env_path <- env_path[!grepl("^#", env_path) & nzchar(env_path)]
  
  if (length(env_path) == 0) {
    stop("No valid Python environment path found in 'python_env_path.txt'. Please enter your path.")
  }
  
  return(trimws(env_path[1]))
}


################ SIM FUNCTION ---------------------

simulate_ex_art <- function(n = 1e6,
                            params,
                            copula = copula::normalCopula(param = params[['rho_xd']])){
  
  ## Independent vars
  ## Copula (X, D)
  u <- copula::rCopula(n, copula = copula)
  uX <- u[, 1]
  uD <- u[, 2]
  
  ## dependent (X, D)
  D <- qbinom(uD, size = 1, prob = params[['p']])
  X <- params[['g_0']] + params[['sd_X']] * qnorm(uX)
  ## portfolio selection
  a_0 <- -1 * sum(params[['a_x']]*params[['g_0']] + params[['a_d']] * params[['p']])
  eta_A <- a_0 + params[['a_x']] * X + params[['a_d']] * D
  pA <- exp(eta_A)/(1 + exp(eta_A))
  A <- rbinom(n, size = 1, prob = pA)
  
  ## Final elements
  Y <- rnorm(n, mean = params[['b_0']] + params[['b_x']] * X + params[['b_d']] * D,
             sd = params[['sd_Y']])
  
  ## Ok goodbye now
  to_return <- data.frame('D' = D, 'X' = X, 'Y' = Y,
                          'A' = A, 'pA' = pA, 'uX' = uX,
                          'uD' = uD)
  
  return(to_return)
}
  

##### GGPLOT2 VARIA ----------------------------------

toLabel <- function(x){
  label <- short_labels_for_parms[which(names(sims) == x)]
  if (length(label) == 0 || is.na(label)) {
    return(x)  # Retourne le nom original s'il n'y a pas de correspondance
  }
  latex2exp::TeX(label)
}

adjust_alpha <- function(color, alpha) {
  scales::alpha(color, alpha)
}


## Different colors per population, consistent all over the graphs
the_palette_fun <- function(pop_name){
  
  if(pop_name == 'inverse'){ # red, for the rho parameter negative
    to_return <- c(RColorBrewer::brewer.pal(12, "Paired")[5:6])
  } else if(pop_name == 'neutral'){ # neutral for the rho parameter zero
    to_return <- c(RColorBrewer::brewer.pal(11, "RdGy")[c(8, 10)])
  } else if(pop_name == 'original'){ # green for the positive rho parameter
    to_return <- c(RColorBrewer::brewer.pal(12, "Paired")[3:4])
  }
  
  to_return
}

the_palette_fun_ptf <- function(pop_name){
  
  if(pop_name %in% c('original', 'unbalA', 'noDtoY', 'noXtoA')){ # red, for the base scenario
    to_return <- c(RColorBrewer::brewer.pal(12, "Paired")[5:6])
  } else if(pop_name %in% c('noDtoA','noselbias', 'noDtoAstrong')){ # neutral for the rho parameter zero
    to_return <- c(RColorBrewer::brewer.pal(11, "RdGy")[c(8, 10)])
  }
  else if(pop_name %in% c('neutral','nodiscr')){ # purple if selection bias is less concern
    to_return <- c(RColorBrewer::brewer.pal(9, "Purples")[c(6, 9)])
  }
  
  to_return
}

## Parse latex in facet 
appender <- function(string) {
  if (length(string) > 1) {
    return(sapply(string, latex2exp::TeX))
  } else {
    return(latex2exp::TeX(string))
  }
}





# Wasserstein Distance
# (lower = similar)
wasserstein_distance <- list('stats' = function(name_col1, name_col2, data) {
  distance <- transport::wasserstein1d(data[[name_col1]], data[[name_col2]], p = 2)
  return(distance)  # Return the Wasserstein distance (lower = similar)
},
'eval' = function(mat){max(mat, na.rm = TRUE)})  

# Create a list structure similar to 'wilcox_test'
wasserstein_corr <- list(
  stats = 
adapW1_eot <- function(x, y) {
  N <- length(x)
  
  # Scale and floor the input vectors
  x_new <- floor(N^(1/3) * x) / N^(1/3)
  y_new <- floor(N^(1/3) * y) / N^(1/3)
  
  # Get unique values and frequencies of x_new
  x_freq <- table(x_new)
  x_val <- as.numeric(names(x_freq))
  
  # Vectorized Wasserstein calculation
  W <- sapply(x_val, function(val) {
    aux <- y_new[x_new == val]
    if (length(aux) == 0) return(0)
    # Calculate Wasserstein distance directly using 1D transport
    wasserstein1d(aux, y_new)
  })
  
  # Normalize and return the result
  denom <- mean(abs(outer(y_new, y_new, "-")))
  return(sum(W * as.numeric(x_freq)) / (N * denom))
},
  eval = function(mat) {
    max(mat, na.rm = TRUE)
  }
)


