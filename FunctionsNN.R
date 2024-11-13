# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # [ToDo] Initialize intercepts as zeros
  b1 = rep(0, hidden_p) # initialising intercepts (biases) for hidden layer
  b2 = rep(0, K) # initialising biases for output layer, one for each class
  
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  set.seed(seed) # setting seed
  W1 = matrix(rnorm(p * hidden_p, mean = 0, sd = scale), p, hidden_p) # weight matrix for input to hidden layer
  W2 = matrix(rnorm(hidden_p * K, mean = 0, sd = scale), hidden_p, K) # weight matrix for hidden to output layer
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2)) # returns the initialised parameters
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  n = length(y)
  # [ToDo] Calculate loss when lambda = 0
  # loss = ...
  P = exp(scores) / rowSums(exp(scores)) # softmax probabilities
   
  # use + 1 to convert indexing
  # sums the log probabilities of the class for each sample plus the ridge penalty term, which is = 0
  loss = - sum(log(P[cbind(1:n, y + 1)])) / n # computes the softmax loss function
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  # error = ...
  predictions = max.col(P) - 1
  error = 100 * mean(predictions != y) # computes the proportion of misclassified samples
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  # grad = ...
  P[cbind(1:n, y + 1)] = P[cbind(1:n, y + 1)] - 1
  grad = P / n # gradient of softmax loss
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){
  n = nrow(X)
  # [To Do] Forward pass
  # From input to hidden 
  z1 = X %*% W1 + matrix(rep(b1, n), nrow = n, byrow = TRUE) # input layer to hidden layer
  # ReLU
  a1 = (abs(z1) + z1) / 2 # exactly the same as H[H < 0] = 0 but faster
  
  # From hidden to output scores
  scores = a1 %*% W2 + matrix(rep(b2, n), nrow = n, byrow = TRUE) # hidden layer to output layer
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  loss_info = loss_grad_scores(y, scores, K) # using the previous function written
  loss = loss_info$loss
  grad_scores = loss_info$grad
  error = loss_info$error
  
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 = t(a1) %*% grad_scores + lambda * W2 # regularisation for W2 with lambda to account for the ridge penalty
  db2 = colSums(grad_scores) + lambda * b2 # regularisation for b2 with lambda to account for the ridge penalty
  
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  da1 = grad_scores %*% t(W2) # gradient of loss with respect to the hidden layer
  dz1 = da1 * (z1 > 0) # derivative of ReLU
  dW1 = t(X) %*% dz1 + lambda * W1 # regularization for W1 with lambda to account for the ridge penalty
  db1 = colSums(dz1) + lambda * b1 # regularization for b1 with lambda to account for the ridge penalty
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = loss, error = error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  # [ToDo] Forward pass to get scores on validation data
  z1_val = Xval %*% W1 + matrix(rep(b1, nrow(Xval)), nrow = nrow(Xval), byrow = TRUE) # input layer to hidden layer
  a1_val = (abs(z1_val) + z1_val) / 2 # exactly the same as Hval[Hval < 0] = 0 but faster
  scores_val = a1_val %*% W2 + matrix(rep(b2, nrow(Xval)), nrow = nrow(Xval), byrow = TRUE) # hidden layer to output layer
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  P_scores_val = exp(scores_val) / rowSums(exp(scores_val))
  pred = max.col(P_scores_val) - 1
  error = 100 * mean(pred != yval) # computes the proportion of misclassified samples
  # return the error and this function evaluates the validation set error with only forward pass
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n / mbatch)

  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  init_list = initialize_bw(ncol(X), hidden_p, length(unique(y)), scale, seed)
  # initialising weights and biases
  # W1, b1, W2 and b2 have been initialised by the function initialize_bw
  W1 = init_list$W1
  b1 = init_list$b1
  W2 = init_list$W2
  b2 = init_list$b2
  
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (epoch in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    epoch_loss = 0 # initialise current loss
    epoch_error = 0 # initialise current error
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    for (batch in 1:nBatch){
      # define batch indices
      batch_idx = batchids[((batch - 1) * mbatch + 1):min(batch * mbatch, n)]
      X_batch = X[batch_idx, , drop = FALSE]
      y_batch = y[batch_idx]
      
      # performing one pass, with forward pass and backward pass
      pass_result = one_pass(X_batch, y_batch, length(unique(y)), W1, b1, W2, b2, lambda)
      loss = pass_result$loss
      error_batch = pass_result$error
      grads = pass_result$grads
      
      # update loss and error
      epoch_loss = epoch_loss + loss
      epoch_error = epoch_error + error_batch
      
      # updating weights and biases via gradient descent
      W1 = W1 - rate * grads$dW1
      b1 = b1 - rate * grads$db1
      W2 = W2 - rate * grads$dW2
      b2 = b2 - rate * grads$db2
    }
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
    error[epoch] = epoch_error / nBatch # computes average loss per batch and error for each epoch
    error_val[epoch] = evaluate_error(Xval, yval, W1, b1, W2, b2) # computes validation loss using forward pass only
    cat("Epoch:", epoch, "Train Error:", error[epoch], "Validation Error:", error_val[epoch], "\n")
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}
