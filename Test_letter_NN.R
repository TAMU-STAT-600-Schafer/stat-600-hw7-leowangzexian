# Load the data
# setwd("/Users/wangzexianleo/Desktop/PhD/STAT600/HW/stat-600-hw3-leowangzexian")

# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# setwd("/Users/wangzexianleo/Desktop/PhD/STAT600/HW/stat-600-hw7-leowangzexian")

# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

#  Apply LR (note that here lambda is not on the same scale as in NN due to scaling by training size)
out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o') # around 19.5 if keep training
plot(out$error_test, type = 'o') # around 25 if keep training


# Apply neural network training with default given parameters
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70)) # 6.33
lines(1:length(out2$error_val), out2$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 16.1

library(microbenchmark)
microbenchmark(
  NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                  rate = 0.1, mbatch = 50, nEpoch = 150,
                  hidden_p = 100, scale = 1e-3, seed = 12345),
  times = 10
)

# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials

# decrease lambda to 0.0000001
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.0000001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70)) # 4.50
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 14.87

# increase hidden_p to 1500
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.0000001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 1500, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70)) # 0.72
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 12.91

# increase nEpoch to 500
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.0000001,
                rate = 0.1, mbatch = 50, nEpoch = 500,
                hidden_p = 1500, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70)) # 0
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 12.28

# decrease rate to 0.093
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.0000001,
                rate = 0.093, mbatch = 50, nEpoch = 500,
                hidden_p = 1500, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70)) # 0
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 12.23

