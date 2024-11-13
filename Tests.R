# This is a script to save your own tests for the function
source("FunctionsNN.R")

# first test case
Y = rbinom(100, size = 8, prob = 0.5) - 1
X = matrix(rnorm(100*99, 1, 0.99), 100)
Yt = rbinom(50, size = 8, prob = 0.5) - 1
Xt = matrix(rnorm(50*99, 1, 0.99), 50)
X = cbind(1, X)
Xt = cbind(1, Xt)
out2 = NN_train(X, Y, Xt, Yt, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error

# 2nd test case
Y = rbinom(200, size = 8, prob = 0.5) - 1
X = matrix(rnorm(200*99, 1, 0.99), 200)
Yt = rbinom(30, size = 8, prob = 0.5) - 1
Xt = matrix(rnorm(30*99, 1, 0.99), 30)
X = cbind(1, X)
Xt = cbind(1, Xt)
out2 = NN_train(X, Y, Xt, Yt, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error

# 3rd test case
Y = rbinom(200, size = 8, prob = 0.5) - 1
X = matrix(rnorm(200*99), 200)
Yt = rbinom(30, size = 8, prob = 0.5) - 1
Xt = matrix(rnorm(30*99), 30)
X = cbind(1, X)
Xt = cbind(1, Xt)
out2 = NN_train(X, Y, Xt, Yt, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error
