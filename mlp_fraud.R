set.seed(2025)

# Config
eta <- 0.01
q <- 16
epochs <- 500
patience <- 50

# Load
suffix <- "_34"

X <- as.matrix(read.csv(paste0('data/X_train', suffix, '.csv')))
y <- read.csv(paste0('data/y_train', suffix, '.csv'))$y
X_test <- as.matrix(read.csv('data/X_test.csv'))
y_test <- read.csv('data/y_test.csv')$y

n <- nrow(X)
d <- ncol(X)

cat(sprintf("Train: %d, Test: %d, Features: %d\n", n, nrow(X_test), d))

# Val split
val_idx <- sample(1:n, floor(0.15 * n))
X_val <- X[val_idx, ]; y_val <- y[val_idx]
X_tr <- X[-val_idx, ]; y_tr <- y[-val_idx]
n_tr <- nrow(X_tr)

# Weights initialization
W1 <- matrix(rnorm(d * q) * sqrt(2/d), nrow = d, ncol = q)
W2 <- matrix(rnorm(q) * sqrt(2/q), nrow = q, ncol = 1)

# Activation function
relu <- function(x) pmax(0, x)
sigmoid <- function(x) 1 / (1 + exp(-x))

# Training
best_W1 <- W1; best_W2 <- W2
best_loss <- Inf; wait <- 0
train_loss <- val_loss <- numeric()

for (ep in 1:epochs) {
  # dW2
  dW2 <- matrix(0, q, 1)
  for (i in 1:n_tr) {
    x <- matrix(X_tr[i,], ncol=1)
    h <- relu(crossprod(W1, x))
    p <- sigmoid(crossprod(W2, h))
    dW2 <- dW2 + (p - y_tr[i])[1] * h / n_tr
  }
  W2 <- W2 - eta * dW2
  
  # dW1
  dW1 <- matrix(0, d, q)
  for (i in 1:n_tr) {
    x <- matrix(X_tr[i,], ncol=1)
    h <- relu(crossprod(W1, x))
    p <- sigmoid(crossprod(W2, h))
    relu_d <- ifelse(h > 0, 1, 0)
    dW1 <- dW1 + (p - y_tr[i])[1] * kronecker(t(W2 * relu_d), x) / n_tr
  }
  W1 <- W1 - eta * dW1
  
  # Loss
  tr_pred <- apply(X_tr, 1, function(row) {
    h <- relu(crossprod(W1, matrix(row, ncol=1)))
    sigmoid(crossprod(W2, h))
  })
  tr_pred <- pmax(pmin(tr_pred, 1-1e-15), 1e-15)
  tr_l <- -mean(y_tr * log(tr_pred) + (1-y_tr) * log(1-tr_pred))
  
  va_pred <- apply(X_val, 1, function(row) {
    h <- relu(crossprod(W1, matrix(row, ncol=1)))
    sigmoid(crossprod(W2, h))
  })
  va_pred <- pmax(pmin(va_pred, 1-1e-15), 1e-15)
  va_l <- -mean(y_val * log(va_pred) + (1-y_val) * log(1-va_pred))
  
  train_loss <- c(train_loss, tr_l)
  val_loss <- c(val_loss, va_l)
  
  # Early stopping
  if (va_l < best_loss) {
    best_loss <- va_l; best_W1 <- W1; best_W2 <- W2; wait <- 0
  } else {
    wait <- wait + 1
  }
  
  if (ep %% 50 == 0) cat(sprintf("Ep %d: train=%.4f val=%.4f\n", ep, tr_l, va_l))
  if (wait >= patience) { cat(sprintf("Early stop at %d\n", ep)); break }
}

W1 <- best_W1; W2 <- best_W2

# Test predictions
probs <- apply(X_test, 1, function(row) {
  h <- relu(crossprod(W1, matrix(row, ncol=1)))
  sigmoid(crossprod(W2, h))
})
preds <- ifelse(probs >= 0.5, 1, 0)

# Metrics
TP <- as.numeric(sum(preds == 1 & y_test == 1))
TN <- as.numeric(sum(preds == 0 & y_test == 0))
FP <- as.numeric(sum(preds == 1 & y_test == 0))
FN <- as.numeric(sum(preds == 0 & y_test == 1))

cat(sprintf("Sensitivity: %f\n", TP / (TP + FN)))
cat(sprintf("Specificity: %f\n", TN / (TN + FP)))
cat(sprintf("Precision: %f\n", TP / (TP + FP)))
cat(sprintf("MCC: %f\n", (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))

# Save
write.csv(data.frame(prob = probs), paste0('results/mlp', suffix, '.csv'), row.names = FALSE)

# Plot
png(paste0('results/mlp_training', suffix, '.png'), width=800, height=400)
par(mfrow=c(1,2))
plot(train_loss, type='l', col='blue', main='Loss', xlab='Epoch', ylab='BCE')
lines(val_loss, col='red')
legend('topright', c('Train','Val'), col=c('blue','red'), lty=1)
hist(probs, breaks=50, main='Predictions', xlab='Probability')
abline(v=0.5, col='red', lty=2)
dev.off()
