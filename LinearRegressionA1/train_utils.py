import random
import numpy as np

# This function returns a scalar representing the quality of input weight and bias, lower is better.
def loss_funct(weight, bias, true, x):
    return pow(true - predict(x, weight, bias), 2)

# Loss function that allows a pre-generated prediction as input.
def loss_funct_pred_provided(true, prediction):
    return pow((true - prediction), 2)

# Performs a prediction using supplied x, weight, and bias.
def predict(x, weight, bias):
    return weight*x + bias

# Initialized training and returns the minimized weight and bias.
def linear_train_init(training_data, learning_rate):

    # Initialize (somewhat randomly) the weight and bias.
    random.seed(123)
    weight_0 = random.uniform(0, np.mean(training_data[1], 0))
    bias_0 = random.uniform(0, np.amax(training_data[1], 0))
    i = 1
    loss_hist = []

    # Perform initial training using the random weight and bias.
    loss, bias_1, weight_1 = linear_train(training_data, weight_0, bias_0, learning_rate)
    loss_hist.append(loss)
    print("Loss - epoch " + "0" + ": " + str(loss_hist[0]))

    # This loop will continue to train until the loss is the same 4 times in a row.
    while True:
        loss, bias_1, weight_1 = linear_train(training_data, weight_1, bias_1, learning_rate)
        loss_hist.append(loss)
        print("Loss - epoch " + str(i) + ": " + str(loss_hist[i]))
        if i > 3:
            if loss_hist[i] == loss_hist[i-1] and loss_hist[i-1] == loss_hist[i-2]:
                break

        i += 1

    print()

    return loss_hist, bias_1, weight_1

# Utility function that performs the bulk of the training.
def linear_train(training_data, weight, bias, learning_rate):
    loss = 0
    bias_gradient = 0
    weight_gradient = 0

    # Train for each value in the training data (an epoch).
    for entry in training_data:
        x_value = entry[0]
        true = entry[1]

        # Compute the loss and gradient for each data point in the training set.
        loss += loss_funct(weight, bias, true, x_value)
        bias_grad_new, weight_grad_new = gradient_descent(weight, bias, true, x_value)
        bias_gradient += bias_grad_new
        weight_gradient += weight_grad_new

    # Adjust the weight and bias according to the gradient computed during training.
    new_bias = bias - learning_rate * bias_gradient
    new_weight = weight - learning_rate * weight_gradient
    return loss, new_bias, new_weight

# Returns the bias gradient and the weight gradient in this respective order.
def gradient_descent(weight, bias, true, x):
    return (2*(true - predict(x, weight, bias))*(-1)), (2*(true - predict(x, weight, bias))*(-x))

