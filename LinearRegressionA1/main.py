# Assignment One for UNO CSCI 3470 Machine Learning by Travis Munyer.
# This code is for the linear regression portion of assignment one.
# For a description of this assignment, see Assignment_1.pdf.

import numpy as np
import pandas as pd
import train_utils
import evaluation_utils
from sklearn.model_selection import train_test_split

data = pd.read_csv('AssignmentOneData.csv').to_numpy(dtype='float32')
test_size = 0.2
learning_rate = 0.0001
np.set_printoptions(suppress=True)
print(data)

# Split the dataset into train and validate, then train the model.
train, validate = train_test_split(data, test_size=test_size, random_state=30, shuffle=True)
loss, final_bias, final_weight = train_utils.linear_train_init(train, learning_rate)

# Evaluate the performance of the model on the validation dataset.
evaluation_utils.validate(validate, final_weight, final_bias)

# Plot loss and the regression line.
evaluation_utils.plot_loss(loss)
evaluation_utils.plot_reg_line(data.transpose()[0], data.transpose()[1], "Years of Experience", "Salary", "Data Plot", "data.png", final_weight, final_bias)