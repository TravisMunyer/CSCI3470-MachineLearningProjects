import numpy as np

def train_and_test(training_data, labels, testing_data):
    c2_sample_count = 0
    c1_sample_count = 0
    training_sample_total = training_data.shape[0]
    testing_sample_total = testing_data.shape[0]
    features_per_sample = testing_data.shape[1]
    c1_0_count = 0
    c1_1_count = 0
    c2_0_count = 0
    c2_1_count = 0

    # Compute prior probability for each class.
    for i in range(training_sample_total):
        if (labels[i] == 1):
            c2_sample_count += 1
            temp_ones_c2, temp_zeros_c2 = count_0s_and_1s(training_data[i])
            c2_1_count += temp_ones_c2
            c2_0_count += temp_zeros_c2
        elif (labels[i] == 0):
            c1_sample_count += 1
            temp_ones_c1, temp_zeros_c1 = count_0s_and_1s(training_data[i])
            c1_1_count += temp_ones_c1
            c1_0_count += temp_zeros_c1

    # Compute conditional probability for each class and feature possibility (this assumes all values are 0 or 1).
    p_of_1_given_c2 = c2_1_count / (c2_1_count + c2_0_count)
    p_of_0_given_c2 = c2_0_count / (c2_1_count + c2_0_count)
    p_of_1_given_c1 = c1_1_count / (c1_1_count + c1_0_count)
    p_of_0_given_c1 = c1_0_count / (c1_1_count + c1_0_count)

    # Compute prior probability for each class and feature possbility
    c2_probability = c2_sample_count / training_sample_total
    c1_probability = c1_sample_count / training_sample_total

    print()
    print("Training Results: ")
    print("P(C1): " + str(c1_probability) + ", P(C2): " + str(c2_probability))
    print("P(1|C1): " + str(p_of_1_given_c1) + ", P(0|C1): " + str(p_of_0_given_c1))
    print("P(1|C2): " + str(p_of_1_given_c2) + ", P(0|C2): " + str(p_of_0_given_c2))
    print()

    print("Testing Results: ")
    for j in range(testing_sample_total):
        # Initialized to 1 for multiplication
        sample_j_probability_x_given_c2 = 1
        sample_j_probability_x_given_c1 = 1

        for n in range(features_per_sample):
            # Multiply by respective '1' case conditional probabilities.
            if (testing_data[j][n] == 1):
                sample_j_probability_x_given_c2 *= p_of_1_given_c2
                sample_j_probability_x_given_c1 *= p_of_1_given_c1

            # Multiply by respective '0' case conditional probabilities.
            elif (testing_data[j][n] == 0):
                sample_j_probability_x_given_c2 *= p_of_0_given_c2
                sample_j_probability_x_given_c1 *= p_of_0_given_c1

        print("Testing results for sample: " + str(testing_data[j]))
        print("P(x|C1): " + str(sample_j_probability_x_given_c1) + ", P(x|C2): " + str(sample_j_probability_x_given_c2))

        # Compute probability for each class.
        p_c2_given_x = (sample_j_probability_x_given_c2 * c2_probability) \
                        / (sample_j_probability_x_given_c2 * c2_probability + sample_j_probability_x_given_c1 * c1_probability)
        p_c1_given_x = (sample_j_probability_x_given_c1 * c1_probability) \
                        / (sample_j_probability_x_given_c2 * c2_probability + sample_j_probability_x_given_c1 * c1_probability)

        print("P(C1|x): " + str(p_c1_given_x) + ", P(C2|x): " + str(p_c2_given_x))
        class_result = ''
        if (p_c1_given_x < p_c2_given_x):
            class_result = "Class 2."
        elif (p_c2_given_x < p_c1_given_x):
            class_result = "Class 1."
        else:
            class_result = "Class probabilities are equal, cannot make decision."

        print("This sample classification result: " + class_result)
        print()

# Returns the number of ones and zeros in the data sample respectively.
def count_0s_and_1s(data):
    zeros_count = 0
    ones_count = 0

    for i in range(data.shape[0]):
        if (data[i] == 1):
            ones_count += 1
        elif (data[i] == 0):
            zeros_count += 1

    return ones_count, zeros_count

print("Results on dataset defined by homework.")

# Training dataset defined by homework.
data = np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 1], [1, 0, 0]])

# For this training, we call 0's labels class 1 and 1's labels class 2.
#labels = np.array([1, 1, 1, 0, 0, 0])
labels = np.array([0, 0, 0, 1, 1, 1])

# Testing dataset defined by homework (with some extra for example).
testing = np.array([[0, 1, 1], [0, 0, 0], [1, 1, 1]])

# Train and test for the training and testing datasets defined by homework.
train_and_test(data, labels, testing)

print("Results on extra dataset to show performance on variable datasets.")

data_2 = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 0, 0], [1, 1, 0, 1, 1], [1, 1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]])
labels_2 = np.array([0, 0, 0, 0, 1, 1])
testing_2 = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])

train_and_test(data_2, labels_2, testing_2)


