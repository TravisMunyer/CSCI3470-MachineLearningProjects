import train_utils
import matplotlib.pyplot as plt

# Computer validation loss.
def validate(validation_data, weight, bias):
    total_loss = 0

    for entry in validation_data:
        x_value = entry[0]
        true = entry[1]

        # Generate predictions and compute loss.
        prediction = train_utils.predict(x_value, weight, bias)
        loss = train_utils.loss_funct_pred_provided(true, prediction)
        total_loss += loss

        # Show the prediction and true values.
        print("Prediction: " + str(prediction))
        print("True:       " + str(true))
        print()

    print("Used weight: " + str(weight))
    print("Used bias: " + str(bias))
    print("Validation loss: " + str(loss))

# Plot the loss.
def plot_loss(loss):
    plt.figure().clear()
    plt.plot(loss)
    plt.title("Training Loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("loss.png")

# Plot regression line based on input weight and bias.
def plot_reg_line(x, y, label_x, label_y, plot_title, filename, weight, bias):
    plt.figure().clear()
    plt.plot(x, y, 'o')
    plt.plot(x, weight*x+bias)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(plot_title)
    plt.savefig(filename)