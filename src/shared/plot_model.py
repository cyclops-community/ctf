import matplotlib.pyplot as plt
import sys

# Read data from local file
def load_data(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Read the model coeff from the file
    model_coeff = lines[0].strip().split()
    for i in range(len(model_coeff)):
        model_coeff[i] = float(model_coeff[i])
    # Read the job instance from the file
    instance = []
    for i in range(1, len(lines)):
        instance.append([float(entry) for entry in lines[i].strip().split()])
    return model_coeff, sorted(instance)

# Get the actual running time and estimited running time
def get_plot_data(model_coeff, instance):
    actual = []
    estimates = []
    for entry in instance:
        prediction = 0.0
        actual.append(entry[0])
        for i in range(len(model_coeff)):
            prediction += model_coeff[i] * entry[i+1]
        estimates.append(prediction)
    return actual, estimates


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage python3 plot_model.py file")
        sys.exit()
    file_name = sys.argv[1]
    model_coeff, instance = load_data(file_name)
    actual, estimates = get_plot_data(model_coeff, instance)
    plt.subplot(2, 1, 1)
    plt.plot(actual, 'b.', label='Actual')
    plt.plot(estimates, 'r.', label='Estimate')
    plt.legend(bbox_to_anchor=(0.75, 1), loc=2, borderaxespad=0.)

    total_under_time = 0.0
    total_over_time = 0.0
    difference = []
    for i in range(len(actual)):
        diff = estimates[i] - actual[i]
        difference.append(diff)
        if diff > 0:
            total_over_time += diff
        else:
            total_under_time -= diff
    plt.subplot(2, 1, 2)
    plt.plot(difference)
    plt.show()
    print("Total under time %f"%total_under_time)
    print("Total over time %f"%total_over_time)
