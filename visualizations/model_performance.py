import matplotlib.pyplot as plt

def plot_accuracy(dt_acc,rf_acc):
    models = ['Decision Tree','Random Forest']
    accs = [dt_acc,rf_acc]

    plt.figure()
    plt.bar(models,accs)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.savefig("visualizations/accuracy_comparison.png")
    