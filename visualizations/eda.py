import matplotlib.pyplot as plt 
import seaborn as sns

def plot_eda(df):
    plt.figure()
    sns.histplot(df['age'],kde=True)
    plt.title("Age Distribution")
    plt.savefig("visualizations/age_distributions.png")

    plt.figure()
    sns.boxplot(x=df['smoker'],y=df['charges'])
    plt.title("Smoker vs Charges")
    plt.savefig("visualizations/smoker_charges.png")
   