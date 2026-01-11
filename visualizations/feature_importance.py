import matplotlib.pyplot as plt 
import pandas as pd 

def plot_feature_importance(model,feature_names):

    importance = model.feature_importances_

    df = pd.DataFrame({
        'feature':feature_names,
        'importance':importance
    }).sort_values(by='importance')

    df.to_csv("data/feature_importance.csv")

    plt.figure()
    plt.barh(df['feature'],df['importance'])
    plt.title("Random Forest Feature Importance")
    plt.savefig("visualizations/feature_importance.png")