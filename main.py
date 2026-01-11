from src.load_data import load_dataset
from src.preprocess import split_features_target, train_test
from src.decision_tree import train_decision_tree
from src.random_forest import train_random_forest
from src.evaluate import evaluate_model

from visualizations.eda import plot_eda
from visualizations.model_performance import plot_accuracy
from visualizations.feature_importance import plot_feature_importance


def main():
    df = load_dataset("data/insurance.csv")

    plot_eda(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test(X, y)

    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    dt_acc, dt_report, dt_cm = evaluate_model(dt_model, X_test, y_test)
    rf_acc, rf_report, rf_cm = evaluate_model(rf_model, X_test, y_test)

    print("Decision Tree Accuracy:", dt_acc)
    print(dt_report)

    print("Random Forest Accuracy:", rf_acc)
    print(rf_report)

    plot_accuracy(dt_acc, rf_acc)
    plot_feature_importance(rf_model, X.columns)


if __name__ == "__main__":
    main()
