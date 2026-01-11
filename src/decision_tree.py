from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train,y_train):
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train,y_train)
    return model