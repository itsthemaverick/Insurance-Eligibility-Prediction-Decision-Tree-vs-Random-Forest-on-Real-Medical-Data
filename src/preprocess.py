from sklearn.model_selection import train_test_split

def split_features_target(df):
    X = df.drop(columns=['charges','eligible'])
    y = df['eligible']
    return X,y

def train_test(X,y):
    return train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )