import pandas as pd 

def load_dataset(path):
    df = pd.read_csv(path)

    df['sex'] = df['sex'].map({'male':1,'female':0})
    df['smoker'] = df['smoker'].map({'yes':1,'no':0})
    df['region'] = df['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})

    df['eligible'] = (df['charges']<df['charges'].median()).astype(int)

    return df