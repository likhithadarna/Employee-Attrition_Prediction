import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df.drop("EmployeeNumber", axis=1, inplace=True)

    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    return X, y