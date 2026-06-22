import pandas as pd

from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_data
from train_model import train_random_forest
from evaluate_model import evaluate_model
from feature_importance import plot_feature_importance

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = train_random_forest(X_train, y_train)

evaluate_model(model, X_test, y_test)

plot_feature_importance(model, X)