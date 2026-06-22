import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, X):

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })

    importance = importance.sort_values(
        by='Importance',
        ascending=False
    )

    print(importance.head(10))

    importance.head(10).plot(
        kind='barh',
        x='Feature',
        y='Importance'
    )

    plt.show()