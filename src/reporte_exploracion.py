# ==========================================================
#  reporte_exploracion.py
#  Football Prediction
#  Author: Manuel Avilés Rodríguez
# ==========================================================

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Global variable for directory
base_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    path = os.path.join(base_dir, '..', 'data', 'processed','laliga_features.csv')
    path = os.path.normpath(path)
    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    df['result_string'].value_counts().plot(kind='bar', ax=axes[0,0], title='Result String Distribution')
    axes[0,0].set_ylabel('Count')

    df['result_abstract'].value_counts().plot(kind='bar', ax=axes[0,1], title='Abstract Result Distribution')
    axes[0,1].set_ylabel('Count')

    df.corr(numeric_only=True)['FTHG'].sort_values(ascending=False).head(15).plot(kind='bar', ax=axes[1,0], title='Top Correlations with FTHG')

    df.corr(numeric_only=True)['FTAG'].sort_values(ascending=False).head(15).plot(kind='bar', ax=axes[1,1], title='Top Correlations with FTAG')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(f"SHAPE DATASET: \n{df.shape}\n")
    print(f"DESCRIBE DATASET: \n{df.describe().T}\n")
    print(f"TOTAL NULOS: \n{df.isnull().sum()}")

    plt.tight_layout()
    plt.show()
