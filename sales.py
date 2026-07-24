# Imports
import kagglehub
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Make plots look neater
plt.rc('font', size=10)
plt.rc('axes', labelsize=10, titlesize=10)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)


def load_visuelle():
    # Download latest version of Visuelle dataset
    path = kagglehub.dataset_download("konradb/visuelle-complete-dataset")

    return pd.read_csv(path + '/train.csv')

visuelle_full = load_visuelle()
    

# print(visuelle_full.info())

# visuelle_full.hist(bins='auto', figsize=(12, 8), color='pink')
# plt.show()


