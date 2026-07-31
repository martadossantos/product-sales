# Imports
import kagglehub
import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

# Make plots look neater
plt.rc('font', size=10)
plt.rc('axes', labelsize=10, titlesize=10)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)


# Download latest version of Visuelle dataset
path = kagglehub.dataset_download("konradb/visuelle-complete-dataset")

def load_visuelle():
    return pd.read_csv(path + '/train.csv')

df_raw = load_visuelle()
df = df_raw.copy()

