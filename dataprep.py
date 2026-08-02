# Imports
import kagglehub
import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

# Download latest version of Visuelle dataset
path = kagglehub.dataset_download("konradb/visuelle-complete-dataset")


# Load data
def load_visuelle():
    return pd.read_csv(path + '/train.csv')

df_raw = load_visuelle()
df = df_raw.copy()

weeks = range(12)
week_cols = [f"W{w + 1:02d}" for w in weeks]

# Function to rename the weeks
def rename_weeks(df):

    df = df.rename(
        columns=dict(zip([str(w) for w in weeks], week_cols)),
        errors="raise"
    )

    return df


# Function to add sales of X number of weeks
def sum_weeks(df, target_weeks):
    number_of_weeks = len(target_weeks)

    df = df.copy()
    df[f"total_sales_{number_of_weeks}"] = df[target_weeks].sum(axis=1)

    return df
