# Imports
import kagglehub
import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

# # Download latest version of Visuelle dataset
# path = kagglehub.dataset_download("konradb/visuelle-complete-dataset")


# # Load data
# def load_visuelle():
#     return pd.read_csv(path + '/train.csv')

# df_raw = load_visuelle()
# df = df_raw.copy()


# Thresholds
# Group: 0–25      | flat | 0
# Group: 37.5–62.5 | slow | 1
# Group: 75–100    | fast | 2
quantiles = {
    "outer_low": 0.25,
    "inner_low": 0.375,
    "inner_high": 0.625,
    "outer_high": 0.75
}

# Target for evaluation
# Uses the buffer zones as part of the 'slow' range of products
evaluation_map = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2
}

# Target for training
# Narrower use of the thresholds
train_map = {
    0: 0,
    2: 1,
    4: 2
}

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


# Function to assign zones
def compute_zones(values, edges):
    return pd.cut(values, edges, labels=False)