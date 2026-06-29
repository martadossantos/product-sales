# Imports
import kagglehub
import pandas as pd

# Download latest version of Visuelle dataset
path = kagglehub.dataset_download("konradb/visuelle-complete-dataset")

print("Path to dataset files:", path)

df_loaded = pd.read_csv(path + '/train.csv')

print(df_loaded.head(10))