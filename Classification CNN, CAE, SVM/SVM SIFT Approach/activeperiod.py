import pickle
import pandas as pd
import load_data

df = pd.read_csv('project_data/artists.csv')

print(df[['name', 'years']])
