import pandas as pd
import numpy as np
import json
from config import *

def compute_1 (df, z):
  unique_values = df['score_1_9'].unique ()
  new_df = []
  data = {}
  new_data = {}

  for unq in unique_values:
    data[ unq ] = df[ df['score_1_9'] == unq ].shape[0]
    new_df.append (df[ df['score_1_9'] == unq ].sample (frac = z))
    new_data[ unq ] = new_df[ -1 ].shape[0]

  new_df = pd.concat (new_df)
  print (json.dumps (data))
  print (json.dumps (new_data))
  print (new_df)
  return new_df

def compute_2 (df, z, reverse):
  unique_values = sorted (list (df['score_1_9'].unique ()), reverse = reverse)
  total = int(df.shape[0] * z)
  print (total)
  new_df = []
  data = {}
  new_data = {}

  count = 0
  for unq in unique_values:
    data[ unq ] = df[ df['score_1_9'] == unq ].shape[0]
    if data[ unq ] + count <= total:
      new_df.append (df[ df['score_1_9'] == unq ])
      new_data[ unq ] = new_df[ -1 ].shape[0]
      count += new_data[ unq ]
    else:
      x = total - count
      if x > 0:
        new_df.append (df[ df['score_1_9'] == unq ].sample (x))
        new_data[ unq ] = new_df[ -1 ].shape[0]
        count += new_data[ unq ]

  new_df = pd.concat (new_df)
  print (json.dumps (data))
  print (json.dumps (new_data))
  print (count)
  return new_df

models = ['model_0']
models = [e + '.pkl' for e in models]

for model_name in models:
  try:
    print (model_name)
    df = pd.read_pickle ('embeddings/' + model_name)
    zs = [0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    for z in zs:
      print (z)
      new_df = compute_1 (df, z)
      new_df.to_pickle ('embeddings/pruned_1_' + str(z) + '_' + model_name)
      new_df = compute_2 (df, z, True)
      new_df.to_pickle ('embeddings/pruned_2_' + str(z) + '_' + model_name)
      new_df = compute_2 (df, z, False)
      new_df.to_pickle ('embeddings/pruned_3_' + str(z) + '_' + model_name)
      new_df = df.sample (frac = z, random_state = 1)
      new_df.to_pickle ('embeddings/random_' + str(z) + '_' + model_name)
  except Exception as e:
    print (e)
