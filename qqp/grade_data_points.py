import pandas as pd
import numpy as np
from config import *

arg_min_index = lambda x: np.argmin(x) if type(x) == np.ndarray else -1
arg_min = lambda x: x[np.argmin(x)] if type(x) == np.ndarray else -1
arg_min_del = lambda x: np.delete (x, np.argmin(x)) if type(x) == np.ndarray else -1

def compute (df):
  columns = list (df.columns)
  for column in columns:
    if 'cluster_' in column:
        index = column.replace ('cluster', '')
        df['ami' + index] = df[ column ].apply (arg_min_index)
        df['am' + index] = df[ column ].apply (arg_min)
        amd = df[ column ].apply (arg_min_del)
        df['ami2' + index] = amd.apply (arg_min_index)
        df['am2' + index] = amd.apply (arg_min)
        df['score_1' + index] = df['am2' + index] - df['am' + index]
        df['score_1' + index] = df['score_1' + index].round (2)

  #print (df.columns)
  df['avg'] = df[['am_' + str(i) for i in range (3, 10)]].mean(axis=1)
  df['avg'] = df['avg'].round (1)
  df['avg2'] = df[['am2_' + str(i) for i in range (3, 10)]].mean(axis=1)
  df['avg2'] = df['avg2'].round (1)
  return df

models = ['model_0']
models = [e + '.pkl' for e in models]

for model_name in models:
  try:
    #print (model_name)
    df = pd.read_pickle (model_save_path + 'embeddings/' + model_name)
    df = compute (df)
    df.to_pickle (model_save_path + 'embeddings/' + model_name)
  except Exception as e:
    print (e)
print ('data graded')
