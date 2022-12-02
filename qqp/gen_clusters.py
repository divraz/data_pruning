from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from config import *

def get_transforms (features, n_clusters):
  kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=25, random_state=0)
  kmeans.fit (features, [])
  return kmeans.transform (features)

models = ['model_0']
models = [e + '.pkl' for e in models]

for model_name in models:
  #print (model_name)
  df = pd.read_pickle (model_save_path + 'embeddings/' + model_name)
  features = np.asarray ([element for element in df["context_vector"].to_numpy()])
  #print (features.shape)
  features = features.astype ('double')

  for n in range (2, 10):
    #print (n)
    d = get_transforms (features, n)
    #print (d[0], d.shape)
    d = [np.asarray (element) for element in d]
    df['cluster_' + str (n)] = pd.Series(d)
  df.to_pickle (model_save_path + 'embeddings/' + model_name)
print ('clusters generated')
