from __future__ import print_function
import os
import time
import random
import pickle
import collections
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset
from datasets import Features, ClassLabel, Value

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoTokenizer, AutoConfig, AutoModel, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from config import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_and_filter_dataset (dataset_name):
  dataset = load_dataset ('glue', dataset_name)
  dataset = dataset.rename_column ("idx", "indexes")

  return dataset

def tokenize_dataset (dataset, model_checkpoint = 'roberta-base'):
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  
  def tokenize_function (examples):
    return tokenizer (examples['question'],
                      examples['sentence'],
                   add_special_tokens = True,
                   max_length=128, 
                   padding = 'max_length', 
                   truncation = True)
    
  tokenized_datasets = dataset.map (tokenize_function,
                                  batched = True,
                                  remove_columns = ['question', 'sentence']
                                  ).with_format("torch")
  return (tokenizer, tokenized_datasets)

class CrossEncoderModel (nn.Module):
  
  def __init__(self, checkpoint, num_labels): 
    super(CrossEncoderModel, self).__init__ () 

    self.num_labels = num_labels 

    # Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained (checkpoint,
                                            config = AutoConfig.from_pretrained(checkpoint, 
                                                                                output_attentions = True,
                                                                                output_hidden_states = True
                                                                                )
                                            )
    self.linear = nn.Linear (768, 768)
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(768 , num_labels)
    
  def forward(self,
              indexes = None,
              input_ids = None,
              attention_mask = None, 
              label = None):
    
    # Extract outputs from the body
    outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)[1]

    # Add custom layers
    context_vector = self.dropout (self.linear (outputs))

    # context_vector to be used as clustering
    logits = self.classifier (context_vector)

    loss = None
    if label is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
    
    return context_vector, TokenClassifierOutput(loss=loss, logits=logits)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat), pred_flat, labels_flat

def get_predictions (model, data_loader):
  prediction = []
  context_vector = []
  indexes = []
  total_accuracy = 0
  total = 0

  pbar = tqdm(data_loader)
  for batch in pbar:
      indexes = np.append (indexes, batch['indexes'].detach ().cpu ().numpy ())
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():        
        result = model(**batch)
      
      loss = result[1].loss
      logits = result[1].logits
      context_vector += list (result[0].detach ().cpu ().numpy ())
    
      logits = logits.detach().cpu().numpy()
      label_ids = batch['label'].to('cpu').numpy()
      (acc, pred, lbls) = flat_accuracy(logits, label_ids)
      total_accuracy += acc
      prediction += list(pred)
      total += len (label_ids)
      pbar.set_description(f"Total: {total:6}, Accuracy : {(total_accuracy / total):.4f}, Loss: {loss:.4f}")
  pbar.close ()
  return indexes, prediction, context_vector

dataset = load_and_filter_dataset (d_name)

def gen_emb (models, dataset):
  
  for model_name in models:
    df = pd.DataFrame ()
    d = {'t_label': []}
    
    for element in dataset['train']:
      for key, value in element.items ():
        if key not in d.keys ():
          d[ key ] = []
        d[ key ].append (value)
      d['t_label'].append (int2label[ element['label']])
    
    for key, value in d.items ():
      df[key] = value
    
    df.set_index('indexes', inplace=True)

    model = torch.load (model_save_path + 'models/' + model_name)
    model.to (device)

    (tokenizer, tokenized_dataset) = tokenize_dataset (dataset, model_checkpoint=model_save_path + 'models/')
    batch_size = 64
    train_loader = DataLoader (tokenized_dataset['train'],
                           batch_size = batch_size)
    indexes, prediction, context_vector = get_predictions (model, train_loader)

    df['context_vector'] = pd.Series(context_vector, index=indexes)
    df['prediction'] = pd.Series(prediction, index=indexes)

    df.to_pickle (model_save_path + 'embeddings/' + model_name + '.pkl')

models = ['model_0']
gen_emb (models, dataset)
print ('embeddings generated')
