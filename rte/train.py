from __future__ import print_function
import os
import time
import ujson as json
import random
import collections
import numpy as np
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

def load_and_filter_dataset (dataset_name):
  dataset = load_dataset ('glue', dataset_name)
  dataset = dataset.rename_column ("idx", "indexes")

  return dataset

def tokenize_dataset (dataset, model_checkpoint = 'roberta-base'):
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  
  def tokenize_function (examples):
    return tokenizer (examples['sentence1'],
                      examples['sentence2'],
                   add_special_tokens = True,
                   max_length=128, 
                   padding = 'max_length', 
                   truncation = True)
    
  tokenized_datasets = dataset.map (tokenize_function,
                                  batched = True,
                                  remove_columns = ['sentence1', 'sentence2']
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

def train (model, optimizer, scheduler, data_loader, epoch, eval = False):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    stage = 'Eval' if eval else 'Train'
    total_loss = 0
    total_accuracy = 0
    total = 0
    true_labels = []
    predictions = []

    if eval == True:
      model.eval ()
    else:
      model.train()

    pbar = tqdm(data_loader)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if eval:
          with torch.no_grad():        
            result = model(**batch)[1]
        else:
          model.zero_grad()        
          result = model(**batch)[1]
          
        loss = result.loss
        logits = result.logits
        total_loss += loss.item()

        if eval == False:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()

          scheduler.step()
            
        logits = logits.detach().cpu().numpy()
        label_ids = batch['label'].to('cpu').numpy()
        (acc, pred, lbls) = flat_accuracy(logits, label_ids)
        total_accuracy += acc
        predictions += list(pred)
        true_labels += list(lbls)
        total += len (label_ids)
        pbar.set_description(f"[{stage}] Epoch: {epoch:3}, Total: {total:6}, Accuracy : {(total_accuracy / total):.4f}, Loss: {loss:.4f}")
    pbar.close ()
    print (f"[{stage}] Epoch: {epoch:3}, Total: {total:6}, Accuracy : {(total_accuracy / total):.4f}, Loss: {loss:.4f}")
    return [total, total_accuracy, total_loss, predictions, true_labels]

def model_training (model, epochs, tokenized_datasets, batch_size):
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to (device)

  train_loader = DataLoader (tokenized_datasets['train'],
                           sampler = RandomSampler(tokenized_datasets['train']),
                           batch_size = batch_size)
  val_loader = DataLoader (tokenized_datasets['validation'],
                          sampler = SequentialSampler(tokenized_datasets['validation']),
                          batch_size = batch_size)
  '''
  test_loader = DataLoader (tokenized_datasets['test'],
                          sampler = SequentialSampler(tokenized_datasets['test']),
                          batch_size = batch_size)
  '''

  optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
  total_steps = len(train_loader) * epochs

  # Create the learning rate scheduler.
  #scheduler = get_linear_schedule_with_warmup(optimizer, 
  #                                          num_warmup_steps = 50, # Default value in run_glue.py
  #                                          num_training_steps = total_steps)
  scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 50, # Default value in run_glue.py
                                            num_training_steps = total_steps,
                                            num_cycles = 2)
  
  seed_val = 42
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  training_stats = {'epoch': [], 'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []}

  for epoch_i in tqdm(range(0, epochs)):

      [total_train, total_train_accuracy, total_train_loss, predictions, true_labels] = train (model, optimizer, scheduler, train_loader, epoch_i, eval = False)
      [total_eval, total_eval_accuracy, total_eval_loss, predictions, true_labels] = train (model, optimizer, scheduler, val_loader, epoch_i, eval = True)
      if epoch_i == 0:
        torch.save (model, model_save_path + 'models/model_' + str (epoch_i))
      # Record all statistics from this epoch.
      training_stats['epoch'].append (epoch_i + 1)
      training_stats['train_loss'].append (total_train_loss / len (train_loader))
      training_stats['eval_loss'].append (total_eval_loss / len (val_loader))
      training_stats['train_acc'].append (total_train_accuracy / total_train)
      training_stats['eval_acc'].append (total_eval_accuracy / total_eval)
  
  print (json.dumps(training_stats))
  test_metrics = {}    
  #[test_metrics['total'], test_metrics['accuracy'], test_metrics['loss'], test_metrics['predictions'], test_metrics['true_labels']] = train (model, optimizer, scheduler, val_loader, 11, eval = True)
  #print (test_metrics)
  #[total_eval, total_eval_accuracy, total_eval_loss, predictions, true_labels] = train (model, optimizer, scheduler, val_loader, 11, eval = False)
  [test_metrics['total'], test_metrics['accuracy'], test_metrics['loss'], test_metrics['predictions'], test_metrics['true_labels']] = train (model, optimizer, scheduler, val_loader, 11, eval = True)
  print (test_metrics['total'], test_metrics['accuracy'], test_metrics['loss'])
  
  return (model, test_metrics)

dataset = load_and_filter_dataset (d_name)
(tokenizer, tokenized_dataset) = tokenize_dataset (dataset, model_checkpoint=model_checkpoint)
tokenizer.save_pretrained (model_save_path + 'models/')

model = CrossEncoderModel (model_checkpoint, 2)
(model, test_metrics) = model_training (model, 20, tokenized_dataset, 64)

torch.save (model, model_save_path + 'models/model')
