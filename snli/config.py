model_save_path = '/home/draj5/projects/data_pruning/experiments/snli/'         
try:                                                                            
  os.mkdir (model_save_path + 'models/')                                        
  os.mkdir (model_save_path + 'logs/')                                          
  os.mkdir (model_save_path + 'embeddings/')                                    
except:                                                                         
  pass                                                                          

label2int = {"entailment": 0, "neutral": 1, "contradiction": 2}                 
int2label = {0: "entailment", 1: "neutral", 2: "contradiction"}                 
label2cos = {"contradiction": -1, "entailment": 1, "neutral": 0}                
model_checkpoint = 'roberta-base' 
                                                                                
d_name = 'snli'
num_classes = 3
