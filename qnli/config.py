model_save_path = '/home/draj5/projects/data_pruning/experiments/qnli/'         
try:                                                                            
  os.mkdir (model_save_path + 'models/')                                        
  os.mkdir (model_save_path + 'logs/')                                          
  os.mkdir (model_save_path + 'embeddings/')                                    
except:                                                                         
  pass                                                                          
                                                                                
label2int = {"not_entailment": 1, "entailment": 0}                                
int2label = {1: "not_entailment", 0: "entailment"}                                
model_checkpoint = 'roberta-base' 
d_name = 'qnli'
