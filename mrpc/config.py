model_save_path = '/home/draj5/projects/data_pruning/experiments/mrpc/'         
try:                                                                            
  os.mkdir (model_save_path + 'models/')                                        
  os.mkdir (model_save_path + 'logs/')                                          
  os.mkdir (model_save_path + 'embeddings/')                                    
except:                                                                         
  pass                                                                          
                                                                                
label2int = {"not_equivalent": 0, "equivalent": 1}                                
int2label = {0: "not_equivalent", 1: "equivalent"}                                
model_checkpoint = 'roberta-base' 
d_name = 'mrpc'
