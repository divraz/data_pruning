model_save_path = '/home/draj5/projects/data_pruning/experiments/qqp/'         
try:                                                                            
  os.mkdir (model_save_path + 'models/')                                        
  os.mkdir (model_save_path + 'logs/')                                          
  os.mkdir (model_save_path + 'embeddings/')                                    
except:                                                                         
  pass                                                                          
                                                                                
label2int = {"not_duplicate": 0, "duplicate": 1}                                
int2label = {0: "not_duplicate", 1: "duplicate"}                                
model_checkpoint = 'roberta-base' 
d_name = 'qqp'
