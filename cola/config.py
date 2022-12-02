model_save_path = '/home/draj5/projects/data_pruning/experiments/cola/'         
try:                                                                            
  os.mkdir (model_save_path + 'models/')                                        
  os.mkdir (model_save_path + 'logs/')                                          
  os.mkdir (model_save_path + 'embeddings/')                                    
except:                                                                         
  pass                                                                          
                                                                                
label2int = {"unacceptable": 0, "acceptable": 1}                                
int2label = {0: "unacceptable", 1: "acceptable"}                                
model_checkpoint = 'roberta-base' 
