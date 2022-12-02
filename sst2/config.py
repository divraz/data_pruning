model_save_path = '/home/draj5/projects/data_pruning/experiments/sst2/'         
try:                                                                            
  os.mkdir (model_save_path + 'models/')                                        
  os.mkdir (model_save_path + 'logs/')                                          
  os.mkdir (model_save_path + 'embeddings/')                                    
except:                                                                         
  pass                                                                          
                                                                                
label2int = {"positive": 1, "negative": 0}                                
int2label = {1: "positive", 0: "negative"}                                
model_checkpoint = 'roberta-base' 
d_name = 'sst2'
