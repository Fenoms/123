import tensorflow as tf 
from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(file_name = "model.ckpt-2000", tensor_name='', all_tensors=True, all_tensor_names= True)