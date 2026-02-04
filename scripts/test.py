import torch
import gc

# # 1. Delete unnecessary variables
# del model 
# del optimizer 
# ... delete other large tensors

# 2. Invoke the garbage collector
gc.collect()

# 3. Empty the PyTorch CUDA cache
torch.cuda.empty_cache()