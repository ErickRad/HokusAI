import torch, gc

torch.cuda.empty_cache()
gc.collect()
torch.cuda.ipc_collect()