import torch 

num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)