import torch
from flops_counter import get_model_complexity_info
from models import CC_Module

with torch.cuda.device(0):
  net = CC_Module(scale=2)
  macs, params = get_model_complexity_info(net, (3, 320, 240), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)

  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
