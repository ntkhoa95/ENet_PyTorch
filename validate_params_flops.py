import torch, time
from models.ENet import ENet
from ptflops import get_model_complexity_info


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enet = ENet(12)
enet = enet.to(device)

macs, params = get_model_complexity_info(enet, (3, 640, 360), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))