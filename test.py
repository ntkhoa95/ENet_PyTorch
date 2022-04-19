import os, sys, torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from models.ENet import ENet
import matplotlib.pyplot as plt


def test(opt):
    # Check if the pretrained model is available
    if not opt.m.endswith('.pth'):
        raise RuntimeError('Unknown file passed. Must end with .pth')
    if opt.image_path is None or not os.path.exists(opt.image_path):
        raise RuntimeError('An image file path must be passed')
    
    h = opt.resize_height
    w = opt.resize_width
    device =  opt.device

    checkpoint = torch.load(opt.m,  map_location=opt.device)
    
    # Assuming the dataset is camvid
    enet = ENet(opt.num_classes)
    enet = enet.to(device)
    enet.load_state_dict(checkpoint['state_dict'])

    img = cv2.imread(opt.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)

    img_tensor = torch.FloatTensor(img).unsqueeze(0).float()
    img_tensor = img_tensor.transpose(2, 3).transpose(1, 2)

    with torch.no_grad():
        out = enet(img_tensor.float().to(device)).squeeze(0)

    b_ = out.data.max(0)[1].cpu().numpy()

    # decoded_segmap = decode_custom_segmap(b_)
    decoded_segmap = decode_camvid_segmap(b_)

    images = {0:['input_image', img], 1:['semantic_prediction', decoded_segmap]}

    show_images(images)

# python init.py --mode test -m ./content/checkpoint/camvid/best_model.pth -i ./content/camvid/test/0001TP_008550.png
