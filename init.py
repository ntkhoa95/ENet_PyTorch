import numpy as np
import argparse
from train import *
from test import *
import warnings
warnings.filterwarnings('ignore')

color_map = {
    'unlabeled'     : (  0,  0,  0),
    'dynamic'       : (111, 74,  0),
    'ground'        : ( 81,  0, 81),
    'road'          : (128, 64,128),
    'sidewalk'      : (244, 35,232),
    'parking'       : (250,170,160),
    'rail track'    : (230,150,140),
    'building'      : ( 70, 70, 70),
    'wall'          : (102,102,156),
    'fence'         : (190,153,153),
    'guard rail'    : (180,165,180),
    'bridge'        : (150,100,100),
    'tunnel'        : (150,120, 90),
    'pole'          : (153,153,153),
    'traffic light' : (250,170, 30),
    'traffic sign'  : (220,220,  0),
    'vegetation'    : (107,142, 35),
    'terrain'       : (152,251,152),
    'sky'           : ( 70,130,180),
    'person'        : (220, 20, 60),
    'rider'         : (255,  0,  0),
    'car'           : (  0,  0,142),
    'truck'         : (  0,  0, 70),
    'bus'           : (  0, 60,100),
    'caravan'       : (  0,  0, 90),
    'trailer'       : (  0,  0,110),
    'train'         : (  0, 80,100),
    'motorcycle'    : (  0,  0,230),
    'bicycle'       : (119, 11, 32)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        type=str,
                        default='./content/checkpoint/cam_vid/best_model.pth',
                        help='The path to the pretrained enet model')

    parser.add_argument('-i', '--image_path',
                        type=str,
                        help='The path to the image to perform semantic segmentation')

    parser.add_argument('-rh', '--resize_height',
                        type=int,
                        default=512,
                        help='The height for the resized image')

    parser.add_argument('-rw', '--resize_width',
                        type=int,
                        default=512,
                        help='The width for the resized image')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=5e-4,
                        help='The learning rate')
    
    parser.add_argument('-optim', '--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer method')

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        default=10,
                        help='The batch size')

    parser.add_argument('-wd', '--weight_decay',
                        type=float,
                        default=2e-4,
                        help='The weight decay')

    parser.add_argument('-c', '--constant',
                        type=float,
                        default=1.02,
                        help='The constant used for calculating the class weights')

    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=102,
                        help='The number of epochs')

    parser.add_argument('-nc', '--num_classes',
                        type=int,
                        default=12,
                        help='Number of unique classes')

    parser.add_argument('-se', '--save_every',
                        type=int,
                        default=5,
                        help='The number of epochs after which to save a model')

    parser.add_argument('-iptr', '--input_path_train',
                        type=str,
                        default='./content/camvid/train/',
                        help='The path to the input dataset')

    parser.add_argument('-lptr', '--label_path_train',
                        type=str,
                        default='./content/camvid/trainannot/',
                        help='The path to the label dataset')

    parser.add_argument('-ipv', '--input_path_val',
                        type=str,
                        default='./content/camvid/val/',
                        help='The path to the input dataset')

    parser.add_argument('-lpv', '--label_path_val',
                        type=str,
                        default='./content/camvid/valannot/',
                        help='The path to the label dataset')

    parser.add_argument('-iptt', '--input_path_test',
                        type=str,
                        default='./content/camvid/test/',
                        help='The path to the input dataset')

    parser.add_argument('-lptt', '--label_path_test',
                        type=str,
                        default='./content/camvid/testannot/',
                        help='The path to the label dataset')

    parser.add_argument('-pe', '--print_every',
                        type=int,
                        default=1,
                        help='The number of epochs after which to print the training loss')

    parser.add_argument('-ee', '--eval_every',
                        type=int,
                        default=1,
                        help='The number of epochs after which to print the validation loss')

    parser.add_argument('--device',
                        type=bool,
                        default=True,
                        help='Whether to use cuda or not')

    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='Whether to train or test')

    parser.add_argument('--save_model_path',
                        type=str,
                        default='./content/checkpoint/camvid',
                        help='The path to save model')
    
    opt, _ = parser.parse_known_args()

    opt.device = torch.device('cuda' if torch.cuda.is_available() and opt.device else 'cpu')

    if opt.mode.lower() == 'train':
        train(opt)
    elif opt.mode.lower() == 'test':
        test(opt)
    else:
        raise RuntimeError('Unknown mode passed. \n Mode passed should be either of "train" or "test"')