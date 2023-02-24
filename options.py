import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self, parser):
        parser.add_argument('--name', type=str, required=True, help='name of experiment')
        parser.add_argument('--out_dir', type=str, default='ckpts', help='path to training results')

        # model
        # parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
        parser.add_argument('--loss_type', type=str, required=True, choices=['singular', 'pair', 'triplet'], help='loss type')
        
        # dataset
        parser.add_argument('--root', type=str, required=True, help='path to images')
        parser.add_argument('--split_ratio', type=float, default=0.8, help='train data radio')
        parser.add_argument('--image_size', type=int, default=512, help='image size')
        
        # test
        # parser.add_argument('--resume', type=str, default='', help='ckeckpoint')
        # parser.add_argument('--split', type=str, default='val', help='data split to rank')
        # parser.add_argument('--num_samples', type=int, default=100, help='data split to rank')

        # training
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--num_threads', type=int, default=4, help='number of threads')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--num_epochs', type=int, default=1, help='total number of training epochs')
        parser.add_argument('--gpus', type=int, default=1, help='total number of gpus')
    
        return parser

    def gather_options(self):
        parser = self.initialize(self.parser)
        opt = parser.parse_args()
        
        return opt

