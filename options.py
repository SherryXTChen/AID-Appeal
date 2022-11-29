import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self, parser):
        parser.add_argument('--name', type=str, required=True, help='name of experiment')
        parser.add_argument('--data_dir', type=str, required=True, help='path to data')
        parser.add_argument('--out_dir', type=str, default='ckpts', help='path to training results')
        parser.add_argument('--rank_dir', type=str, default='test', help='path to ranked results')

        parser.add_argument('--image_size', type=int, default=256, help='image size')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--num_threads', type=int, default=4, help='number of threads')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--num_epochs', type=int, default=30, help='total number of training epochs')
        parser.add_argument('--gpus', type=int, default=1, help='total number of gpus')
    
        return parser

    def gather_options(self):
        parser = self.initialize(self.parser)
        opt = parser.parse_args()
        
        return opt

