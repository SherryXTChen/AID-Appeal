import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self, parser):
        parser.add_argument('--name', type=str, required=True, help='name of experiment')
        parser.add_argument('--out_dir', type=str, default='ckpts', help='path to training results')

        # model
        parser.add_argument('--model_type', type=str, required=True, choices=['siamese', 'concate'], help='model type')
        parser.add_argument('--loss_type', type=str, required=True, choices=['pair', 'triplet'], help='loss type')
        
        # dataset
        parser.add_argument('--appeal_root_list', nargs='*', type=str, default=[], help='path to appealing images')
        parser.add_argument('--unappeal_root_list', nargs='*', type=str, default=[], help='path to unappealing images')
        parser.add_argument('--split_radio', type=float, default=0.8, help='train data radio')
        parser.add_argument('--image_size', type=int, default=256, help='image size')
        
        # test
        parser.add_argument('--num_samples', type=int, default=100, help='number of test samples per folder')
        parser.add_argument('--resume', type=str, default='last.ckpt', help='checkpoint to be resumed')
        parser.add_argument('--split', type=str, choices=['train', 'val'], default='val', help='data split to rank')

        # training
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--num_threads', type=int, default=4, help='number of threads')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--num_epochs', type=int, default=1, help='total number of training epochs')
        parser.add_argument('--gpus', type=int, default=1, help='total number of gpus')
    
        return parser

    def gather_options(self):
        parser = self.initialize(self.parser)
        opt = parser.parse_args()
        
        return opt

