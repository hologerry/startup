import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

from modules.utils import count_parameters, count_trainable_parameters

from modules.resblk import ResBlk


class Model(nn.Module):
    """Use channel wise attention between attr and segmentation"""
    def __init__(self, opts):
        super().__init__()

        self.mode = opts.mode

        self.multi_nodes = opts.multi_nodes
        self.local_rank = opts.local_rank
        self.style_dim = opts.style_dim

        if self.mode == 'train':
            self.lambda_ = opts.lambda_

        if self.multi_nodes:
            self.device = torch.device("cuda:{}".format(self.local_rank))
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.M = ResBlk(3, 64)

        self.to_device()

        if self.mode == 'train':
            self.optim = torch.optim.SGD(self.M.parameters(), lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
            self.scheduler = StepLR(self.optim, step_size=opts.decay_step, gamma=opts.decay_ratio)

    def summary_model(self):
        print("-------------- Generator --------------")
        print(self.M)
        all_params = count_parameters(self.M)
        trainable_params = count_trainable_parameters(self.M)
        print("Number of parameters in Localizer:", all_params, "trainable:", trainable_params)
        print("---------------------------------------")

    def to_device(self):
        self.M = self.M.to(self.device)

    def parallel(self):
        if not self.multi_nodes and torch.cuda.is_available():
            self.M = nn.DataParallel(self.M)

        if self.multi_nodes:
            self.M = nn.SyncBatchNorm.convert_sync_batchnorm(self.M)
            self.M = nn.parallel.DistributedDataParallel(self.M, device_ids=[self.local_rank], output_device=self.local_rank)

    def train(self):
        self.M.train()

    def eval(self):
        self.M.eval()

    def train_model(self, data_batch):
        self.input = data_batch['image'].to(self.device)
        self.label = data_batch['label'].to(self.device)

        self.optim.zero_grad()
        self.output = self.M(self.input)
        _loss = self.lambda_ * F.cross_entropy(self.output, self.label)

        loss_M = _loss

        loss_dict = {}
        loss_dict["M loss"] = loss_M.item()
        loss_dict["_loss"] = _loss.item()

        loss_M.backward(retain_graph=True)
        self.optim.step()

        return loss_dict

    def update_lr(self):
        self.scheduler.step()

    def save_training_image_sample(self, file_path):
        pass

    @torch.no_grad()
    def predict(self, data_batch, file_path):
        self.M.eval()
        pass

    def save_ckpt(self, file_path):
        print('Saving weights to', file_path, '...')
        state_dict = self.M.module.state_dict() if torch.cuda.is_available() else self.M.state_dict()
        torch.save({
            'state_dict': state_dict,
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, file_path)

    def load_ckpt(self, file_path):
        print('Loading saved weights from', file_path, '...')
        # map_location = {"cuda:0": "cuda:{}".format(self.local_rank)}
        states = torch.load(file_path, map_location=self.device)
        if 'state_dict' in states:
            self.M.load_state_dict(states['state_dict'])
        if 'optim' in states and self.mode == 'train':
            self.optim.load_state_dict(states['optim'])
        if 'scheduler' in states and self.mode == 'train':
            self.scheduler.load_state_dict(states['scheduler'])
