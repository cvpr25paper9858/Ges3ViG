import lightning.pytorch as pl
import torch
import torch.nn as nn

class HumanLosses(pl.LightningModule):
    def __init__(self, include_dir_loss = False):
        super().__init__()
        #self.include_dir_loss = include_dir_loss
        self.lr_criterion = nn.CrossEntropyLoss()
    def forward(self, data_dict, output_dict, loss_dict):
        diffs = output_dict["joints"] - data_dict["human_joint_matrix"]
        loss_dict["human_joint_loss"] = diffs.norm(dim = 2).sum(dim = 1).mean(dim = 0)
        loss_dict["lr_loss"] = self.lr_criterion(output_dict['human_lr'], data_dict["human_lr_label"])
