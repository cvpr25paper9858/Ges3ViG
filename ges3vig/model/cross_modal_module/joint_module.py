import lightning.pytorch as pl
import torch
import torch.nn as nn
from ges3vig.model.cross_modal_module.attention import MultiHeadAttention


class JointModule(pl.LightningModule):
    def __init__(self, feat_channel, input_channel, head, depth):
        super().__init__()
        self.depth = depth
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(
                d_model=feat_channel,
                h=head,
                d_k=feat_channel // head,
                d_v=feat_channel // head,
                dropout=0.1
            ) for _ in range(depth)
        )
    def forward(self, joint_features):
        for i in range(self.depth):
            aabb_features = self.self_attn[i](
                joint_features, joint_features, joint_features
            )
        return joint_features
