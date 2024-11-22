import torch
import torch.nn as nn
import lightning.pytorch as pl
torch.autograd.set_detect_anomaly(True)
class HumanProcessor(pl.LightningModule):
    def __init__(self, valid_sem_classes, in_channels=32, out_channels=128, num_joints = 55):
        super(HumanProcessor, self).__init__()
        self.human_label = valid_sem_classes.index(31)
        self.joint_regressor = JointRegressor([in_channels, 48], num_joints)
        self.feature_extracting_branch = FeatureExtractionBranch([in_channels, 64, out_channels])
        self.lr_classifier = nn.Sequential(
                nn.Linear(out_channels//4, 2)
            )
        self.dim_reducer = nn.Sequential(
                nn.Linear(in_channels+64+out_channels, out_channels//4)
                )
        self.bias_weigher = nn.Sequential(
                nn.Linear(out_channels//4, 2),
                nn.Softmax(dim = 1)
                )
        self.num_joints = num_joints
        self.in_channels = in_channels
    def forward(self, point_xyz, point_features, vert_batch_ids, semantic_scores, batch_size, scene_id=None):
        semantic_preds = semantic_scores.argmax(1).to(torch.int16)
        human_indices = torch.nonzero(semantic_preds == self.human_label)[:,0]
        human_features = point_features[human_indices]
        human_points = point_xyz[human_indices]
        human_batch_ids = vert_batch_ids[human_indices]

        joint_features = []
        joints = []
        has_human = torch.zeros(batch_size, device = self.device)
        for i in range(batch_size):
            batchwise_human_features = human_features[human_batch_ids == i]
            batch_human = human_points[human_batch_ids == i]
            if batchwise_human_features.shape[0] > 300:
                has_human[i] = True

                joint_weights = self.joint_regressor(batchwise_human_features)
                batch_joints = joint_weights @ batch_human
                batch_joints_features = joint_weights @ batchwise_human_features
            else:
                batch_joints = torch.zeros((self.num_joints, 3), device = self.device)
                batch_joints_features = torch.zeros((self.num_joints, self.in_channels), device = self.device)
            joint_features.append(batch_joints_features)
            joints.append(batch_joints)
        joints = torch.stack(joints)
        joint_features = torch.stack(joint_features)
        joint_features, agg_feats = self.feature_extracting_branch(joint_features)
        agg_feats = self.dim_reducer(agg_feats)
        lr = self.lr_classifier(agg_feats)
        bias_weights = self.bias_weigher(agg_feats)
        return joints, joint_features, lr, bias_weights, has_human


class FeatureExtractionBranch(pl.LightningModule):
    def __init__(self, layerwise_in_channel_config):
        super(FeatureExtractionBranch, self).__init__()
        for i in range(len(layerwise_in_channel_config)-1):
            self.register_module(f"human_feat_extractor_{i}",nn.Sequential(
                 nn.Conv1d(layerwise_in_channel_config[i],layerwise_in_channel_config[i+1],1),
                 nn.LeakyReLU(),
            ))
    def forward(self, human_features):
        output = human_features.permute(0,2,1)
        aggFeats = []
        aggFeats.append(nn.AvgPool1d(output.shape[2])(output))
        for name, layer in self.named_children():
            if "human_feat_extractor" in name:
                output = layer(output)
                aggFeats.append(nn.AvgPool1d(output.shape[2])(output))

        aggFeats = torch.cat(aggFeats, dim = 1)[:,:,0]
        output = output.permute(0,2,1)
        return output, aggFeats


class JointRegressor(pl.LightningModule):
    def __init__(self, layerwise_in_channel_config, num_joints):
        super(JointRegressor, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        for i in range(len(layerwise_in_channel_config)-1):
            self.register_module(f"human_feat_extractor_{i}",nn.Sequential(
                nn.Conv1d(layerwise_in_channel_config[i],layerwise_in_channel_config[i+1],1),
                nn.LeakyReLU(),
            ))

        self.joint_weigher = nn.Conv1d(layerwise_in_channel_config[-1], num_joints, 1)

    def forward(self, human_features):
        output = human_features.permute(1,0)
        for name, layer in self.named_children():
            if "human_feat_extractor" in name:
                output = layer(output)
        output = self.joint_weigher(output)
        output = self.softmax(output)
        return output
