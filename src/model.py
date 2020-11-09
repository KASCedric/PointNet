import torch
import torch.nn as nn


# PointNet semantic segmentation network
class PointNetSemSeg(nn.Module):
    def __init__(self, n_classes, bn):
        super(PointNetSemSeg, self).__init__()

        self.n_classes = n_classes  # n_classes: number of classes for the per-point classification task
        self.bn = bn  # bn = True <=> we use batch normalization

        self.local_feats_net = PointNetLocalFeatures(bn=bn)
        self.global_feats_net = PointNetGlobalFeatures(bn=bn)

        # Shared MLP(512, 256, 128, 128, n_classes) layers
        self.shared_mlp = nn.Sequential(
            # Conv1d: (in_channels, out_channels, kernel_size)
            # BatchNorm1d: (n_features)
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, n_classes, 1),
            nn.BatchNorm1d(n_classes) if bn else nn.Identity(),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        # point_net_sem_seg_input: (batch_size, n_channels, n_points)
        point_net_sem_seg_input = x                                            # (batch_size, 3, n_points)
        batch_size, n_channels, n_points = point_net_sem_seg_input.size()

        # Local features extractions
        local_features = self.local_feats_net(point_net_sem_seg_input)         # (batch_size, 64, n_points)

        # Global features extractions (from local features)
        global_features = self.global_feats_net(local_features)                # (batch_size, 1024)
        global_features = global_features.unsqueeze(dim=2)                     # (batch_size, 1024, 1)
        global_features = global_features.repeat(1, 1, n_points)               # (batch_size, 1024, n_points)

        # Local and global features concatenation
        all_features = torch.cat([local_features, global_features], dim=1)     # (batch_size, 1088, n_points)

        # Shared MLP(512, 256, 128, 128, n_classes) layers
        mlp_output = self.shared_mlp(all_features)                             # (batch_size, n_classes, n_points)

        point_net_sem_seg_output = mlp_output                                  # (batch_size, n_classes, n_points)

        return point_net_sem_seg_output


# PointNet classification network
class PointNetCls(nn.Module):
    def __init__(self, n_classes, bn, do):
        super(PointNetCls, self).__init__()

        self.n_classes = n_classes  # n_classes: number of classes for the classification task
        self.bn = bn  # bn = True <=> we use batch normalization
        self.do = do  # do = True <=> we use dropout during training

        self.local_feats_net = PointNetLocalFeatures(bn=bn)
        self.global_feats_net = PointNetGlobalFeatures(bn=bn)

        # Fully connected MLP((512, 256, n_classes) layers
        self.mlp = nn.Sequential(
            # Linear: (in_features, out_features)
            # BatchNorm1d: (n_features)
            # Note: We use LogSoftmax instead of Softmax for numerical properties and stability
            # /!\ LogSoftmax makes us use the NLLLoss instead of the CrossEntropyLoss. (See pytorch doc)
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3) if do else nn.Identity(),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # t_net_input_data: (batch_size, n_channels, n_points)
        point_net_cls_input = x                                            # (batch_size, 3, n_points)

        # Local features extractions
        local_features = self.local_feats_net(point_net_cls_input)         # (batch_size, 64, n_points)

        # Global features extractions (from local features)
        global_features = self.global_feats_net(local_features)            # (batch_size, 1024)

        # Fully connected MLP((512, 256, n_classes) layers
        mlp_output = self.mlp(global_features)                             # (batch_size, n_classes)

        point_net_cls_output = mlp_output                                  # (batch_size, n_classes)

        return point_net_cls_output


# Spatial Temporal Networks (STN) for the Input & Feature Transforms
class TNet(nn.Module):
    def __init__(self, k, bn):
        super(TNet, self).__init__()

        self.k = k  # TNet transform matrix's shape = (k, k)
        self.bn = bn  # bn = True <=> we use batch normalization

        # Shared MLP(64, 128, 1024) layers
        self.shared_mlp = nn.Sequential(
            # Conv1d: (in_channels, out_channels, kernel_size)
            # BatchNorm1d: (n_features)
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024) if bn else nn.Identity(),
            nn.ReLU(),
        )

        # Fully connected MLP((512, 256) layers
        self.mlp = nn.Sequential(
            # Linear: (in_features, out_features)
            # BatchNorm1d: (n_features)
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, k**2),
        )

    def forward(self, x):
        # t_net_input_data: (batch_size, n_channels, n_points)
        t_net_input_data = x                                               # (batch_size, 3, n_points)
        batch_size, n_channels, n_points = t_net_input_data.size()

        # Shared MLP(64, 128, 1024) layers
        shared_mlp_output = self.shared_mlp(t_net_input_data)              # (batch_size, 1024, n_points)

        # Max Pooling
        max_pooling = nn.MaxPool1d(n_points)(shared_mlp_output)            # (batch_size, 1024, 1)
        max_pooling_flat = nn.Flatten(1)(max_pooling)                      # (batch_size, 1024)

        # Fully connected MLP((512, 256, k**2) layers
        mlp_output = self.mlp(max_pooling_flat)                            # (batch_size, k**2)

        # Note: As TNet outputs a transform matrix, the output is initialized with an identity matrix so that there is
        # no transform applied to the input point cloud at initialization
        identity = torch.eye(self.k).repeat(batch_size, 1, 1)
        transform_matrix = mlp_output.view(-1, self.k, self.k) + identity  # (batch_size, k, k)

        return transform_matrix


# Local points Features extractor of PointNet
class PointNetLocalFeatures(nn.Module):
    def __init__(self, bn):
        super(PointNetLocalFeatures, self).__init__()

        self.bn = bn  # bn = True <=> we use batch normalization

        self.t_net_3d = TNet(k=3, bn=bn)
        self.t_net_64d = TNet(k=64, bn=bn)

        # Shared MLP(64, 64) layers
        self.shared_mlp = nn.Sequential(
            # Conv1d: (in_channels, out_channels, kernel_size)
            # BatchNorm1d: (n_features)
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64) if bn else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        # local_feats_input: (batch_size, n_channels, n_points)
        local_feats_input = x                                               # (batch_size, 3, n_points)

        # Input transform
        t_net_3d_matrix = self.t_net_3d(local_feats_input)                  # (batch_size, 3, 3)
        input_transform = torch.matmul(t_net_3d_matrix, local_feats_input)  # (batch_size, 3, n_points)

        # Shared MLP(64, 64) layers
        shared_mlp_output = self.shared_mlp(input_transform)                # (batch_size, 64, n_points)

        # Feature transform
        t_net_64d_matrix = self.t_net_64d(shared_mlp_output)                # (batch_size, 64, 64)
        feat_transform = torch.matmul(t_net_64d_matrix, shared_mlp_output)  # (batch_size, 64, n_points)

        local_features = feat_transform                                     # (batch_size, 64, n_points)
        return local_features


# Global points Features extractor of PointNet (from local features)
class PointNetGlobalFeatures(nn.Module):
    def __init__(self, bn):
        super(PointNetGlobalFeatures, self).__init__()

        self.bn = bn  # bn = True <=> we use batch normalization

        # Shared MLP(64, 128, 1024) layers
        self.shared_mlp = nn.Sequential(
            # Conv1d: (in_channels, out_channels, kernel_size)
            # BatchNorm1d: (n_features)
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024) if bn else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        # global_feats_input: (batch_size, n_channels, n_points)
        global_feats_input = x                                              # (batch_size, 64, n_points)
        batch_size, n_channels, n_points = global_feats_input.size()

        # Shared MLP(64, 128, 1024) layers
        shared_mlp_output = self.shared_mlp(global_feats_input)             # (batch_size, 64, n_points)

        # Max Pooling
        max_pooling = nn.MaxPool1d(n_points)(shared_mlp_output)             # (batch_size, 1024, 1)
        max_pooling_flat = nn.Flatten(1)(max_pooling)                       # (batch_size, 1024)

        global_features = max_pooling_flat                                  # (batch_size, 1024)
        return global_features


if __name__ == '__main__':

    batch_s, num_channels, num_points = (4, 64, 5)
    input_data = torch.randn(batch_s, num_channels, num_points)
    net = TNet(k=num_channels, bn=True)
    matrix = net(input_data)
    print("TNet matrix shape: ", matrix.size())

    batch_s, num_channels, num_points = (4, 3, 5)
    input_data = torch.randn(batch_s, num_channels, num_points)
    local_feats_net = PointNetLocalFeatures(bn=True)
    local_feats = local_feats_net(input_data)
    print("local_feats shape: ", local_feats.size())

    batch_s, num_channels, num_points = (4, 64, 5)
    input_data = torch.randn(batch_s, num_channels, num_points)
    global_feats_net = PointNetGlobalFeatures(bn=True)
    global_feats = global_feats_net(input_data)
    print("global_feats shape: ", global_feats.size())

    batch_s, num_channels, num_points = (4, 3, 5)
    input_data = torch.randn(batch_s, num_channels, num_points)
    point_net_cls = PointNetCls(n_classes=10, bn=True, do=True)
    point_net_cls_pred = point_net_cls(input_data)
    print("point_net_cls_pred shape: ", point_net_cls_pred.size())

    batch_s, num_channels, num_points = (4, 3, 5)
    input_data = torch.randn(batch_s, num_channels, num_points)
    point_net_sem_seg = PointNetSemSeg(n_classes=10, bn=True)
    point_net_sem_seg_pred = point_net_sem_seg(input_data)
    print("point_net_sem_seg_pred shape: ", point_net_sem_seg_pred.size())
