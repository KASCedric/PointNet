import torch
import torch.nn as nn


# Spatial Temporal Networks (STN) for the Input & Feature Transforms
class TNet(nn.Module):
    def __init__(self, k, bn):
        super(TNet, self).__init__()

        self.k = k  # TNet transform matrix's shape = (k, k)
        self.bn = bn  # bn = True <=> we use batch normalization

        # Shared MLP(64,128,1024) layers
        self.shared_mlp = nn.Sequential(
            # Conv1d: (in_channels, out_channels, kernel_size)
            # BatchNorm1d: (n_features)
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64) if bn else nn.Identity(),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if bn else nn.Identity(),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024) if bn else nn.Identity(),
            # nn.BatchNorm1d(1024),
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

        # Shared MLP(64,128,1024) layers
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


if __name__ == '__main__':
    B, N, K = (32, 1000, 64)
    input_data = torch.randn(B, K, N)
    net = TNet(k=K, bn=True)
    matrix = net(input_data)
