import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim: int):
        """
        Простая полносвязная сеть:
        - Linear(input_dim->64) + ReLU + Dropout
        - Linear(64->32)      + ReLU + Dropout
        - Linear(32->1)       + Sigmoid
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)