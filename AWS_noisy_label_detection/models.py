import torch
import torch.nn as nn
import torch.nn.functional as F
class QyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        h = self.conv(x)
        logits = self.fc(h)
        return logits


class QzNet(nn.Module):
    def __init__(self, num_classes=10, z_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4 + num_classes, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

    def forward(self, x, y_onehot):
        h = self.conv(x)
        h = torch.cat([h, y_onehot], dim=1)
        h = self.fc(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class PxNet(nn.Module):
    def __init__(self, num_classes=10, z_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32x32
        )

    def forward(self, z, y_onehot):
        h = torch.cat([z, y_onehot], dim=1)
        h = self.fc(h)
        logits = self.deconv(h)
        return logits


class CorruptionModel(nn.Module):
    def __init__(self, num_classes=10, init_C=None):
        super().__init__()
        if init_C is None:
            init_logits = torch.eye(num_classes)
        else:
            init_logits = torch.log(torch.tensor(init_C) + 1e-6)
        self.logits = nn.Parameter(init_logits)

    def forward(self):
        # row-stochastic: each row is p(y_tilde | y)
        return F.softmax(self.logits, dim=1)


class DeepGenerativeNoiseModel(nn.Module):
    def __init__(self, num_classes=10, z_dim=64, init_C=None):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.qy = QyNet(num_classes)
        self.qz = QzNet(num_classes, z_dim)
        self.px = PxNet(num_classes, z_dim)
        self.corruption = CorruptionModel(num_classes, init_C=init_C)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

def gaussian_kl(mu, logvar):
    # KL(q(z|x)||N(0,I))
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)

def bernoulli_log_prob_with_logits(x, logits):
    # return log p(x|z,y)
    # negative BCE summed over pixels
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return -bce.flatten(start_dim=1).sum(dim=1)

def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()
