import numpy as np
import torch
from torch.utils.data import Dataset
def make_symmetric_C(K, eta):
    C = np.full((K, K), eta / (K - 1), dtype=np.float32)
    np.fill_diagonal(C, 1.0 - eta)
    return C
def make_asymmetric_cifar10_C(eta):
    C = np.eye(10, dtype=np.float32)
    mapping = {
        9: 1,  # truck -> automobile
        2: 0,  # bird -> airplane
        3: 5,  # cat -> dog
        5: 3,  # dog -> cat
        4: 7,  # deer -> horse
    }
    for src, dst in mapping.items():
        C[src, src] = 1.0 - eta
        C[src, dst] = eta
    return C
def corrupt_labels(y_true, C, seed=42):
    rng = np.random.default_rng(seed)
    y_noisy = []
    is_noisy = []
    for y in y_true:
        new_y = rng.choice(len(C), p=C[y])
        y_noisy.append(new_y)
        is_noisy.append(int(new_y != y))
    return np.array(y_noisy), np.array(is_noisy)


class NoisyCIFAR10(Dataset):
    def __init__(self, base_dataset, C, seed=42):
        self.base_dataset = base_dataset
        self.x_data = []
        self.y_true = []
        for x, y in base_dataset:
            self.x_data.append(x)
            self.y_true.append(y)
        self.y_true = np.array(self.y_true)
        self.y_noisy, self.is_noisy = corrupt_labels(self.y_true, C, seed=seed)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        return {
            "x": x,
            "index": idx,
            "y_true": int(self.y_true[idx]),
            "y_noisy": int(self.y_noisy[idx]),
            "is_noisy": int(self.is_noisy[idx]),
        }
    
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(
    batch_size=128,
    noise_type="symmetric",
    noise_rate=0.2,
    root="./data",
    seed=42,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_base = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test_base = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    if noise_type == "symmetric":
        C_train = make_symmetric_C(10, noise_rate)
        C_test = make_symmetric_C(10, noise_rate)
    elif noise_type == "asymmetric":
        C_train = make_asymmetric_cifar10_C(noise_rate)
        C_test = make_asymmetric_cifar10_C(noise_rate)
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    train_dataset = NoisyCIFAR10(train_base, C_train, seed=seed)
    test_dataset = NoisyCIFAR10(test_base, C_test, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader, C_train, C_test