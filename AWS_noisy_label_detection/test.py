import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from inference import load_model
from dataset import get_dataloaders
from evaluate import score_dataset

device = "cuda"
model = load_model("checkpoints/model.pt")

_, test_dataset, _, test_loader, _, _ = get_dataloaders(
    batch_size=128,
    noise_type="symmetric",
    noise_rate=0.2,
    root="./data",
    seed=42,
)

results = score_dataset(model, test_loader, device=device)

scores = results["score"]
top_idx = np.argsort(-scores)[:10]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for ax, idx in zip(axes, top_idx):
    sample = test_dataset[int(results["index"][idx])]
    img = sample["x"].permute(1, 2, 0).numpy()

    ax.imshow(img)
    ax.set_title(
        f"score={results['score'][idx]:.2f}\n"
        f"true={results['y_true'][idx]}, noisy={results['y_tilde'][idx]}"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig("top_suspicious.png", dpi=200, bbox_inches="tight")
print("saved to top_suspicious.png")