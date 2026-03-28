import numpy as np
import torch
import matplotlib.pyplot as plt
@torch.no_grad()
def score_dataset(model, loader, device):
    model.eval()

    all_scores = []
    all_posteriors = []
    all_is_noisy = []
    all_y_true = []
    all_y_tilde = []
    all_indices = []

    for batch in loader:
        x = batch["x"].to(device)
        y_tilde = batch["y_noisy"].to(device)

        qy_logits = model.qy(x)
        qy_probs = torch.softmax(qy_logits, dim=1)

        prob_clean = qy_probs.gather(1, y_tilde.unsqueeze(1)).squeeze(1)
        score = 1.0 - prob_clean

        all_scores.append(score.cpu())
        all_posteriors.append(qy_probs.cpu())
        all_is_noisy.append(batch["is_noisy"].cpu())
        all_y_true.append(batch["y_true"].cpu())
        all_y_tilde.append(batch["y_noisy"].cpu())
        all_indices.append(batch["index"].cpu())

    return {
        "score": torch.cat(all_scores).numpy(),
        "posterior": torch.cat(all_posteriors).numpy(),
        "is_noisy": torch.cat(all_is_noisy).numpy(),
        "y_true": torch.cat(all_y_true).numpy(),
        "y_tilde": torch.cat(all_y_tilde).numpy(),
        "index": torch.cat(all_indices).numpy(),
    }

def show_top_suspicious(dataset, results, class_names, top_k=10):
    idx_sorted = np.argsort(-results["score"])[:top_k]

    plt.figure(figsize=(15, 6))
    for i, ridx in enumerate(idx_sorted):
        ds_idx = int(results["index"][ridx])
        sample = dataset[ds_idx]

        x = sample["x"].permute(1, 2, 0).numpy()
        y_true = int(results["y_true"][ridx])
        y_tilde = int(results["y_tilde"][ridx])
        is_noisy = int(results["is_noisy"][ridx])
        score = float(results["score"][ridx])

        posterior = results["posterior"][ridx]
        top3 = np.argsort(-posterior)[:3]
        top3_text = ", ".join([f"{class_names[c]}:{posterior[c]:.2f}" for c in top3])

        plt.subplot(2, 5, i + 1)
        plt.imshow(x)
        plt.axis("off")
        plt.title(
            f"score={score:.2f}\n"
            f"true={class_names[y_true]}\n"
            f"noisy={class_names[y_tilde]}\n"
            f"is_noisy={is_noisy}\n"
            f"{top3_text}",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()

def show_high_score_clean(dataset, results, class_names, top_k=10):
    mask = (results["is_noisy"] == 0)
    candidate_idx = np.where(mask)[0]
    sorted_idx = candidate_idx[np.argsort(-results["score"][candidate_idx])[:top_k]]

    plt.figure(figsize=(15, 6))
    for i, ridx in enumerate(sorted_idx):
        ds_idx = int(results["index"][ridx])
        sample = dataset[ds_idx]

        x = sample["x"].permute(1, 2, 0).numpy()
        y_true = int(results["y_true"][ridx])
        y_tilde = int(results["y_tilde"][ridx])
        score = float(results["score"][ridx])

        posterior = results["posterior"][ridx]
        top3 = np.argsort(-posterior)[:3]
        top3_text = ", ".join([f"{class_names[c]}:{posterior[c]:.2f}" for c in top3])

        plt.subplot(2, 5, i + 1)
        plt.imshow(x)
        plt.axis("off")
        plt.title(
            f"score={score:.2f}\n"
            f"true={class_names[y_true]}\n"
            f"noisy={class_names[y_tilde]}\n"
            f"{top3_text}",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()

def show_low_score_noisy(dataset, results, class_names, top_k=10):
    mask = (results["is_noisy"] == 1)
    candidate_idx = np.where(mask)[0]
    sorted_idx = candidate_idx[np.argsort(results["score"][candidate_idx])[:top_k]]

    plt.figure(figsize=(15, 6))
    for i, ridx in enumerate(sorted_idx):
        ds_idx = int(results["index"][ridx])
        sample = dataset[ds_idx]

        x = sample["x"].permute(1, 2, 0).numpy()
        y_true = int(results["y_true"][ridx])
        y_tilde = int(results["y_tilde"][ridx])
        score = float(results["score"][ridx])

        posterior = results["posterior"][ridx]
        top3 = np.argsort(-posterior)[:3]
        top3_text = ", ".join([f"{class_names[c]}:{posterior[c]:.2f}" for c in top3])

        plt.subplot(2, 5, i + 1)
        plt.imshow(x)
        plt.axis("off")
        plt.title(
            f"score={score:.2f}\n"
            f"true={class_names[y_true]}\n"
            f"noisy={class_names[y_tilde]}\n"
            f"{top3_text}",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()