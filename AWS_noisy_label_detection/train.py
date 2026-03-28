import os
import math
import torch
import torch.nn.functional as F

from models import (
    DeepGenerativeNoiseModel,
    one_hot,
    gaussian_kl,
    bernoulli_log_prob_with_logits,
)
def compute_elbo(model, x, y_tilde, beta=0.3):
    """
    x: [B,3,32,32]
    y_tilde: [B]
    returns:
        loss, stats dict
    """
    B = x.size(0)
    K = model.num_classes
    device = x.device

    qy_logits = model.qy(x)
    qy_log_probs = F.log_softmax(qy_logits, dim=1)           # [B,K]
    qy_probs = torch.softmax(qy_logits, dim=1)               # [B,K]

    C = model.corruption()                                   # [K,K]
    log_C = torch.log(C + 1e-8)

    log_p_y = math.log(1.0 / K)

    expected_recon = torch.zeros(B, device=device)
    expected_kl_z = torch.zeros(B, device=device)
    expected_log_p_ytilde_given_y = torch.zeros(B, device=device)

    for k in range(K):
        yk = torch.full((B,), k, dtype=torch.long, device=device)
        yk_onehot = one_hot(yk, K).to(device)

        mu_k, logvar_k = model.qz(x, yk_onehot)
        z_k = model.sample_z(mu_k, logvar_k)
        x_logits_k = model.px(z_k, yk_onehot)

        log_p_x_given_z_y = bernoulli_log_prob_with_logits(x, x_logits_k)   # [B]
        kl_z_k = gaussian_kl(mu_k, logvar_k)                                # [B]

        log_p_ytilde_given_yk = log_C[k, y_tilde]                           # [B]

        qk = qy_probs[:, k]                                                 # [B]

        expected_recon += qk * log_p_x_given_z_y
        expected_kl_z += qk * kl_z_k
        expected_log_p_ytilde_given_y += qk * log_p_ytilde_given_yk

    entropy_qy = -torch.sum(qy_probs * qy_log_probs, dim=1)                 # [B]
    elbo = (
        expected_recon
        - beta * expected_kl_z
        + expected_log_p_ytilde_given_y
        + log_p_y
        + entropy_qy
    )

    loss = -elbo.mean()

    stats = {
        "elbo": elbo.mean().item(),
        "recon": expected_recon.mean().item(),
        "kl_z": expected_kl_z.mean().item(),
        "log_p_ytilde_given_y": expected_log_p_ytilde_given_y.mean().item(),
        "entropy_qy": entropy_qy.mean().item(),
    }
    return loss, stats, qy_probs

def train_one_epoch(model, loader, optimizer, device, beta=0.3, log_every=20):
    model.train()
    total_loss = 0.0
    total_stats = {
        "elbo": 0.0,
        "recon": 0.0,
        "kl_z": 0.0,
        "log_p_ytilde_given_y": 0.0,
        "entropy_qy": 0.0,
    }
    count = 0

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        y_tilde = batch["y_noisy"].to(device)

        optimizer.zero_grad()
        loss, stats, _ = compute_elbo(model, x, y_tilde, beta=beta)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        for k in total_stats:
            total_stats[k] += stats[k] * bs
        count += bs

        if step % log_every == 0:
            print(
                f"  step {step}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"elbo={stats['elbo']:.4f} | "
                f"recon={stats['recon']:.4f}"
            )

    avg_loss = total_loss / count
    avg_stats = {k: v / count for k, v in total_stats.items()}
    return avg_loss, avg_stats

@torch.no_grad()
def eval_one_epoch(model, loader, device, beta=0.3):
    model.eval()
    total_loss = 0.0
    total_stats = {
        "elbo": 0.0,
        "recon": 0.0,
        "kl_z": 0.0,
        "log_p_ytilde_given_y": 0.0,
        "entropy_qy": 0.0,
    }
    count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_tilde = batch["y_noisy"].to(device)

        loss, stats, _ = compute_elbo(model, x, y_tilde, beta=beta)
        bs = x.size(0)
        total_loss += loss.item() * bs
        for k in total_stats:
            total_stats[k] += stats[k] * bs
        count += bs

    avg_loss = total_loss / count
    avg_stats = {k: v / count for k, v in total_stats.items()}
    return avg_loss, avg_stats

def train_model(
    train_loader,
    val_loader,
    init_C,
    num_epochs=20,
    lr=1e-3,
    checkpoint_dir="checkpoints",
    checkpoint_name="model.pt",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = DeepGenerativeNoiseModel(
        num_classes=10,
        z_dim=64,
        init_C=init_C,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_stats = eval_one_epoch(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Train Stats: {train_stats}")
        print(f"  Val Stats:   {val_stats}")

        # 存最後一次
        last_ckpt_path = os.path.join(checkpoint_dir, "last_" + checkpoint_name)
        torch.save(model.state_dict(), last_ckpt_path)

        # 存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  Saved best model to {best_ckpt_path}")

    return model
from dataset import get_dataloaders

if __name__ == "__main__":
    train_dataset, test_dataset, train_loader, test_loader, C_train, C_test = get_dataloaders(
        batch_size=128,
        noise_type="symmetric",
        noise_rate=0.2,
        root="./data",
        seed=42,
    )

    model = train_model(
        train_loader=train_loader,
        val_loader=test_loader,
        init_C=C_train,
        num_epochs=10,
        lr=1e-3,
        checkpoint_dir="checkpoints",
        checkpoint_name="model.pt",
    )
