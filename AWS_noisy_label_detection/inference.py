import io
import torch
from PIL import Image
from torchvision import transforms

from config import DEVICE, NUM_CLASSES, Z_DIM, IMAGE_SIZE
from models import DeepGenerativeNoiseModel

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_model(checkpoint_path, init_C=None):
    model = DeepGenerativeNoiseModel(
        num_classes=NUM_CLASSES,
        z_dim=Z_DIM,
        init_C=init_C
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

@torch.no_grad()
def infer_posterior(model, x):
    logits = model.qy(x)
    probs = torch.softmax(logits, dim=1)
    return probs

from config import CLASS_NAMES
@torch.no_grad()
def predict_from_tensor(model, x, y_tilde):
    """
    x: torch.Tensor of shape [C,H,W] or [1,C,H,W]
    y_tilde: int
    """
    model.eval()

    if x.dim() == 3:
        x = x.unsqueeze(0)

    x = x.to(DEVICE)
    y_tilde_tensor = torch.tensor([y_tilde], device=DEVICE)

    qy_probs = infer_posterior(model, x)
    prob_observed = qy_probs.gather(1, y_tilde_tensor.unsqueeze(1)).squeeze(1)
    noise_score = 1.0 - prob_observed
    predicted_label = torch.argmax(qy_probs, dim=1)

    return {
        "noise_score": float(noise_score.item()),
        "prob_observed_label": float(prob_observed.item()),
        "observed_label": int(y_tilde),
        "observed_label_name": CLASS_NAMES[y_tilde],
        "predicted_label": int(predicted_label.item()),
        "predicted_label_name": CLASS_NAMES[predicted_label.item()],
        "posterior": qy_probs.squeeze(0).cpu().tolist(),
    }

def predict_from_pil(model, image: Image.Image, y_tilde: int):
    image = image.convert("RGB")
    x = transform(image)
    return predict_from_tensor(model, x, y_tilde)

def predict_from_bytes(model, image_bytes: bytes, y_tilde: int):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return predict_from_pil(model, image, y_tilde)