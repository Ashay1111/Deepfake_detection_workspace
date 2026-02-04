import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms


st.set_page_config(page_title="Deepfake Image Check", page_icon="🧪", layout="centered")

BASE_DIR = Path(__file__).resolve().parent


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(name: str):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        input_size = 224
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
        input_size = 224
    else:
        raise ValueError("Unsupported model. Use resnet18 or efficientnet_b0")
    return model, input_size


def find_default_model():
    default_path = BASE_DIR / "artifacts" / "baseline_model.pth"
    if default_path.exists():
        return default_path
    candidates = sorted((BASE_DIR / "artifacts").glob("*.pth"))
    return candidates[-1] if candidates else None


@st.cache_resource
def load_model(model_path: str):
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt["model"]
    input_size = ckpt["input_size"]
    model, _ = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, input_size, ckpt.get("pos_class", "fake"), device


def get_transforms(input_size: int):
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


st.title("Deepfake Image Check")
st.write("Upload a face image and get a real vs fake prediction from your trained model.")

model_path = find_default_model()

with st.sidebar:
    st.header("Model")
    if model_path is None:
        st.warning("No model found in image_classifier/artifacts. Train the model first.")
        st.stop()
    model_path = st.text_input("Model path", value=str(model_path))
    confidence_threshold = st.slider("Fake threshold", 0.0, 1.0, 0.5, 0.01)

try:
    model, input_size, pos_class, device = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    tfms = get_transforms(input_size)
    x = tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).squeeze(1)
        prob_fake = torch.sigmoid(logits).item()

    label = "FAKE" if prob_fake >= confidence_threshold else "REAL"
    st.subheader(f"Prediction: {label}")
    st.write(f"Fake probability: {prob_fake:.3f}")
    st.write(f"Real probability: {1 - prob_fake:.3f}")

    st.caption(f"Positive class is treated as: {pos_class}. Threshold: {confidence_threshold:.2f}")
else:
    st.info("Upload a JPG/PNG image to get a prediction.")
