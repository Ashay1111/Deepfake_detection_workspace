import io
import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
import cv2


st.set_page_config(page_title="Video Deepfake Check", page_icon="🎬", layout="centered")

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_CLASSIFIER_DIR = BASE_DIR.parent / "image_classifier"


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
    default_path = IMAGE_CLASSIFIER_DIR / "artifacts" / "baseline_model.pth"
    if default_path.exists():
        return default_path
    candidates = sorted((IMAGE_CLASSIFIER_DIR / "artifacts").glob("*.pth"))
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


def sample_frames(video_path: Path, every_n: int, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


st.title("Video Deepfake Check")
st.write("Upload a video and get a video-level real vs fake prediction using frame sampling.")

model_path = find_default_model()

with st.sidebar:
    st.header("Model")
    if model_path is None:
        st.warning("No model found in image_classifier/artifacts. Train the image model first.")
        st.stop()
    model_path = st.text_input("Model path", value=str(model_path))
    confidence_threshold = st.slider("Fake threshold", 0.0, 1.0, 0.5, 0.01)
    every_n = st.slider("Sample every Nth frame", 1, 20, 5, 1)
    max_frames = st.slider("Max frames", 4, 64, 16, 1)

try:
    model, input_size, pos_class, device = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded is not None:
    st.info("Processing video... this may take a moment.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    frames = sample_frames(tmp_path, every_n=every_n, max_frames=max_frames)
    if not frames:
        st.error("Could not read frames from this video.")
        st.stop()

    tfms = get_transforms(input_size)
    probs = []

    with torch.no_grad():
        for frame in frames:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = tfms(img).unsqueeze(0).to(device)
            logits = model(x).squeeze(1)
            prob_fake = torch.sigmoid(logits).item()
            probs.append(prob_fake)

    mean_fake = sum(probs) / len(probs)
    label = "FAKE" if mean_fake >= confidence_threshold else "REAL"

    st.subheader(f"Prediction: {label}")
    st.write(f"Frames analyzed: {len(probs)}")
    st.write(f"Mean fake probability: {mean_fake:.3f}")
    st.caption(f"Positive class treated as: {pos_class}. Threshold: {confidence_threshold:.2f}")

    # Show a few sampled frames
    st.write("Sampled frames")
    thumbs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[:6]]
    st.image(thumbs, width="stretch")
else:
    st.info("Upload a video to get a prediction.")
