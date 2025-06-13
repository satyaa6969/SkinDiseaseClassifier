import torch
from torchvision import transforms
from PIL import Image
import timm
import gradio as gr

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained ViT model
model = timm.create_model(
    "vit_tiny_patch16_224", pretrained=True, num_classes=7
)
model.load_state_dict(
    torch.load("vit_skin_disease.pth", map_location=device)
)
model.to(device)
model.eval()

# Class names (must match training)
classes = [
    'Actinic_keratoses', 'Basal_cell_carcinoma', 'Benign_keratosis',
    'Dermatofibroma', 'Melanocytic_nevi', 'Melanoma', 'Vascular_lesions'
]

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image):
    """
    Predict the skin disease class for a given image.
    Returns a dictionary of class probabilities.
    """
    # Preprocess
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, k=3)

    # Format output
    return {classes[i]: float(f"{probs[i]:.2f}") for i in top_idxs}

# Gradio interface
title = "Skin Disease Classifier (ViT)"
description = (
    "Upload a skin lesion image to classify it into one of seven categories: "
    "Actinic keratoses, Basal cell carcinoma, Benign keratosis, Dermatofibroma, "
    "Melanocytic nevi, Melanoma, or Vascular lesions."
)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions (%)"),
    title=title,
    description=description
)

if __name__ == "__main__":
    interface.launch()