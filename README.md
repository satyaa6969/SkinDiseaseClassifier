---
title: SkinDiseaseClassifier
emoji: 🌖
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.33.2
app_file: app.py
pinned: false
license: mit
short_description: A ViT-based web app that classifies skin diseases
---

---

# Vision Transformer (ViT) - Skin Disease Classifier 🩺

This app classifies dermatoscopic images into seven categories of skin diseases using a Vision Transformer (ViT) model fine-tuned on the HAM10000 dataset.

---

## 🧠 Model

- **Architecture:** `vit_tiny_patch16_224` from the `timm` library
- **Dataset:** HAM10000 (Human Against Machine with 10000 training images)
- **Labels:**  
  - Melanocytic nevi (nv)  
  - Melanoma (mel)  
  - Benign keratosis-like lesions (bkl)  
  - Basal cell carcinoma (bcc)  
  - Actinic keratoses (akiec)  
  - Vascular lesions (vasc)  
  - Dermatofibroma (df)  
- **Framework:** Trained and deployed using PyTorch

---

## 🔍 Usage

Upload a dermatoscopic image, and the model will predict the top 3 most likely skin conditions. This can help provide a second opinion or aid research and screening — **note: this is not a substitute for professional medical advice**.

---

## 📦 Tech Stack

- Python  
- PyTorch  
- Gradio  
- Hugging Face Spaces  
- timm (PyTorch Image Models)

---

## 📄 License

MIT License


