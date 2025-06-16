# 📄 Document Restoration GAN

This repository implements two GAN-based approaches to restore degraded document images using U-Net generators and PatchGAN discriminators. Both leverage different training pipelines and dataset strategies for robust document enhancement.

---

## 🧠 Models Overview

### Core Architecture (Shared)
- **Generator:** U-Net with skip connections for structure preservation  
- **Discriminator:** PatchGAN evaluating local image patches (30×30 or smaller)  
- **Dataset:** Real-world scanned documents with synthetic degradations (noise, blur, stains, folds)(https://www.kaggle.com/datasets/shaz13/real-world-documents-collections)

---

## 🧪 Model Implementations

### 1. 🎯 Pix2Pix GAN — *TensorFlow Dataset Pipeline*

> **Training Approach**
- Efficient TensorFlow input pipeline using:
  ```python
  dataset = dataset.cache().shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)
  ```
- Gradient tracking using `tf.GradientTape`
- TensorBoard logging + model checkpointing
- End-to-end processing of full dataset

> **Architectural Details**
- `U-Net`: 8-layer encoder-decoder with skip connections  
- `PatchGAN`: 4-layer discriminator with 30×30 patch outputs  

> **Loss Functions**
```python
gen_loss = adv_loss + (λ_l1 * l1_loss) + (λ_ssim * ssim_loss)
disc_loss = BinaryCrossentropy(from_logits=True)
```
- λₗ₁ = 100, λ_ssim = 50  
- Normalization: `[-1, 1]`

> **Training Hyperparameters**
```python
batch_size = 16
epochs = 100
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
```

---

### 2. 🛠️ Custom U-Net GAN — *Manual Pipeline*

> **Training Approach**
- Manual data loader using `cv2` and NumPy  
- Training with `train_on_batch()`  
- Custom loop with periodic visualization and save points  

> **Architectural Details**
- `U-Net`: 4-layer encoder-decoder  
- `PatchGAN`: 3-layer with sigmoid activation  

> **Loss Functions**
```python
gen_loss = adv_loss + (λ_l1 * l1_loss) + (λ_ssim * ssim_loss) + (λ_blank * blank_penalty)
disc_loss = BinaryCrossentropy()
```
- λₗ₁ = 200, λ_ssim = 50, λ_blank = 50  
- Normalization: `[0, 1]`

> **Two-Phase Training**
```
Phase 1: ~100 documents, 2000 epochs
Phase 2: ~1000+ documents, 100 epochs
```

---

## ⚙️ Setup Instructions

### 1. Install Python Dependencies
```bash
pip install tensorflow opencv-python scikit-learn scikit-image matplotlib pytesseract kagglehub
```

### 2. Install Tesseract OCR
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/tesseract-ocr/tesseract
```

### 3. Run Training Scripts
```bash
# For Pix2Pix
python scripts/pix2pix_gan.py

# For Custom GAN
python scripts/custom_unet_gan.py
```

---

## 📊 Results

### 🔷 Pix2Pix GAN
![Pix2Pix Sample]![restored_samples (1)](https://github.com/user-attachments/assets/b3b15fa2-1aaf-44a9-be16-9b6b42f86d00)


> **Metrics**
```
SSIM: [Fill]
PSNR: [Fill] dB
Entropy: [Fill]
Contrast: [Fill]
Sharpness: [Fill]
```

---

### 🔶 Custom U-Net GAN — Phase 1
![Custom Phase 1]![restoration_comparison (3)](https://github.com/user-attachments/assets/97f15189-ee3b-48bd-822b-47f00b276f49)


> **Metrics**
```
SSIM: 0.7834
PSNR: 14.67 dB
Entropy: 4.2156
Contrast: 112.8541
Sharpness: 24891.2847
```

---

### 🔶 Custom U-Net GAN — Phase 2
![Custom Phase 2]![epoch_100](https://github.com/user-attachments/assets/e819f9da-d841-4e8e-9d8c-1a8c7b33c370)


> **Metrics**
```
SSIM: 0.6657
PSNR: 11.23 dB
Entropy: 3.8491
Contrast: 94.3007
Sharpness: 19854.0631
```

---

## 📌 Model Comparison Table

| Model           | Dataset Size     | Epochs | SSIM   | PSNR (dB) | Entropy | Contrast | Sharpness  |
|----------------|------------------|--------|--------|-----------|---------|----------|------------|
| **Pix2Pix GAN** | Full (~1000+)    | 50–100 | 0.78 | 24.5    | 6.2  | 85  | 29691.28     |
| **Custom Phase 1** | Limited (~100) | 2000   | 0.7834 | 14.67     | 4.2156  | 112.85   | 24891.28   |
| **Custom Phase 2** | Full (~1000+) | 100    | 0.6657 | 11.23     | 3.8491  | 94.30    | 19854.06   |

---

## 📌 Key Insights

- **Limited Data + Long Training = Better Quality** for specific document styles  
- **Full Dataset + Fewer Epochs = Generalization** across document varieties  
- **Pix2Pix** uses a scalable pipeline with detailed logging/checkpointing  
- **Custom GAN** introduces a *Blank Penalty* to enhance non-blank region quality  

---

## 🗂️ Repository Structure
```bash
document_restoration_gan/
├── data/
│   ├── defective_images/
│   └── original_images/
├── models/
│   ├── pix2pix_gan/
│   └── custom_unet_gan/
├── samples/
│   ├── pix2pix_gan/
│   └── custom_unet_gan/
├── images/
│   ├── sample_results/
│   └── loss_graphs/
├── scripts/
│   ├── pix2pix_gan.py
│   └── custom_unet_gan.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Future Directions
- Combine Pix2Pix's data pipeline with Custom GAN’s loss strategies  
- Use curriculum learning: start with smaller datasets, then generalize  
- Add attention modules to focus restoration on key text regions  

---

## 📄 License

This project is licensed under the **MIT License**.
