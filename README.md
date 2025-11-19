# Meitei Mayek OCR System (Deep Learning) 📝

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 🚀 Project Overview
This project implements an **Optical Character Recognition (OCR)** system specifically designed for the **Meitei Mayek script** (the official script of Manipur, India). Since Meitei Mayek is a low-resource language with limited digital datasets, this project involved end-to-end pipeline development—from dataset curation and preprocessing to training a custom **Convolutional Neural Network (CNN)**.

The system achieves **94% accuracy** on handwritten character recognition, bridging the gap between physical manuscripts and digital accessibility.

**Key Highlight:** This technology has direct applications in **Document Digitization, KYC Automation, and Archival Preservation** for regional languages.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (Custom CNN Architecture)
* **Image Processing:** OpenCV (cv2), PIL
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

---

## 📊 Methodology & Pipeline

### 1. Dataset Curation & Preprocessing
Handling real-world noisy data was a primary challenge. The pipeline includes:
* **Grayscale Conversion & Binarization:** Converting raw images to binary using Otsu’s thresholding.
* **Noise Reduction:** Removing salt-and-pepper noise using Gaussian Blur.
* **Contour Detection:** Segmenting individual characters from full-page documents.
* **Normalization:** Resizing all character crops to a standard 32x32 or 64x64 pixel format.

### 2. Model Architecture (CNN)
A custom CNN was designed to capture the unique geometric strokes of Meitei Mayek:
* **Input Layer:** Processed Image
* **Conv Layers:** 3 layers of Conv2D + ReLU + MaxPool (Feature Extraction)
* **Dropout:** Implemented to prevent overfitting on the limited dataset.
* **Fully Connected Layers:** Dense layers for final classification.
* **Output:** Softmax probability distribution over the character classes.

---

## 📈 Results & Performance

* **Training Accuracy:** ~96%
* **Test Accuracy:** **94%**
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam

### Sample Predictions
> *[Insert a screenshot here showing an Input Image of a handwritten character and the Model's predicted text]*

---

## 💡 Use Cases (Why this matters?)
1.  **Fintech & Banking (KYC):** Automating the extraction of details from handwritten forms in regional languages (Meitei Mayek) for local banks.
2.  **Digital Archiving:** Preserving historical Manipuri manuscripts by converting them into editable digital text.
3.  **Education:** Enabling digital learning tools for students studying the script.

---

## 💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/AnshAttre/Meitei-Mayek-OCR.git](https://github.com/AnshAttre/Meitei-Mayek-OCR.git)
   cd Meitei-Mayek-OCR
