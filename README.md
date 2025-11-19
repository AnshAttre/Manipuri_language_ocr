# 📜 Meitei Mayek Optical Character Recognition (OCR)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **A deep learning-based OCR system designed to digitize the low-resource Meitei Mayek (Manipuri) script with 94% accuracy using Custom CNNs.**

---

## 📌 Project Overview
In the era of global digitization, many regional languages remain "low-resource," meaning they lack sufficient digital datasets for AI training. **Meitei Mayek**, the official script of Manipur, faces this challenge.

This project bridges that gap by implementing an **End-to-End OCR Pipeline**. It takes raw images of handwritten characters, processes them using computer vision techniques to remove noise, and passes them through a **Custom Convolutional Neural Network (CNN)** to classify them into digital text.

### 🎯 Key Objectives
* **Digitization:** converting historical manuscripts and handwritten forms into editable text.
* **Accuracy:** Achieving high recognition rates despite variations in handwriting styles.
* **Scalability:** Building a modular pipeline that can be extended to other Indic scripts.

---

## 🚀 Real-World Applications (Why this matters?)
While this project focuses on a specific script, the underlying technology has direct applications in the **Fintech and Banking sectors**:
1.  **Automated KYC Processing:** Extracting user details from handwritten application forms in regional languages.
2.  **Cheque Processing:** Reading amounts and names from handwritten bank cheques.
3.  **Document Archival:** Digitizing legal land records and legacy financial documents stored in physical formats.

---

## 🛠️ Technical Architecture

The system follows a 3-stage pipeline: **Preprocessing ➔ Segmentation ➔ Classification**.

### 1. Image Preprocessing (OpenCV)
Raw images are often noisy or have uneven lighting. We apply:
* **Grayscale Conversion:** Reducing computational complexity.
* **Gaussian Blurring:** To remove salt-and-pepper noise from scanned documents.
* **Adaptive Thresholding (Otsu’s Method):** To create a clean binary image (black text on white background).

### 2. Character Segmentation
* Used **Contour Detection** to identify boundaries of individual characters.
* Extracted Bounding Boxes for each character and resized them to a standard **32x32 pixel** format for the model.

### 3. The Model (Custom CNN)
We avoided heavy pre-trained models like ResNet to keep inference **fast and lightweight**.
* **Conv Layer 1:** 32 filters, 3x3 kernel, ReLU activation.
* **Conv Layer 2:** 64 filters, 3x3 kernel, ReLU + MaxPool.
* **Conv Layer 3:** 128 filters (Deep feature extraction).
* **Dropout (0.5):** To prevent overfitting on the limited dataset.
* **Fully Connected Layers:** Flattening features to map to the character classes.

---

## 📊 Performance Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Training Accuracy** | **97.2%** | On the augmented training set. |
| **Validation Accuracy** | **94.1%** | On unseen handwritten samples. |
| **Inference Time** | **~20ms** | Per character on a standard CPU. |
| **Loss Function** | CrossEntropy | Standard for multi-class classification. |

> *Note: The model demonstrates robust performance even when inputs are slightly rotated or have minor ink blots.*

---

## 💻 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/AnshAttre/Meitei-Mayek-OCR.git](https://github.com/AnshAttre/Meitei-Mayek-OCR.git)
cd Meitei-Mayek-OCR
