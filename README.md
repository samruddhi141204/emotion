#  Emotion Classification from Speech using CNN and Spectrograms

This project implements a deep learning pipeline to classify human emotions from speech audio using **Convolutional Neural Networks (CNNs)** trained on **spectrogram images**. The model identifies emotions such as **happy**, **sad**, **angry**, **calm**, and more.

---

##  Project Overview

The goal is to automatically detect the emotional state of a speaker based on their vocal patterns. The system uses spectrograms derived from audio files and trains a CNN to recognize emotion categories with high accuracy.

---

##  Dataset

- Emotion-labeled speech audio files
- Labels include: `angry`, `happy`, `sad`, `neutral`, `calm`, `fearful`, `disgust`, `surprised`
- Audio files were preprocessed into spectrograms

---

## Preprocessing Methodology

1. **Audio Loading**:  
   Used `librosa` to load `.wav`/`.mp3` files at a consistent sample rate (e.g., 22050 Hz)

2. **Spectrogram Generation**:  
   - Created **Mel-spectrograms** using `librosa.feature.melspectrogram`
   - Converted power spectrogram to decibel units with `librosa.power_to_db`

3. **Image Conversion**:  
   - Saved spectrograms as grayscale or RGB images (`.png` or `.jpg`) of size `128x128`
   - Organized images into labeled folders for training



---

## 🧪 Model Pipeline

### 🧱 CNN Architecture
- Input: 128×128 Spectrogram Images
- Layers:
  - Convolutional layers with ReLU + MaxPooling
  - Dropout regularization
  - Fully connected dense layers
  - Softmax output for multi-class classification

### 🛠 Tools & Libraries
- Python, NumPy, Pandas
- Librosa
- TensorFlow / Keras or PyTorch
- Matplotlib (for EDA & plotting)
- scikit-learn (for evaluation)

---

## 📊 Evaluation Metrics


=== Confusion Matrix ===
[[66  0  1  3  3  0  1  1]
 [ 0 56  0  1  6  4  8  0]
 [ 4  1 28  2  1  0  2  1]
 [ 5  0  0 61  4  0  5  0]
 [ 3  1  0  3 61  1  5  1]
 [ 0  0  0  1  0 32  5  0]
 [ 0  6  1 10  5  2 51  0]
 [ 0  0  1  3  5  1  0 29]]

=== Classification Report ===
              precision    recall  f1-score   support

       angry       0.85      0.88      0.86        75
        calm       0.88      0.75      0.81        75
     disgust       0.90      0.72      0.80        39
     fearful       0.73      0.81      0.77        75
       happy       0.72      0.81      0.76        75
     neutral       0.80      0.84      0.82        38
         sad       0.66      0.68      0.67        75
   surprised       0.91      0.74      0.82        39

    accuracy                           0.78       491
   macro avg       0.80      0.78      0.79       491
weighted avg       0.79      0.78      0.78       491


=== Accuracy: 78.21%
=== Macro F1 Score: 78.83%

> *Note: Metrics were computed on the test set using stratified split*







