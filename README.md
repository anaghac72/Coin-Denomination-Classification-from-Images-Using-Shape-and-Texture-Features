# 🪙 Indian Coin Denomination Classification

### Shape & Texture Features · Classical Machine Learning · No Deep Learning

---

# 📌 Problem Statement

Automatic coin denomination recognition is an important task in vending machines, banking systems, automated payment systems, and assistive technologies for visually impaired users.

The objective of this project is to classify Indian coin denominations (₹1, ₹2, ₹5, ₹10, and ₹20) from coin images using Classical Machine Learning techniques without using Deep Learning models.

The system must correctly identify coins under different:

- lighting conditions
- orientations
- backgrounds
- coin wear conditions

---

# 📖 Project Definition

This project classifies Indian coin denominations from photographs using handcrafted image-processing features and machine learning algorithms.

The system follows these stages:

1. Image preprocessing
2. Coin segmentation using Circular Hough Transform
3. Feature extraction using:
   - Diameter Ratio
   - Edge Density
   - Local Binary Pattern (LBP)
   - Histogram of Oriented Gradients (HOG)
   - HSV Color Histogram
4. Feature scaling using StandardScaler
5. Classification using:
   - Random Forest
   - SVM
   - KNN

The project avoids deep learning entirely and demonstrates how classical computer vision techniques can still achieve effective image classification performance.

---

# 👨‍💻 Team Members

| Name | Course | Register Number |
|---|---|---|
| Angel George | BioAI | 253203 |
| Vishnumaya | CSDA | 253013 |
| Anagha C | DACS | 253126 |

---

# 📂 Dataset

- **Dataset:** Indian Coin Denomination Dataset (ICDD)
- **Source:** Kaggle
- **Dataset Link:**  
https://www.kaggle.com/datasets/lazrus/indian-coin-denomination-dataset-icdd

## Dataset Statistics

- Total Images: **900**
- Classes: **5**
- Average Images per Class: **180**
- Train/Test Split: **80/20**


# 🧠 Feature Extraction

## 1️⃣ Diameter Ratio

- Measures normalized coin size
- Helps distinguish denominations based on physical dimensions

## 2️⃣ Edge Density

- Computed using Canny Edge Detection
- Captures engraving and relief complexity

## 3️⃣ Local Binary Pattern (LBP)

- Extracts surface texture patterns
- Useful for embossing and fine texture identification

## 4️⃣ Histogram of Oriented Gradients (HOG)

- Captures structural edge orientation
- Helps identify denomination-specific designs

## 5️⃣ HSV Color Histogram

- Differentiates metallic colors
- Useful for separating silver-toned and bi-metallic coins

---

# 🤖 Machine Learning Models Used

| Model | Accuracy |
|---|---|
| Random Forest | **70%** |
| KNN | 61% |
| SVM (RBF) | 58% |

---

# 🏆 Best Performing Model

## Random Forest

### Why it performed best

- Handles high-dimensional features effectively
- Robust against noise and outliers
- Performs well on combined shape + texture + color descriptors

---

# 📊 Random Forest Classification Report

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| ₹1 | 0.59 | 0.83 | 0.69 |
| ₹2 | 0.67 | 0.25 | 0.36 |
| ₹5 | 0.79 | 0.70 | 0.75 |
| ₹10 | 0.74 | 0.86 | 0.79 |
| ₹20 | 0.73 | 0.67 | 0.70 |

---

# 🌐 Live Demo

🚀 Streamlit Application:

[🔗 Indian Coin Denomination Classification App](https://coin-denomination-classification.streamlit.app/)

Upload any Indian coin image (`JPG`, `PNG`) and get instant denomination prediction using the deployed Random Forest model.
---

# 📸 Deployment Results

## ₹1 Coin Prediction

<img src="deployment_results/coin 1.png" width="400">

---

## ₹2 Coin Prediction

<img src="deployment_results/coin 2.png" width="400">

---

## ₹5 Coin Prediction

<img src="deployment_results/coin 5.png" width="400">

---

## ₹10 Coin Prediction

<img src="deployment_results/coin 10.png" width="400">

---

## ₹20 Coin Prediction

<img src="deployment_results/coin 20.png" width="400">

# ⚠️ Limitations

## Works Well For

- Clean dataset images
- Centered coins
- Plain backgrounds
- Good lighting conditions

## Works Less Well For

- Real-world backgrounds
- Tilted coins
- Worn or damaged coins
- Low-light images

---

# 🚀 Future Improvements

- Better segmentation techniques
- GLCM texture features
- Data augmentation
- Edge geometry descriptors
- Ensemble learning
- Multi-view coin recognition
- Real-world robustness improvements

---

# 🛠️ Technologies Used

- Python
- OpenCV
- Scikit-Learn
- NumPy
- Streamlit
- Scikit-Image
- PIL


# ✅ Conclusion

This project successfully demonstrates Indian coin denomination classification using classical machine learning and handcrafted image-processing features.

The Random Forest classifier achieved the best overall performance with **70% accuracy**, outperforming SVM and KNN.

The project proves that classical computer vision techniques can still provide effective image recognition performance without requiring deep learning models.
