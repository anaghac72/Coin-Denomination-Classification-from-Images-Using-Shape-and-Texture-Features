## Coin Denomination Classification from Images Using Shape and Texture Features
## Project Overview

This project focuses on classifying Indian coin denominations (₹1, ₹2, ₹5, ₹10, and ₹20) from coin images using Classical Machine Learning techniques without using Deep Learning.
## Team Members
|Name|Course|RegNo|
|----|------|----|
|Angel George|BioAI||
|Vishnumaya|CSDA|253013|
|Anagha C|DACS|253126|
## Dataset 
- Source: Kaggle (Indian Coin Denomination Dataset (ICDD)) 
- Contains ₹1, ₹2, ₹5, ₹10, and ₹20 coin images with different lighting conditions and viewing angles.
- Dataset link: https://www.kaggle.com/datasets/lazrus/indian-coin-denomination-dataset-icdd

## Problem Definition

This project classifies Indian coin denominations (₹1, ₹2, ₹5, ₹10, ₹20) from photographs using classical machine learning — no deep learning. Features are extracted using shape, texture, and color descriptors after segmenting the coin via Circular Hough Transform. The system is designed to work across worn, new, and differently-lit coin images.
# Features
# Image Preprocessing
- Image resizing and normalization
- Noise reduction using Gaussian Blur
- Coin segmentation using Circular Hough Transform
- Background removal
  Feature Extraction

# The following handcrafted features are extracted from each segmented coin image:

1. Diameter Ratio
- Measures normalized coin size
- Useful for differentiating denominations with different physical dimensions
2. Edge Density
- Calculated using Canny Edge Detection
- Captures relief complexity and engraving density
3. Local Binary Pattern (LBP)
- Extracts surface texture patterns
- Helps identify coin embossing and fine textures
4. Histogram of Oriented Gradients (HOG)
- Captures edge orientation and structural details
- Useful for identifying denomination-specific designs
5. HSV Color Histogram
- Differentiates coins based on metal color
- Helps separate silver-toned and bi-metallic coins
## Streamlit App Link
https://coin-denomination-classification.streamlit.app/
