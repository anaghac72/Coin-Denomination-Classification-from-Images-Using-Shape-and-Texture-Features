## Coin Denomination Classification from Images Using Shape and Texture Features
## Project Overview

This project focuses on classifying Indian coin denominations (₹1, ₹2, ₹5, ₹10, and ₹20) from coin images using Classical Machine Learning techniques without using Deep Learning.
## Team Members
1. Angel George 
2. Vishnumaya
3. Anagha C
## Dataset 
- Source: Kaggle (Indian Coin Denomination Dataset (ICDD)) 
- Contains ₹1, ₹2, ₹5, ₹10, and ₹20 coin images with different lighting conditions and viewing angles.
- Dataset link: https://www.kaggle.com/datasets/lazrus/indian-coin-denomination-dataset-icdd

Stage 1: Problem Definition & Literature Review
1.1 Problem Statement
The goal is to build a classical machine learning pipeline (no deep learning) that can identify Indian coin denominations — ₹1, ₹2, ₹5, ₹10, and ₹20 — from photographs. The system must work reliably across:

Coin condition: brand new, moderately worn, heavily worn
Lighting: bright daylight, dim indoor, harsh flash, shadow-cast
Orientation: any rotation or slight tilt

This is a 5-class image classification problem, but the feature extraction is constrained to handcrafted geometric, texture, and color features — making it a study in classical computer vision.
