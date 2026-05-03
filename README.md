# 🧠 Brain Tumor Detection

This project aims to classify brain MRI images into four categories:
glioma, meningioma, pituitary tumor, and no tumor.

## Approach
- CNN (MobileNetV2) is used for feature extraction
- Ensemble models are used for classification:
  - Random Forest
  - Gradient Boosting

## 📊 Evaluation
Models are compared using:
- Accuracy
- Precision
- Recall
- F1-score

## 📁 Dataset
Brain MRI Dataset:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Run the App

```bash
streamlit run app.py

👨‍💻 Author

Moataz Nageh
