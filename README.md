---
title: LSTM Resume Classifier
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
pinned: false
---

# 📄 LSTM Resume Classifier

An AI-powered web application that classifies resumes into 24 different job categories using a Bidirectional LSTM neural network.

## 🚀 Deployment

This project is configured for **Streamlit Community Cloud** and **Hugging Face Spaces**.

### Local Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Features
- ✨ **Clean UI**: Built with Streamlit for a premium user experience.
- 🎯 **High Accuracy**: Bi-LSTM model trained on a diverse resume dataset.
- 📂 **Multi-format Support**: Upload PDFs or plain text files.
- 📊 **Predictions**: Shows top 5 matching categories with confidence scores.

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **Model**: PyTorch (LSTM)
- **Data Handling**: Pandas, NumPy
- **Text Extraction**: PyPDF
